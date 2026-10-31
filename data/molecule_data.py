"""
Molecule datasets for ULD-vs-LDM experiments.

SmilesDataset: a generic fixed-length one-hot SMILES dataset that reads
from any CSV with a SMILES column. Each SMILES is truncated / padded to
``max_len`` and one-hot encoded over a vocabulary built from the actual
data; we flatten to a single vector of dimension ``max_len * vocab_size``
and treat generation as continuous vector diffusion, with argmax decoding
at inference. Validity is checked with RDKit.

Instantiate it for any CSV:

    ds = SmilesDataset(csv_path='/path/to/x.csv', smiles_col='smiles', max_len=64)

The QM9SmilesDataset below is a thin wrapper that fixes the QM9 defaults
for backward compatibility.

This setup lets us reuse the existing TwoStageManifoldDiffusion (MLP
mode, image_shape=None) and LatentDiffusionCNN (MLP mode) classes
without any architectural changes. The same fair-comparison story as
Stacked MNIST applies: identical encoder/decoder/score-net; only the
Stage 1 loss differs between ULD and LDM.
"""

import csv
import os
import torch
import numpy as np


# ----------------------------------------------------------------------
# Validity-aware SMILES repair (used by evaluate_samples(repair=True))
# ----------------------------------------------------------------------

# Single-letter organic-subset atoms the SMILES grammar allows without brackets:
_BARE_ATOMS = set('BCNOFPSIH')
_AROMATIC = set('cnops')
# Character subset we consider well-formed as a prefix of any SMILES
_ALLOWED = set('BCNOFPSIHclnopsri' + '()[]=#-+/\\.'
               + '0123456789%@')


def compute_internal_diversity(mols, radius=2, n_bits=1024, max_n=2000, seed=0):
    """
    Internal diversity = 1 - mean pairwise Tanimoto similarity over
    Morgan fingerprints of a molecule set.

    Parameters
    ----------
    mols : list of RDKit ``Mol`` objects
    radius : int
        Morgan fingerprint radius (ECFP4 = radius 2)
    n_bits : int
        Fingerprint size
    max_n : int
        Subsample the input to at most this many molecules (pairwise
        Tanimoto is O(n^2), so this bounds the cost). 2000 is plenty
        for a stable estimate.

    Returns
    -------
    float in [0, 1], higher = more diverse.
    """
    from rdkit.Chem import AllChem, DataStructs
    import random
    if len(mols) < 2:
        return 0.0
    if len(mols) > max_n:
        rng = random.Random(seed)
        mols = rng.sample(mols, max_n)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
           for m in mols]
    n = len(fps)
    total = 0.0
    count = 0
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        total += sum(sims)
        count += len(sims)
    if count == 0:
        return 0.0
    mean_sim = total / count
    return float(1.0 - mean_sim)


def compute_qed_stats(mols):
    """
    QED (Bickerton et al. 2012): quantitative drug-likeness score.

    Computed via RDKit's ``Chem.QED.qed``. Returns mean and std over
    valid molecules. QED is in [0, 1]; higher = more drug-like.
    Training molecules on QM9 score ~0.47; on drug-like datasets
    (HIV, MUV, PCBA) they typically score 0.55-0.75.
    """
    from rdkit.Chem import QED
    import math
    if len(mols) == 0:
        return {'qed_mean': 0.0, 'qed_std': 0.0, 'qed_n': 0}
    vals = []
    for m in mols:
        try:
            vals.append(QED.qed(m))
        except Exception:
            pass
    if len(vals) == 0:
        return {'qed_mean': 0.0, 'qed_std': 0.0, 'qed_n': 0}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {'qed_mean': float(mean),
            'qed_std':  float(math.sqrt(var)),
            'qed_n':    len(vals)}


def compute_fcd_score(gen_smiles, ref_smiles, device=None, n_jobs=1):
    """
    Fréchet ChemNet Distance (FCD; Preuer et al. 2018) — analogous to
    FID for molecules. Uses features from a ChemNet model pretrained
    on ZINC. Lower = generated distribution closer to reference.

    Typical values:
      * FCD(train, train_held_out) ~ 0.1 - 1.0
      * strong models ~ 1 - 5
      * weak / noisy distributions ~ 10+

    Parameters
    ----------
    gen_smiles : list of str
        Canonical SMILES of generated valid molecules.
    ref_smiles : list of str
        Canonical SMILES of reference (typically training) molecules.
    device : torch device or None
        If None, uses cuda if available else cpu.
    n_jobs : int
        Parallel workers for feature extraction.
    """
    try:
        from fcd_torch import FCD
    except ImportError:
        return float('nan')
    if len(gen_smiles) == 0 or len(ref_smiles) == 0:
        return float('nan')
    import torch
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        fcd = FCD(device=str(device), n_jobs=n_jobs)
        return float(fcd(list(gen_smiles), list(ref_smiles)))
    except Exception as e:
        print(f"  [warn] FCD failed: {type(e).__name__}: {e}")
        return float('nan')


def _build_constrained_masker(vocab):
    """
    Helper: returns a function that, given the parser state and remaining
    positions, produces a boolean mask (length vocab_size) indicating
    which tokens are legal at the current decoding step.

    State tracked:
      - paren_depth:  number of unclosed '('
      - open_rings:   set of digit *indices* currently open
      - prev_idx:     the last emitted token (used to rule out e.g.
                      bond-after-bond which is invalid SMILES)

    The masker is intentionally conservative: it only enforces
    paren/ring balance and the "don't open new structures if there's
    no room to close them" rule, plus a handful of trivial rules like
    "no ')' when depth==0". This is not a full SMILES parser; it
    catches the dominant failure modes observed in continuous decoding
    while leaving enough freedom that the model's learned distribution
    over valid chemistry still drives the generation.
    """
    import numpy as np
    char2idx = {c: i for i, c in enumerate(vocab)}
    pad = char2idx.get(' ', 0)
    lpar = char2idx.get('(', -1)
    rpar = char2idx.get(')', -1)
    digit_indices = [char2idx[c] for c in '0123456789' if c in char2idx]
    # Bond / connective chars: forbid "bond after bond"
    bond_indices = [char2idx[c] for c in '=#-+/\\' if c in char2idx]
    bond_set = set(bond_indices)

    V = len(vocab)

    def is_closing_token(i, open_rings):
        return i == rpar or i in open_rings

    def make_mask(paren_depth, open_rings, prev_idx, remaining):
        mask = np.ones(V, dtype=bool)
        closures_needed = paren_depth + len(open_rings)

        # 1) No ')' when we have nothing to close
        if rpar >= 0 and paren_depth == 0:
            mask[rpar] = False

        # 2) No '(' if there's no room for the extra closure it creates
        #    (need at least 1 more slot for the matching ')')
        if lpar >= 0 and remaining < closures_needed + 2:
            mask[lpar] = False

        # 3) No NEW ring open if no room for its closing digit
        for d in digit_indices:
            if d not in open_rings and remaining < closures_needed + 2:
                mask[d] = False

        # 4) PAD is only valid when everything is balanced
        if closures_needed > 0:
            mask[pad] = False

        # 5) Two consecutive bond symbols are invalid SMILES
        if prev_idx is not None and prev_idx in bond_set:
            for b in bond_indices:
                mask[b] = False

        # 6) HARD CONSTRAINT: if closures_needed >= remaining, we must
        #    spend every remaining slot on a closing token.
        if closures_needed >= remaining:
            close_mask = np.zeros(V, dtype=bool)
            if paren_depth > 0 and rpar >= 0:
                close_mask[rpar] = True
            for d in open_rings:
                close_mask[d] = True
            mask = close_mask

        # Safety net: if nothing is legal (shouldn't happen with the
        # rules above), fall back to allowing PAD so we stop cleanly.
        if not mask.any():
            mask = np.zeros(V, dtype=bool)
            mask[pad] = True
        return mask

    return {
        'pad': pad, 'lpar': lpar, 'rpar': rpar,
        'digit_indices': set(digit_indices),
        'bond_set': bond_set,
        'make_mask': make_mask,
    }


def decode_onehot_constrained(x, vocab):
    """
    Grammar-constrained left-to-right decoding of a (B, L, V) float
    tensor (or equivalent (B, L*V) flat tensor). At each position we
    build a validity mask from the parser state and pick the argmax
    OVER valid tokens. Forces parens and rings to close when the
    remaining positions equal the number of unclosed structures.

    Returns a list of B strings.
    """
    import numpy as np
    import torch
    if x.dim() == 2:
        L = len(vocab); V = L  # placeholders, overwritten below
        V = len(vocab)
        L = x.shape[1] // V
        x = x.view(-1, L, V)
    B, L, V = x.shape
    scores = x.detach().cpu().numpy()

    m = _build_constrained_masker(vocab)
    pad = m['pad']
    lpar = m['lpar']
    rpar = m['rpar']
    digit_indices = m['digit_indices']
    make_mask = m['make_mask']

    NEG_INF = -1e30
    results = []
    for b in range(B):
        paren_depth = 0
        open_rings = set()
        prev_idx = None
        out = []
        for i in range(L):
            remaining = L - i
            mask = make_mask(paren_depth, open_rings, prev_idx, remaining)
            masked = scores[b, i].copy()
            masked[~mask] = NEG_INF
            tok = int(np.argmax(masked))

            if tok == pad:
                break
            if tok == lpar:
                paren_depth += 1
            elif tok == rpar:
                paren_depth -= 1
            elif tok in digit_indices:
                if tok in open_rings:
                    open_rings.discard(tok)
                else:
                    open_rings.add(tok)
            prev_idx = tok
            out.append(vocab[tok])

        # Hard-close any residual structure (shouldn't happen if the
        # forced-close branch fires correctly, but belt-and-braces)
        while paren_depth > 0:
            out.append(')')
            paren_depth -= 1
        for d in sorted(open_rings):
            out.append(vocab[d])

        results.append(''.join(out))
    return results


def rdkit_longest_valid_prefix(raw):
    """
    Greedy longest-valid-prefix repair: try RDKit.MolFromSmiles on the
    full string; if invalid, strip one char from the end and retry
    until a valid prefix is found or the string is empty.

    This is a stronger repair than ``repair_smiles`` because it uses
    the real RDKit parser as the oracle (catches valence errors and
    other chemistry constraints my syntactic repair misses). Cost:
    O(L) RDKit parses per input, so O(L * n_samples) total. For
    L=32 and n=10k that's ~300k RDKit calls, about 5-15 seconds.

    Apply the SAME strategy to every method being compared for
    fairness. Typically pushes repaired validity from ~90% to ~98%+
    at the cost of occasionally returning shorter molecules.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    # First strip PAD and anything past the first PAD
    s = raw
    if ' ' in s:
        s = s.split(' ', 1)[0]
    if not s:
        return ''
    # Try the full string first
    m = Chem.MolFromSmiles(s)
    if m is not None:
        return s
    # Otherwise shrink from the right until valid
    for end in range(len(s) - 1, 0, -1):
        m = Chem.MolFromSmiles(s[:end])
        if m is not None:
            return s[:end]
    return ''


def repair_smiles(raw):
    """
    Validity-aware repair of a decoded SMILES string.

    Strategy: return the LONGEST PREFIX of ``raw`` where every ring label
    and parenthesis is fully balanced. Any such prefix is a plausible
    atomic SMILES (e.g. ``C1CCNC1`` is a balanced prefix of
    ``C1CCNC1OCCOC1``, which has an unclosed third ``1``).

    Walks the string left-to-right tracking paren depth and ring-label
    parity. At every position where everything is balanced, we record
    the length. At the end we return the prefix of that length,
    trimmed of any trailing dangling bond character (``= # - + / \\``).

    This is NOT a real SMILES parser and does NOT guarantee chemical
    validity — but on the character-level failure modes produced by
    continuous diffusion (off-by-one brackets, unclosed rings,
    truncation), it substantially raises the RDKit parse rate. Apply
    the SAME repair to all methods being compared.
    """
    out = []
    paren_depth = 0
    ring_state = {}   # digit → 'open' / 'closed'
    last_balanced = 0  # length of longest balanced prefix

    for ch in raw:
        if ch == ' ':   # PAD reached — end of molecule
            break
        if ch not in _ALLOWED:
            break       # illegal char — stop
        if ch == '(':
            paren_depth += 1
            out.append(ch)
        elif ch == ')':
            if paren_depth == 0:
                break
            paren_depth -= 1
            out.append(ch)
        elif ch.isdigit():
            state = ring_state.get(ch)
            if state is None or state == 'closed':
                # First use or label reused after a previous pair
                ring_state[ch] = 'open'
            else:  # 'open'
                ring_state[ch] = 'closed'
            out.append(ch)
        else:
            out.append(ch)

        if paren_depth == 0 and all(s != 'open' for s in ring_state.values()):
            last_balanced = len(out)

    result = out[:last_balanced]
    # Strip a trailing bond char that would leave an incomplete bond
    while result and result[-1] in '=#-+/\\':
        result.pop()
    return ''.join(result)


class SmilesDataset:
    """
    Generic fixed-length one-hot SMILES dataset.

    Parameters
    ----------
    csv_path : str
        CSV file with at least a SMILES column.
    smiles_col : str
        Name of the SMILES column in the CSV (default 'smiles').
    max_len : int
        Truncate/pad SMILES to this length. Molecules longer than
        ``max_len`` are dropped (skip_long=True) so we never have to
        truncate mid-ring.
    max_samples : int or None
        Cap on dataset size (for quick tests). None = use all.
    """

    PAD_CHAR = ' '  # outside every natural SMILES vocab

    def __init__(self, csv_path, smiles_col='smiles',
                 max_len=32, max_samples=None, seed=42,
                 skip_long=True, name=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")
        self.max_len = max_len
        self.csv_path = csv_path
        self.smiles_col = smiles_col
        self.seed = seed
        self.name = name or os.path.basename(csv_path)

        # --- Load SMILES from CSV ---
        smiles = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            if smiles_col not in reader.fieldnames:
                raise ValueError(
                    f"Column {smiles_col!r} not in CSV; "
                    f"available: {reader.fieldnames}"
                )
            for row in reader:
                s = row[smiles_col]
                if not s:
                    continue
                if len(s) > max_len:
                    if skip_long:
                        continue
                    s = s[:max_len]
                smiles.append(s)
                if max_samples is not None and len(smiles) >= max_samples:
                    break

        if len(smiles) == 0:
            raise RuntimeError(
                f"No SMILES loaded from {csv_path} "
                f"(smiles_col={smiles_col!r}, max_len={max_len})"
            )

        # --- Build vocab from the actual data ---
        chars = set()
        for s in smiles:
            chars.update(s)
        chars.discard(self.PAD_CHAR)
        vocab = [self.PAD_CHAR] + sorted(chars)
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = vocab

        self.d = max_len * self.vocab_size
        self.n_total = len(smiles)
        self.training_smiles = smiles  # keep for novelty check

        # --- One-hot tensor: (N, max_len, vocab_size) ---
        one_hot = torch.zeros(self.n_total, max_len, self.vocab_size,
                              dtype=torch.float32)
        for i, s in enumerate(smiles):
            for j in range(max_len):
                if j < len(s):
                    one_hot[i, j, self.char2idx[s[j]]] = 1.0
                else:
                    one_hot[i, j, 0] = 1.0  # PAD
        self.data_onehot = one_hot
        self.data = one_hot.view(self.n_total, self.d)  # flat view
        self.labels = torch.zeros(self.n_total, dtype=torch.long)

        # PCA for diagnostics (computed on a subset for speed)
        self._compute_pca()

        print(f"  {type(self).__name__} ({self.name}) loaded: {self.n_total} molecules, "
              f"max_len={max_len}, vocab={self.vocab_size} "
              f"({' '.join(repr(c) for c in vocab[:10])}...), "
              f"d={self.d}")


    def _compute_pca(self):
        mean = self.data.mean(dim=0)
        centered = self.data - mean
        n_sub = min(5000, self.n_total)
        U, S, Vt = torch.linalg.svd(centered[:n_sub], full_matrices=False)
        self.pca_components = Vt
        self.pca_singular_values = S
        self.pca_mean = mean
        self.pca_explained_variance = (S ** 2) / (n_sub - 1)

    # ------------------------------------------------------------------
    # Sampling interface
    # ------------------------------------------------------------------

    def sample_clean(self, n):
        idx = torch.randint(0, self.n_total, (n,))
        return self.data[idx], self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        return x0 + sigma_t * eps, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)

    # ------------------------------------------------------------------
    # Decoding + validity
    # ------------------------------------------------------------------

    def decode_onehot(self, x_flat):
        """
        Decode a (N, d) or (N, max_len, vocab_size) tensor to a list of
        SMILES strings. Argmax over the vocab axis; PAD tokens are
        stripped from the end.
        """
        if x_flat.dim() == 2:
            x = x_flat.view(-1, self.max_len, self.vocab_size)
        else:
            x = x_flat
        idx = x.argmax(dim=-1).cpu().numpy()  # (N, max_len)
        out = []
        for row in idx:
            s = ''.join(self.idx2char[c] for c in row if c != 0)
            out.append(s)
        return out

    def evaluate_samples(self, x_flat, training_set=None,
                         repair=False, constrained=False,
                         decode_mode=None,
                         compute_diversity=False,
                         compute_qed=False,
                         compute_fcd=False,
                         fcd_reference_max=10000,
                         diversity_max_n=2000):
        """
        Validity, uniqueness, novelty, and average heavy-atom count of
        a batch of generated samples.

        Parameters
        ----------
        decode_mode : str or None
            If set, overrides ``repair`` and ``constrained`` with one of:
              'raw'       — argmax decode, no repair
              'repair'    — argmax + longest-balanced-prefix syntactic repair
              'constrained' — constrained argmax (parser-state masking)
              'rdkit_longest' — greedy longest prefix accepted by RDKit
        repair, constrained :
            Legacy kwargs, kept for backward compatibility.
        compute_diversity : bool
            If True, also compute internal diversity (1 - mean pairwise
            Tanimoto similarity over Morgan fingerprints) over valid
            molecules, subsampled to ``diversity_max_n`` for tractable
            O(n^2) cost.
        compute_qed : bool
            If True, also compute mean and std of RDKit's QED drug-
            likeness score over valid molecules.
        compute_fcd : bool
            If True, also compute Fréchet ChemNet Distance between
            valid generated molecules and the training set (capped
            at ``fcd_reference_max`` reference samples for speed).
            Requires ``fcd_torch``; returns NaN if unavailable.
        """
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')

        # Normalize to decode_mode
        if decode_mode is None:
            if constrained:
                decode_mode = 'constrained'
            elif repair:
                decode_mode = 'repair'
            else:
                decode_mode = 'raw'

        if decode_mode == 'raw':
            smis = self.decode_onehot(x_flat)
        elif decode_mode == 'repair':
            smis = [repair_smiles(s) for s in self.decode_onehot(x_flat)]
        elif decode_mode == 'constrained':
            smis = decode_onehot_constrained(x_flat, self.vocab)
        elif decode_mode == 'rdkit_longest':
            # Start from raw argmax decoding then apply RDKit-aware shrink
            smis = [rdkit_longest_valid_prefix(s)
                    for s in self.decode_onehot(x_flat)]
        else:
            raise ValueError(f"Unknown decode_mode={decode_mode!r}")

        n_total = len(smis)
        if n_total == 0:
            return {'validity': 0.0, 'uniqueness': 0.0, 'novelty': 0.0,
                    'n_total': 0, 'n_valid': 0, 'n_unique': 0, 'n_novel': 0,
                    'avg_heavy_atoms_valid': 0.0,
                    'decode_mode': decode_mode}

        canonical = []
        valid_mols = []
        heavy_atoms = []
        for s in smis:
            if not s:
                continue
            try:
                m = Chem.MolFromSmiles(s)
                if m is None:
                    continue
                canonical.append(Chem.MolToSmiles(m))
                valid_mols.append(m)
                heavy_atoms.append(m.GetNumHeavyAtoms())
            except Exception:
                continue
        n_valid = len(canonical)
        unique = set(canonical)
        n_unique = len(unique)

        if training_set is None:
            training_set = set(self.training_smiles)
        # Canonicalize training set once (cached)
        if not hasattr(self, '_training_canonical'):
            tc = set()
            for s in self.training_smiles:
                try:
                    m = Chem.MolFromSmiles(s)
                    if m is not None:
                        tc.add(Chem.MolToSmiles(m))
                except Exception:
                    pass
            self._training_canonical = tc
        n_novel = sum(1 for s in unique if s not in self._training_canonical)

        avg_ha = (sum(heavy_atoms) / len(heavy_atoms)) if heavy_atoms else 0.0

        out = {
            'validity':   n_valid / n_total,
            'uniqueness': (n_unique / n_valid) if n_valid > 0 else 0.0,
            'novelty':    (n_novel / n_unique) if n_unique > 0 else 0.0,
            'novel_over_valid': (n_novel / n_valid) if n_valid > 0 else 0.0,
            'n_total':  n_total,
            'n_valid':  n_valid,
            'n_unique': n_unique,
            'n_novel':  n_novel,
            'avg_heavy_atoms_valid': avg_ha,
            'decode_mode': decode_mode,
        }

        # ---- Optional diversity / drug-likeness / FCD ----
        if compute_diversity and n_valid >= 2:
            out['internal_diversity'] = compute_internal_diversity(
                valid_mols, max_n=diversity_max_n)
        else:
            out['internal_diversity'] = None

        if compute_qed and n_valid > 0:
            qed = compute_qed_stats(valid_mols)
            out['qed_mean'] = qed['qed_mean']
            out['qed_std'] = qed['qed_std']
            out['qed_n']   = qed['qed_n']
        else:
            out['qed_mean'] = None
            out['qed_std'] = None
            out['qed_n']   = 0

        if compute_fcd and n_valid > 0:
            # Canonicalize + cap the reference set for tractable cost
            if not hasattr(self, '_training_canonical_list'):
                self._training_canonical_list = sorted(self._training_canonical)
            ref = self._training_canonical_list
            if fcd_reference_max and len(ref) > fcd_reference_max:
                ref = ref[:fcd_reference_max]
            out['fcd'] = compute_fcd_score(canonical, ref)
        else:
            out['fcd'] = None

        return out


class QM9SmilesDataset(SmilesDataset):
    """QM9 defaults — 133k small organic molecules, max_len=32, vocab~21."""

    def __init__(self, csv_path='/home/weihuang/data/molecules/qm9.csv',
                 max_len=32, max_samples=None, seed=42):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"QM9 CSV not found at {csv_path}. "
                "Download with:\n  curl -o /home/weihuang/data/molecules/qm9.csv "
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
            )
        super().__init__(csv_path=csv_path, smiles_col='smiles',
                         max_len=max_len, max_samples=max_samples,
                         seed=seed, name='QM9')


class SelfiesDataset:
    """
    SELFIES tokenization of a SMILES CSV — 100% validity by construction.

    Each SMILES is converted to SELFIES (Krenn et al. 2020), then
    split into tokens, padded to ``max_len``, and one-hot encoded.
    At decode time, argmax → token list → sf.decoder() → canonical
    SMILES → RDKit. Because SELFIES is a surjective map onto valid
    molecules, EVERY decoded string produces a chemically valid
    molecule — the validity problem disappears entirely.

    Parameters
    ----------
    csv_path : str
        CSV file with a SMILES column.
    smiles_col : str
        Name of the SMILES column.
    max_len : int
        Maximum SELFIES token length (molecules longer are dropped).
    max_samples : int or None
        Cap on dataset size.
    """

    PAD_TOKEN = '[nop]'

    def __init__(self, csv_path, smiles_col='smiles',
                 max_len=24, max_samples=None, seed=42, name=None):
        import selfies as sf
        self.sf = sf

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")
        self.max_len = max_len
        self.csv_path = csv_path
        self.seed = seed
        self.name = name or os.path.basename(csv_path)

        # --- Load + convert SMILES → SELFIES ---
        selfies_list = []
        smiles_list = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = row.get(smiles_col, '')
                if not smi:
                    continue
                try:
                    sel = sf.encoder(smi)
                    tokens = list(sf.split_selfies(sel))
                    if len(tokens) <= max_len:
                        selfies_list.append(tokens)
                        smiles_list.append(smi)
                except Exception:
                    continue
                if max_samples is not None and len(selfies_list) >= max_samples:
                    break

        if len(selfies_list) == 0:
            raise RuntimeError(f"No SELFIES loaded from {csv_path}")

        # --- Build vocab from the actual data ---
        all_tokens = set()
        for toks in selfies_list:
            all_tokens.update(toks)
        all_tokens.discard(self.PAD_TOKEN)
        vocab = [self.PAD_TOKEN] + sorted(all_tokens)
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = vocab

        self.d = max_len * self.vocab_size
        self.n_total = len(selfies_list)
        self.training_smiles = smiles_list

        # --- One-hot encode ---
        one_hot = torch.zeros(self.n_total, max_len, self.vocab_size,
                              dtype=torch.float32)
        for i, toks in enumerate(selfies_list):
            for j in range(max_len):
                if j < len(toks) and toks[j] in self.char2idx:
                    one_hot[i, j, self.char2idx[toks[j]]] = 1.0
                else:
                    one_hot[i, j, 0] = 1.0  # PAD
        self.data_onehot = one_hot
        self.data = one_hot.view(self.n_total, self.d)
        self.labels = torch.zeros(self.n_total, dtype=torch.long)

        self._compute_pca()
        print(f"  SelfiesDataset ({self.name}) loaded: {self.n_total} molecules, "
              f"max_len={max_len}, vocab={self.vocab_size}, d={self.d}")

    def _compute_pca(self):
        mean = self.data.mean(dim=0)
        centered = self.data - mean
        n_sub = min(5000, self.n_total)
        U, S, Vt = torch.linalg.svd(centered[:n_sub], full_matrices=False)
        self.pca_components = Vt
        self.pca_singular_values = S
        self.pca_mean = mean
        self.pca_explained_variance = (S ** 2) / (n_sub - 1)

    def sample_clean(self, n):
        idx = torch.randint(0, self.n_total, (n,))
        return self.data[idx], self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        return x0 + sigma_t * eps, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)

    def decode_onehot(self, x_flat):
        """Argmax → SELFIES tokens → sf.decoder() → canonical SMILES."""
        import selfies as sf
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')

        if x_flat.dim() == 2:
            x = x_flat.view(-1, self.max_len, self.vocab_size)
        else:
            x = x_flat
        idx = x.argmax(dim=-1).cpu().numpy()
        out = []
        for row in idx:
            tokens = [self.idx2char[c] for c in row if c != 0]
            sel_str = ''.join(tokens)
            try:
                smi = sf.decoder(sel_str)
                if smi:
                    m = Chem.MolFromSmiles(smi)
                    if m is not None:
                        out.append(Chem.MolToSmiles(m))
                        continue
            except Exception:
                pass
            out.append(sel_str)  # fallback: raw SELFIES string
        return out

    def evaluate_samples(self, x_flat, training_set=None,
                         compute_diversity=False,
                         compute_qed=False,
                         compute_fcd=False,
                         fcd_reference_max=10000,
                         diversity_max_n=2000, **kwargs):
        """
        Evaluate generated samples. Validity should be ~100% by
        construction (SELFIES guarantees it). All extended metrics
        (FCD, QED, IntDiv) computed on the valid decoded molecules.
        """
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        import selfies as sf

        # Decode: argmax → SELFIES → SMILES → RDKit Mol
        if x_flat.dim() == 2:
            x = x_flat.view(-1, self.max_len, self.vocab_size)
        else:
            x = x_flat
        idx = x.argmax(dim=-1).cpu().numpy()

        canonical = []
        valid_mols = []
        heavy_atoms = []
        n_total = idx.shape[0]
        for row in idx:
            tokens = [self.idx2char[c] for c in row if c != 0]
            sel_str = ''.join(tokens)
            try:
                smi = sf.decoder(sel_str)
                if not smi:
                    continue
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                canonical.append(Chem.MolToSmiles(m))
                valid_mols.append(m)
                heavy_atoms.append(m.GetNumHeavyAtoms())
            except Exception:
                continue

        n_valid = len(canonical)
        unique = set(canonical)
        n_unique = len(unique)

        # Novelty
        if not hasattr(self, '_training_canonical'):
            tc = set()
            for s in self.training_smiles:
                try:
                    m = Chem.MolFromSmiles(s)
                    if m is not None:
                        tc.add(Chem.MolToSmiles(m))
                except Exception:
                    pass
            self._training_canonical = tc
            self._training_canonical_list = sorted(tc)
        n_novel = sum(1 for s in unique if s not in self._training_canonical)

        avg_ha = (sum(heavy_atoms) / len(heavy_atoms)) if heavy_atoms else 0.0

        out = {
            'validity':   n_valid / n_total if n_total > 0 else 0.0,
            'uniqueness': (n_unique / n_valid) if n_valid > 0 else 0.0,
            'novelty':    (n_novel / n_unique) if n_unique > 0 else 0.0,
            'novel_over_valid': (n_novel / n_valid) if n_valid > 0 else 0.0,
            'n_total':  n_total,
            'n_valid':  n_valid,
            'n_unique': n_unique,
            'n_novel':  n_novel,
            'avg_heavy_atoms_valid': avg_ha,
            'decode_mode': 'selfies',
        }

        if compute_diversity and n_valid >= 2:
            out['internal_diversity'] = compute_internal_diversity(
                valid_mols, max_n=diversity_max_n)
        else:
            out['internal_diversity'] = None

        if compute_qed and n_valid > 0:
            qed = compute_qed_stats(valid_mols)
            out.update(qed)
        else:
            out['qed_mean'] = None; out['qed_std'] = None; out['qed_n'] = 0

        if compute_fcd and n_valid > 0:
            ref = self._training_canonical_list
            if fcd_reference_max and len(ref) > fcd_reference_max:
                ref = ref[:fcd_reference_max]
            out['fcd'] = compute_fcd_score(canonical, ref)
        else:
            out['fcd'] = None

        return out
