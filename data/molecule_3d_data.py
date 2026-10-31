"""
QM9 3D coordinate dataset for direct comparison with EDM/GeoLDM.

Representation: [atom_type_onehot(5) | x, y, z] × 29 atoms = 232 dims.
Includes H atoms (matching EDM's representation). Coordinates centered.

Evaluation uses rdDetermineBonds for proper bond-order-aware atom/mol
stability checks — same metrics as the EDM/GeoLDM table.
"""

import csv
import os
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Geometry import Point3D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


ATOM_TYPES = ['H', 'C', 'N', 'O', 'F']  # QM9 atom types (with H)
ATOM_VALENCE = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1}
N_ATOM_TYPES = len(ATOM_TYPES)
ATOM2IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
MAX_ATOMS = 29  # QM9 max total atoms including H
DIMS_PER_ATOM = N_ATOM_TYPES + 3  # 5 type + 3 xyz = 8
D = MAX_ATOMS * DIMS_PER_ATOM  # 29 * 8 = 232


class QM93DDataset:
    """
    QM9 molecules as fixed-size 3D vectors (including H atoms).

    d = 29 × 8 = 232 dims. Same representation as EDM (minus
    E(3) equivariance). Random rotation augmentation during training
    helps learn approximate rotation invariance.
    """

    def __init__(self, csv_path='/home/weihuang/data/molecules/qm9.csv',
                 max_samples=None, seed=42, augment_rotation=True):
        self.max_atoms = MAX_ATOMS
        self.n_atom_types = N_ATOM_TYPES
        self.dims_per_atom = DIMS_PER_ATOM
        self.d = D
        self.augment_rotation = augment_rotation
        self.seed = seed

        smiles_all = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                smiles_all.append(row['smiles'])
        if max_samples:
            smiles_all = smiles_all[:max_samples]

        vectors = []
        valid_smiles = []
        n_fail = 0
        for smi in smiles_all:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_fail += 1; continue
            mol_h = Chem.AddHs(mol)
            if mol_h.GetNumAtoms() > MAX_ATOMS:
                n_fail += 1; continue
            ret = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
            if ret != 0:
                ret = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
                if ret != 0:
                    n_fail += 1; continue
            try:
                AllChem.MMFFOptimizeMolecule(mol_h)
            except Exception:
                pass

            n_atoms = mol_h.GetNumAtoms()
            conf = mol_h.GetConformer()
            vec = np.zeros((MAX_ATOMS, DIMS_PER_ATOM), dtype=np.float32)
            ok = True
            for i in range(n_atoms):
                sym = mol_h.GetAtomWithIdx(i).GetSymbol()
                if sym not in ATOM2IDX:
                    ok = False; break
                pos = conf.GetAtomPosition(i)
                vec[i, ATOM2IDX[sym]] = 1.0
                vec[i, N_ATOM_TYPES] = pos.x
                vec[i, N_ATOM_TYPES + 1] = pos.y
                vec[i, N_ATOM_TYPES + 2] = pos.z
            if not ok:
                n_fail += 1; continue

            # Center coordinates
            coords = vec[:n_atoms, N_ATOM_TYPES:]
            vec[:n_atoms, N_ATOM_TYPES:] -= coords.mean(axis=0)
            vectors.append(vec.flatten())
            valid_smiles.append(smi)

        self.data = torch.tensor(np.stack(vectors), dtype=torch.float32)
        self.training_smiles = valid_smiles
        self.n_total = len(self.data)
        self.labels = torch.zeros(self.n_total, dtype=torch.long)

        # PCA
        mean = self.data.mean(dim=0)
        centered = self.data - mean
        n_sub = min(5000, self.n_total)
        U, S, Vt = torch.linalg.svd(centered[:n_sub], full_matrices=False)
        self.pca_explained_variance = (S ** 2) / (n_sub - 1)

        print(f"  QM93DDataset loaded: {self.n_total} molecules "
              f"(failed {n_fail}), d={self.d}, "
              f"max_atoms={MAX_ATOMS}, augment={augment_rotation}")

    def _random_rotation(self, vec_batch):
        B = vec_batch.shape[0]
        vec = vec_batch.view(B, MAX_ATOMS, DIMS_PER_ATOM).clone()
        M = torch.randn(B, 3, 3)
        Q, R = torch.linalg.qr(M)
        det = torch.det(Q)
        Q[det < 0, :, 0] *= -1
        coords = vec[:, :, N_ATOM_TYPES:]
        vec[:, :, N_ATOM_TYPES:] = torch.bmm(coords, Q)
        return vec.view(B, self.d)

    def sample_clean(self, n):
        idx = torch.randint(0, self.n_total, (n,))
        x = self.data[idx]
        if self.augment_rotation:
            x = self._random_rotation(x)
        return x, self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        return x0 + sigma_t * eps, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)

    # ------------------------------------------------------------------
    # 3D evaluation (same metrics as EDM/GeoLDM table)
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_3d(x_flat, n_eval_max=10000):
        """
        Evaluate generated 3D molecules on EDM/GeoLDM metrics:
          - Atom stability (%)
          - Molecule stability (%)
          - Validity (%)
          - Valid & Unique (%)

        Uses rdDetermineBonds for proper bond-order inference.
        """
        B = x_flat.shape[0]
        x = x_flat.view(B, MAX_ATOMS, DIMS_PER_ATOM).cpu().numpy()

        total_atoms = 0
        stable_atoms = 0
        n_mol_stable = 0
        n_valid = 0
        valid_smiles_set = set()
        n_total = min(B, n_eval_max)

        for b in range(n_total):
            # Decode: argmax type, take xyz
            types = []
            positions = []
            for i in range(MAX_ATOMS):
                type_logits = x[b, i, :N_ATOM_TYPES]
                if type_logits.max() < 0.3:
                    continue  # empty/padding slot
                t_idx = int(type_logits.argmax())
                types.append(ATOM_TYPES[t_idx])
                positions.append(x[b, i, N_ATOM_TYPES:])

            if len(types) < 2:
                total_atoms += max(len(types), 1)
                continue

            # Build RDKit mol from types + 3D positions
            try:
                rwmol = Chem.RWMol()
                conf = Chem.Conformer(len(types))
                for i, (t, p) in enumerate(zip(types, positions)):
                    rwmol.AddAtom(Chem.Atom(t))
                    conf.SetAtomPosition(i, Point3D(
                        float(p[0]), float(p[1]), float(p[2])))
                rwmol.AddConformer(conf)

                # Infer bonds from 3D with rdDetermineBonds
                rdDetermineBonds.DetermineBonds(rwmol, charge=0)

                # Atom stability: check valence per atom
                mol_all_stable = True
                for atom in rwmol.GetAtoms():
                    sym = atom.GetSymbol()
                    val = atom.GetTotalValence()
                    expected = ATOM_VALENCE.get(sym, 0)
                    is_stable = (val == expected)
                    stable_atoms += is_stable
                    total_atoms += 1
                    if not is_stable:
                        mol_all_stable = False

                if mol_all_stable:
                    n_mol_stable += 1

                # Validity: try sanitization
                try:
                    Chem.SanitizeMol(rwmol)
                    smi = Chem.MolToSmiles(Chem.RemoveHs(rwmol))
                    n_valid += 1
                    valid_smiles_set.add(smi)
                except Exception:
                    pass

            except Exception:
                # rdDetermineBonds failed → molecule is invalid
                # Still count atoms for the stability denominator
                total_atoms += len(types)

        return {
            'atom_stability': stable_atoms / max(total_atoms, 1),
            'molecule_stability': n_mol_stable / max(n_total, 1),
            'validity': n_valid / max(n_total, 1),
            'valid_unique': len(valid_smiles_set) / max(n_total, 1),
            'n_total': n_total,
            'n_valid': n_valid,
            'n_unique_valid': len(valid_smiles_set),
            'total_atoms_checked': total_atoms,
            'stable_atoms': stable_atoms,
        }
