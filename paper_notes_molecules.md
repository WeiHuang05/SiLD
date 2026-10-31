# Molecule-experiment paper notes

These are the talking points and draft LaTeX for how to frame the molecule
section. Addressing two likely reviewer concerns:

1. **Why is validity so low (~39% on QM9 for ULD) vs. G-SchNet / EDM / JODO at >90%?**
2. **Why only one baseline (LDM-CNN)?**

Both concerns dissolve once the paper's claim is scoped correctly.

## The correct scope for the molecule experiments

This is **not** a molecule-generation SOTA paper. It is a **latent-geometry**
paper where molecules happen to be one of several test domains. The claim
is: *at identical encoder, decoder, and score-network capacity, the
denoising-AE training objective produces a better latent for diffusion
than the $\beta$-VAE training objective.* Molecules are a clean test of
this claim because validity (via RDKit) gives us an objective, bias-free
metric that doesn't depend on any learned feature extractor like
FID.

State-of-the-art molecule generators (G-SchNet, EDM, JODO, MoFlow, GDSS)
use **strong molecular inductive biases** — E(3)-equivariant 3D
coordinates, graph-based atom-by-atom construction, valency rules,
junction-tree structure, or discrete diffusion on tokens. Our setting
is **representation-free**: flattened one-hot SMILES into a generic MLP
with continuous diffusion in latent space and argmax decoding. This is
the same setting in which:

- **SMILES-VAE** (Gómez-Bombarelli et al. 2018): **0.7%** validity
- **GrammarVAE** (Kusner et al. 2017): **7.2%** validity
- Character-level continuous diffusion baselines: typically 10–50%

Our ULD at 39.2% is **well ahead of the closest comparable baselines**
on the same representation, and the controlled comparison against the
matched LDM-CNN is where the paper's claim lives.

## Why only LDM-CNN as a baseline?

LDM-CNN is **the** matched-capacity VAE-LDM baseline — same encoder,
same decoder, same score network, same representation, same training
protocol, only the Stage-1 loss differs. Any other baseline either (a)
uses a different representation (EDM, JODO, MoFlow — not comparable),
(b) uses a different architecture but our framework (CharRNN — added
in v2), or (c) cites published numbers for context (see text).

## Draft "Scope" paragraph (paste into Section *Molecules*)

> **Scope.** Our molecule-generation experiments are *not* aimed at
> state-of-the-art validity. Methods such as G-SchNet~\cite{gschnet},
> EDM~\cite{edm}, JODO~\cite{jodo}, and MoFlow~\cite{moflow} achieve
> $>90\%$ validity via strong molecular inductive biases — E(3)
> equivariance on 3D coordinates, graph-based atom-by-atom construction,
> valency rules, or discrete diffusion on tokens. We deliberately adopt
> a *representation-free* setting: flattened one-hot SMILES fed into a
> generic MLP, continuous diffusion in latent space, argmax decoding.
> This is the same setting in which SMILES-VAE~\cite{gomez2018} and
> GrammarVAE~\cite{kusner2017} report $0.7\%$ and $7.2\%$ validity
> respectively. Our goal is not to chase validity in absolute terms;
> it is to isolate the effect of the autoencoder training objective on
> latent diffusion. All experiments compare methods at *identical*
> encoder, decoder, and score-network capacity, so any gap is
> attributable to the Stage-1 loss alone.

## Draft table caption (v2 multi-dataset)

> Table N: Molecule generation on four MoleculeNet benchmarks. All
> methods share an identical MLP encoder, MLP decoder, and latent score
> network; only the Stage-1 loss differs between ULD (denoising AE at
> small $t_0$) and LDM-CNN ($\beta$-VAE with KL regularizer). CharRNN
> is a matched-parameter-count 2-layer LSTM on the same one-hot SMILES
> representation, trained with teacher-forced next-token
> cross-entropy. Validity is reported both raw (argmax decode, then
> RDKit) and after a cheap validity-aware repair that returns the
> longest bracket- and ring-balanced prefix of each generation. The
> same repair is applied to all three methods.

## Things to cite

- Gómez-Bombarelli, R., et al. "Automatic chemical design using a data-driven continuous representation of molecules." ACS Central Science 4.2 (2018): 268-276.  — SMILES-VAE numbers.
- Kusner, M. J., Paige, B., Hernández-Lobato, J. M. "Grammar variational autoencoder." ICML 2017. — GrammarVAE numbers.
- Segler, M. H. S., et al. "Generating focused molecule libraries for drug discovery with recurrent neural networks." ACS Central Science 4.1 (2018): 120-131. — CharRNN baseline.
- Hoogeboom, E., Satorras, V. G., Vignac, C., Welling, M. "Equivariant diffusion for molecule generation in 3D." ICML 2022. — EDM, cited as SOTA context.
- Hang, T., Gu, S., Li, C., Bao, J., Chen, D., Hu, H., Geng, X., Guo, B. "Efficient diffusion training via Min-SNR weighting strategy." ICCV 2023. — Min-SNR-gamma.

## Checklist for the final paper

- [ ] Include the Scope paragraph at the start of the Molecules section
- [ ] Report BOTH raw and repaired validity in the table (so reviewers can see
      the effect of repair and the fact that the ULD > LDM delta is preserved
      either way)
- [ ] Include the CharRNN baseline as a third row (once v2 runs finish)
- [ ] Include example SMILES grids (10-20 per method) in an appendix
- [ ] Cite SMILES-VAE and GrammarVAE explicitly as the same-setting reference
      points; cite EDM / JODO / G-SchNet for SOTA context with inductive biases
- [ ] Make it clear in the captions and main text that the claim is about
      controlled comparison at matched capacity, not absolute validity
