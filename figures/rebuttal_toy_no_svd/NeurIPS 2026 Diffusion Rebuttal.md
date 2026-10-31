---
title: NeurIPS 2026 Diffusion Rebuttal

---

## Response to Reviewer 6niy

We thank the reviewer for the careful examination of both the manuscript and the released code. We provide detailed responses to the concerns below.


### W1. Theory–practice correspondence and the role of KL

We thank the reviewer for pointing this out. We clarify this distinction below

**Analyzable SiLD.** Algorithm 1 is the ambient-space construction studied in the theoretical sections. Stage 1 uses a conservative-form network to learn an explicit projection map, Stage 2 uses an ambient residual-score random-feature model, and the additional high-noise head is introduced only to establish the full-path sampling guarantee.

**Practical SiLD.** The MNIST, CelebA, CelebA-HQ, and molecular experiments use the following scalable implementation:

**Input:** Dataset $\mathcal D$; deterministic encoder $E_\phi$ and decoder $G_\psi$; latent noise predictor $\epsilon_\theta$; Stage-1 low-noise level $t_{\mathrm{geom}}$; and Stage-2 interval $[t_{\mathrm{cut}},T]$.

**Stage 1: Low-noise representation learning**

1. For each minibatch $x_0\sim\mathcal D$, sample
   $\epsilon\sim\mathcal N(0,I)$.

2. Construct the low-noise input using the VP forward process:

$$
x_{t_{\mathrm{geom}}} =
\sqrt{\bar{\alpha}_{t_{\mathrm{geom}}}}\,x_0
+
\sqrt{1-\bar{\alpha}_{t_{\mathrm{geom}}}}\,\epsilon,
$$

where $t_{\mathrm{geom}}=0.005$ and $\{\bar{\alpha}_t\}_{t\in[0,T]}$ denotes the VP noise schedule.

3. Compute the noisy-to-clean reconstruction:

$$
\widehat{x}_0 =
G_\psi\!\left(
E_\phi(x_{t_{\mathrm{geom}}})
\right).
$$

4. Update $(\phi,\psi)$ using

$$
\mathcal L_{\mathrm{stage1}} =
\left\|
\widehat{x}_0-x_0
\right\|_2^2.
$$

5. Freeze $E_\phi$ and $G_\psi$.

**Stage 2: Standard latent diffusion**

1. For each minibatch $x_0\sim\mathcal D$, compute
   $z_0=E_\phi(x_0)$ and sample

$$
t\sim\operatorname{Unif}[t_{\mathrm{cut}},T],
\qquad
\xi\sim\mathcal N(0,I).
$$

2. Construct the noisy latent using the VP forward process:

$$
z_t =
\sqrt{\bar{\alpha}^{\,z}_t}\,z_0
+
\sqrt{1-\bar{\alpha}^{\,z}_t}\,\xi,
$$

where $\{\bar{\alpha}^{\,z}_t\}$ denotes the latent-space VP schedule.

3. Update $\theta$ using

$$
\mathcal L_{\mathrm{stage2}} =
\left\|
\epsilon_\theta(z_t,t)-\xi
\right\|_2^2.
$$

**Sampling.** Sample $z_T\sim\mathcal N(0,I)$, integrate the latent reverse process from $T$ to $t_{\mathrm{cut}}$, apply the Tweedie endpoint estimate, and decode the resulting clean latent using $G_\psi$.


Algorithm 1 is the analyzable ambient-space construction used in the theory, whereas the real-data experiments evaluate Practical SiLD as specified above. The current theorems do not directly establish convergence of this CNN/MLP implementation. Their connection is the geometry-before-density mechanism, not exact architectural identity.

**Role of KL.** KL appeared in the original discussion because the LDM baseline learns its representation using a reconstruction-plus-KL VAE objective. To isolate these effects, we additionally ran a deterministic clean-reconstruction autoencoder followed by the same Stage-2 pipeline as Practical SiLD. Across three seeds on Stacked MNIST, we obtain:

| Method | Recon MSE ↓ | FID ↓ | Modes ↑ |
|---|---:|---:|---:|
| Practical SiLD | $0.003539\pm0.000017$ | $7.951\pm0.262$ | 1000 |
| CleanAE + LDM | $0.003542\pm0.000036$ | $8.199\pm0.137$ | 1000 |
| VAE-LDM | $0.006284\pm0.000052$ | $16.152\pm0.098$ | 1000 |

Here the second stage of SiLD adopts $t \in [0.001,0.999].$ This control gives a more nuanced conclusion. Replacing the stochastic KL-constrained VAE with a deterministic autoencoder explains most of the improvement over VAE-LDM on Stacked MNIST. Relative to CleanAE+LDM, low-noise denoising yields essentially identical reconstruction and a modest additional FID improvement, from $8.199$ to $7.951$.


### W2. Practical implementation and the Stage-2 noise interval

We thank the reviewer for carefully inspecting the released code.  Practical SiLD is not intended to introduce a new latent diffusion backbone; its practical contribution is the score-motivated Stage-1 representation-learning procedure and the explicit separation of low-noise geometry learning from subsequent density learning.

In the real-data experiments, Practical SiLD proceeds as follows:

1. A deterministic encoder–decoder is trained to reconstruct clean data from a fixed small corruption.
2. The encoder and decoder are frozen.
3. A standard latent diffusion model is trained on the resulting latent codes.

The analyzable construction in Algorithm 1 is more explicit: it separately models the projection, residual-score, and high-noise components so that their optimization and generalization properties can be studied. Practical SiLD does not instantiate these components as separate random-feature heads. Instead, it realizes the same geometry-before-density organization using a denoising autoencoder followed by a single standard latent diffusion network. The correspondence is at the mechanism level rather than exact architectural equivalence.

The reviewer also correctly observed that the submitted implementations used different lower endpoints for Stage 2. The standard LDM baseline used $t \in [0.001,0.999],$ whereas the truncated Practical SiLD configuration used $t \in [t_{\mathrm{cut}},T]=[0.005,0.999],$ with the Stage-1 corruption time fixed at $t_{\mathrm{geom}}=0.005$

To determine whether this difference in the Stage-2 interval accounts for the observed improvement, we reran Practical SiLD using the same Stage-2 interval as the LDM baseline, while keeping the encoder–decoder, latent dimension, diffusion architecture, optimizer, training budget, and all other settings fixed. Across three seeds, we obtain:

| Practical SiLD Stage-2 interval | FID ↓ | Modes ↑ |
|---|---:|---:|
| $[0.001,0.999]$ (matched to LDM) | $7.951\pm0.262$ | 1000/1000 |
| $[0.005,0.999]$ (truncated) | $7.968\pm0.304$ | 1000/1000 |

The two settings are essentially indistinguishable: the mean FID differs by only $0.017$, substantially less than the run-to-run standard deviation, and both recover all 1000 modes. Thus, the truncated Stage-2 interval does not explain the performance improvement over LDM.


### W3. Use of the known intrinsic dimension in the toy experiment

We thank the reviewer for raising this concern. To determine whether knowledge of $k$ is necessary, we repeated the toy experiment after removing all uses of the true intrinsic dimension from training. In the new experiment, the learned matrix is $W\in\mathbb R^{100\times 200},$ and we perform neither rank-$k$ truncation nor post-training SVD. The true value $k=5$ is used only to generate the synthetic data and to compute oracle evaluation metrics; it is never provided to the learning algorithm.

We additionally consider the rank-free idempotence regularizer

$$
\mathcal R_{\mathrm{proj}}(W) =
\left\|
(WW^\top)^2-WW^\top
\right\|_F^2.
$$

This regularizer encourages $WW^\top$ to approach an idempotent projection operator, but does not specify its rank. Hence, it does not require prior knowledge of the intrinsic dimension.

The non-oracle results are:

| Stage-1 regularizer | Training uses $k$? | Post-training SVD/rank truncation? | Total MSE ↓ | Linear-score MSE ↓ | Nonlinear-score MSE ↓ | $\|WW^\top-P_\star\|_F$ ↓ |
|---|:---:|:---:|---:|---:|---:|---:|
| None | No | No | 4.930 | 0.362 | 0.224 | 0.377 |
| $\ell_2$, $10^{-4}$ | No | No | 4.930 | 0.362 | 0.224 | 0.377 |
| $\ell_2$, $10^{-3}$ | No | No | 4.929 | 0.361 | 0.224 | 0.376 |
| $\mathcal R_{\mathrm{proj}}$, coefficient $10$ | No | No | **4.800** | **0.160** | **0.157** | **0.097** |

These results show that the toy model can learn an approximately projective operator without access to $k$. Standard $\ell_2$ regularization has little effect, whereas the rank-free idempotence penalty substantially improves projection recovery and both linear- and nonlinear-score errors.


### Q1. Definition and computation of $s_{\mathrm{res}}^*$


The residual score is the population quantity

$$
s_{\mathrm{res}}^*(x,t)
:=
s^*(x,t)
+
\frac{x-\Pi_{\mathcal M}(x)}{h(t)}.
$$

It characterizes the population optimum and is not evaluated explicitly during training. In the analyzable Algorithm 1, after obtaining $\widehat{x}_{t_2}=P_{\widehat q}(x_{t_2})$, Stage 2 uses the standard DSM sample target

$$
y_{\mathrm{res}} =
\frac{x_{t_2}-\widehat{x}_{t_2}}{h(t_2)}
-\frac{\epsilon}{\sqrt{h(t_2)}},
$$

which is fully computable from the sampled corruption and the frozen Stage-1 output. Its conditional expectation is the residual score up to the Stage-1 projection error. Practical SiLD does not explicitly construct $s_{\mathrm{res}}^*$; it uses the standard latent noise-prediction objective in Stage 2.



### Q2. Notation and typographical corrections

We thank the reviewer for identifying these issues.

1. The correct Stage-1 projection update is

   $$
   \widehat{x}=x+h(t_1)f_1(x),
   $$

   consistent with the definition of $f_1$ and Eq. (3). The minus sign in the submitted Algorithm 1 was a typographical error.

2. The Stage-2 objective will explicitly use

   $$
   x_{t_2}\sim p_{t_2},
   \qquad
   \widehat{x}_{t_2},
   \qquad
   f_2(\widehat{x}_{t_2},t_2),
   $$

   rather than the generic variables $x,\widehat{x},t$.

3. Since Theorem 3.4 analyzes Stage 1 at the fixed noise level $t_1$, Eq. (11) should use

   $$
   F_{t_1}(q_s),
   $$

   rather than the generic $F_t(q_s)$. It should not be $F_{t_2}$, because $t_2$ denotes the Stage-2 noise level.

4. We will define the kernel integral operator before the Stage-2 theorem:

   $$
   (T_Kg)(z) =
   \int_{\mathcal M}
   K(z,z')g(z')\,\mu_{t_2}(dz'),
   $$

   where $\mu_{t_2}$ is the distribution of the projected samples at noise level $t_2$. For the vector-valued residual score, $T_K$ acts componentwise. The source condition

   $$
   s_{\mathrm{res}}^*=T_K^\alpha g
   $$

   means that the residual score belongs to the range of the fractional operator $T_K^\alpha$, with smoothness controlled by $\alpha$.

We thank the reviewer for these careful comments. We will incorporate the clarified practical algorithm, the two new controlled experiments, and the notation corrections into the final manuscript.



## Response to Reviewer LLXz

We thank the reviewer for the careful examination of the manuscript and the released code. We address the concerns point by point below.

### W1/Q5. Comparison with recent latent-space diffusibility methods

We thank the reviewer for pointing out the recent latent-diffusibility literature [1,2]. These methods are relevant to the practical motivation of learning a latent space that is well suited for diffusion, but they address a different mechanism from SiLD.

Skorokhodov et al. [1] improve latent diffusibility by imposing frequency-consistency constraints on the autoencoder. Lai et al. [2] propose FreqWarm, which keeps the autoencoder fixed and modifies Stage-2 diffusion training through a frequency warm-up curriculum. In contrast, SiLD starts from the small-noise score decomposition and modifies the representation-learning stage: low-noise denoising first learns a geometry-preserving representation, after which standard latent diffusion learns the density.

To provide a controlled comparison with the Stage-2 approach most directly compatible with our setting, we implemented FreqWarm on top of the matched LDM baseline on Stacked MNIST. The LDM architecture, autoencoder, training budget, and evaluation protocol are unchanged; only the FreqWarm curriculum is added. Across three seeds:

| Method | FID ↓ | Epsilon loss ↓ | KL-to-uniform ↓ | Modes ↑ |
|---|---:|---:|---:|---:|
| Practical SiLD | **$7.968\pm0.304$** | $0.39215\pm0.00061$ | **$0.02756\pm0.00144$** | 1000/1000 |
| LDM | $16.193\pm0.165$ | **$0.37037\pm0.01547$** | $0.02917\pm0.00054$ | 1000/1000 |
| LDM + FreqWarm | $16.326\pm0.419$ | $0.37175\pm0.01331$ | $0.02841\pm0.00051$ | 1000/1000 |

FreqWarm does not improve the matched LDM in this experiment: its mean FID changes from $16.193$ to $16.326$, with no consistent improvement across seeds. In contrast, Practical SiLD achieves an FID of $7.968$ under the same evaluation protocol. All methods recover all 1000 modes, so the FID difference is not explained by mode-count collapse.

This controlled result indicates that the improvement obtained by SiLD is not reproduced by adding a Stage-2 frequency curriculum to the standard LDM pipeline in this setting. It also supports the distinction between the two mechanisms: FreqWarm modifies how diffusion learns a fixed latent representation, whereas SiLD modifies how the representation itself is learned from the small-noise score geometry.

We do not claim that this single Stacked-MNIST experiment establishes universal superiority over all latent-diffusibility methods or settings. We will add [1,2] to the related-work discussion, report this matched FreqWarm comparison, and clarify that frequency-based diffusibility methods and SiLD are complementary approaches acting on different stages of the pipeline.

### W2. Interpretation of the CelebA ablation and auxiliary regularization

Table 2 characterizes Practical SiLD under different regularization choices; it is not intended to claim that the unregularized variant must outperform LDM on every metric. The core SiLD row uses only low-noise denoising, without KL, MMD, or GAN. It improves reconstruction MSE over LDM ($0.00238$ vs. $0.00265$), while giving a higher FID ($46.22$ vs. $43.58$). This reflects a reconstruction–generation trade-off rather than a failure of the score-induced representation.

MMD and GAN are optional empirical extensions that probe latent smoothness and perceptual quality; they are not required to define SiLD. SiLD+MMD reaches FID $42.52$ at $\alpha=1$, and under matched GAN+EMA SiLD reaches $39.57$ versus $40.69$ for LDM, while retaining lower reconstruction error.

These observations do not contradict the theory, which concerns projection alignment and intrinsic-dimensional residual-score estimation in the analyzable model, not optimal FID for every finite-capacity CNN. We will refine the claim to:

> The score-induced objective is sufficient to learn a geometry-preserving latent representation without KL regularization; auxiliary regularization may further improve finite-capacity diffusibility and generation quality.

### W3. Practical algorithm and algorithmic contribution

At the implementation level, the reviewer is correct that Practical SiLD uses a standard time-conditioned latent diffusion architecture and the standard noise-prediction objective in Stage 2. Practical SiLD is not intended to introduce a new latent diffusion backbone.

The real-data implementation consists of:

1. training a deterministic encoder–decoder to reconstruct clean data from a fixed low-noise VP corruption;
2. freezing the encoder and decoder;
3. training a standard latent diffusion model on the learned codes.

Algorithm 1 is instead the analyzable ambient-space construction studied in the theory. It explicitly separates the projection, residual-score, and high-noise components to make their optimization and generalization properties tractable. Practical SiLD does not instantiate these components as separate random-feature heads. The correspondence is at the level of the geometry-before-density mechanism, rather than exact architectural equivalence. The current theorems do not directly establish convergence of the CNN/MLP implementation.

The practical contribution is therefore the theory-motivated representation-learning procedure and the division of learning across noise regimes, not a new Stage-2 architecture.

To distinguish low-noise denoising from the simpler effect of replacing the KL-VAE with a deterministic autoencoder, we ran an additional CleanAE+LDM control. CleanAE+LDM uses clean-to-clean deterministic reconstruction and the same Stage-2 pipeline as Practical SiLD. Across three seeds on Stacked MNIST:

| Method | Recon MSE ↓ | FID ↓ | Modes ↑ |
|---|---:|---:|---:|
| Practical SiLD | $0.003539\pm0.000017$ | $7.951\pm0.262$ | 1000 |
| CleanAE + LDM | $0.003542\pm0.000036$ | $8.199\pm0.137$ | 1000 |
| VAE-LDM | $0.006284\pm0.000052$ | $16.152\pm0.098$ | 1000 |

This control provides a more precise conclusion. Replacing the stochastic KL-constrained VAE with a deterministic autoencoder accounts for most of the improvement over VAE-LDM on Stacked MNIST. Relative to CleanAE+LDM, low-noise denoising gives essentially identical reconstruction and a modest additional FID improvement, from $8.199$ to $7.951$.

Thus, we do not claim that low-noise denoising is the sole source of the improvement. Rather, the experiment shows that it is a competitive score-motivated KL-free representation objective and provides a further modest benefit beyond deterministic clean reconstruction.

### Q1. Meaning of “dimensional collapse”

The phrase “network weights collapse onto the tangent space” was imprecise. Theorem 3.4 controls the operator-level risk

$$
F_{t_1}(q_s)=
\frac12
\mathbb E
\left[
\|P_{q_s}(x)-\Pi_{\mathcal M}(x)\|^2
\right].
$$

The correct conclusion is that the induced denoising map approaches the manifold projection. On a linear manifold, its effective range aligns with the data subspace; this does not imply that every individual weight lies in a tangent space. We will replace the weight-level wording accordingly.

### Q2. Noise levels used in the two stages

Practical Stage 1 uses a fixed low-noise VP corruption,

$$
x_{t_{\mathrm{geom}}} =
\sqrt{\bar\alpha_{t_{\mathrm{geom}}}}\,x_0
+
\sqrt{1-\bar\alpha_{t_{\mathrm{geom}}}}\,\epsilon,
\qquad
t_{\mathrm{geom}}=0.005.
$$

Stage 2 samples $t\sim\operatorname{Unif}[t_{\mathrm{cut}},T]$. The reported SiLD used $[0.005,0.999]$, while LDM used $[0.001,0.999]$. We reran SiLD with the matched interval:

| SiLD Stage-2 interval | FID ↓ | Modes ↑ |
|---|---:|---:|
| $[0.001,0.999]$ | $7.951\pm0.262$ | 1000/1000 |
| $[0.005,0.999]$ | $7.968\pm0.304$ | 1000/1000 |

The $0.017$ FID difference is far below run-to-run variation, ruling out the lower cutoff as the source of the improvement. We also clarify that $t_{\mathrm{geom}}$ and $t_{\mathrm{cut}}$ are distinct diffusion-time parameters, not noise standard deviations.

### Q3. Does the theoretical regularizer alter the learned score?

Yes. The term $\mathcal R$ in Algorithm 1 is the ridge penalty for the analyzable Stage-2 random-feature estimator:

$$
\widehat U =
\arg\min_U
\frac{1}{n}
\sum_{i=1}^n
\left\|
U\Phi(\widehat x_i,t_i)
-y_{\mathrm{res},i}
\right\|^2
+
\lambda\|U\|_F^2.
$$

For finite $\lambda>0$, ridge regularization changes the finite-sample estimator and introduces the standard regularization bias. It is not intended to leave the learned score exactly unchanged. Its role is to control the estimator norm and finite-sample variance; consistency with the unregularized population target requires a vanishing-regularization regime.

This theoretical ridge penalty is distinct from MMD, GAN, or LPIPS regularization in the deep experiments. Practical SiLD does not explicitly use the ambient random-feature residual estimator of Algorithm 1; its Stage 2 uses the standard latent noise-prediction loss.


### Q4. Theoretical status of MMD, GAN, and LPIPS

The current theory does not justify MMD, GAN, or LPIPS. They are optional empirical extensions addressing global latent smoothness or perceptual quality, which lie outside the present theorem. We will state this separation explicitly and will not use auxiliary-regularized results as evidence that the unregularized theoretical model must be optimal in all practical settings.

We thank the reviewer for these comments. We will incorporate the clarified practical algorithm, controlled experiments, and refined claims into the final manuscript.



## Response to Reviewer BHi9

We thank the reviewer for the careful reading and constructive questions. We address the concerns point by point below.

### W1. Connection between the score decomposition and KL regularization

Our theory does not claim that KL regularization is intrinsically inferior or that SiLD must universally outperform a KL-VAE. The theoretical result is that small-noise DSM directly provides a geometric signal for learning the manifold projection, and therefore yields a principled KL-free representation-learning objective.

For the VP forward process,

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,
$$

the population-optimal denoiser satisfies

$$
\sqrt{\bar\alpha_t}\,
\mathbb E[x_0\mid x_t] =
x_t+h(t)s^*(x_t,t),
\qquad
h(t)=1-\bar\alpha_t.
$$

Combining this identity with the small-noise score decomposition gives

$$
x_t+h(t)s^*(x_t,t) =
\Pi_{\mathcal M}(x_t)+O(h(t)).
$$

Thus, noisy-to-clean training is directly motivated by the projection-learning mechanism analyzed in Stage 1. This explains why SiLD can learn a geometry-preserving representation without imposing a Gaussian KL prior. It does not imply that KL is always harmful or that removing KL alone explains every empirical gain.

We will therefore describe SiLD as a **score-induced KL-free alternative** for representation learning, rather than as a universally superior regularizer.

### W2. Scope of the geometric non-degeneracy assumption

The assumption used in Stage 1 is a functional Polyak–Łojasiewicz condition, not a log-Sobolev assumption. A log-Sobolev inequality was mentioned only as a stronger sufficient condition.

We verify the required non-degeneracy condition explicitly for linear manifolds. For general curved manifolds, the Stage-1 convergence theorem is conditional on this assumption. We will state this scope more prominently:

> For general smooth manifolds, the Stage-1 convergence theorem is conditional on the geometric PL inequality; the present work verifies this condition explicitly for linear manifolds.

### W3. Interpretation of the CelebA ablation and MMD regularization

Table 2 is intended to characterize Practical SiLD under different regularization choices, rather than to claim that the unregularized variant must outperform LDM on every metric.

The core unregularized SiLD variant uses only low-noise denoising, without KL, MMD, or adversarial regularization. It improves reconstruction MSE over LDM,

$$
0.00238 \quad \text{vs.} \quad 0.00265,
$$

while obtaining a higher FID,

$$
46.22 \quad \text{vs.} \quad 43.58.
$$

This demonstrates a reconstruction–generation trade-off. MMD and GAN are optional empirical extensions used to study latent smoothness and finite-capacity diffusibility; they are not required to define SiLD. Under matched GAN+EMA regularization, SiLD obtains FID $39.57$ versus $40.69$ for LDM, while retaining slightly lower reconstruction error.

The sensitivity of the MMD coefficient is also visible in the reported sweep, so we do not claim that auxiliary regularization is parameter-free or theoretically guaranteed. These losses are outside the scope of the current theorem.

We will refine the empirical claim to:

> The score-induced objective is sufficient to learn a geometry-preserving latent representation without KL regularization; auxiliary distributional or perceptual regularization may further improve finite-capacity diffusibility and generation quality.

### W4. How the two stages are implemented in practice

The real-data experiments do not use a fixed random-feature network.

We distinguish two realizations:

- **Analyzable SiLD:** the ambient-space construction studied in the theory, with a conservative Stage-1 projection network, a random-feature residual-score model, and an auxiliary high-noise head for the full-path guarantee.
- **Practical SiLD:** the implementation used for MNIST, CelebA, CelebA-HQ, and molecules.

Practical SiLD:

1. trains a deterministic encoder–decoder to reconstruct clean data from a fixed low-noise VP corruption;
2. freezes the encoder and decoder;
3. trains a standard time-conditioned latent diffusion model on the learned codes.

Therefore, Practical SiLD does not introduce a new latent diffusion backbone. Its practical contribution is the score-motivated representation-learning stage and the geometry-before-density separation. The current theorems analyze the conservative/random-feature construction and do not directly establish convergence of the CNN/MLP implementation; the correspondence is mechanism-level rather than architectural.

### W5. “Weights collapse” versus operator alignment

The reviewer is correct that the theorem controls the learned output map rather than individual weights. Theorem 3.4 bounds

$$
F_{t_1}(q_s) =
\frac12
\mathbb E
\left[
\|P_{q_s}(x)-\Pi_{\mathcal M}(x)\|^2
\right].
$$

The precise conclusion is that the induced denoising/projection map approaches the manifold projection. On a linear manifold, the effective range of this operator aligns with the data subspace, but this does not imply that each individual weight lies on the manifold or in a data-dependent tangent space.

We will replace “network weights collapse onto the tangent space” by:

> the induced denoising map rapidly aligns with the manifold projection.

### W6. Dependence on the ambient dimension

The reviewer is correct that the complete end-to-end bound is not independent of the ambient dimension.

The correct distinction is:

- the nonparametric manifold-regime contribution is governed by the intrinsic dimension $k$;
- the high-noise extension retains a parametric dependence on the ambient dimension $d$.

Thus, the claim is not that $d$ disappears entirely, but that it does not enter the nonparametric exponent governing the manifold regime. We will revise the abstract and introduction accordingly.

### W7. Role of the tied/conservative architecture

The conservative structure is used essentially in the theory; the analysis is not simply the standard mean-field analysis of an arbitrary two-layer vector-valued network.

The Stage-1 network is constructed so that

$$
f_1(x;\theta) =
-\nabla_x\Phi_{\mathrm{net}}(x;\theta),
$$

matching the conservative structure of the dominant normal score

$$
-\frac{x-\Pi_{\mathcal M}(x)}{h(t_1)}.
$$

This identity is used to connect Stage-1 DSM training to approximation of the projection map. The subsequent mean-field analysis studies the distributional limit of this conservative-form architecture, with some parameters frozen for tractability.

### W8. Reconstruction-only autoencoder baseline

We ran the suggested deterministic clean-reconstruction autoencoder control. CleanAE+LDM uses clean-to-clean deterministic reconstruction and the same Stage-2 pipeline as Practical SiLD. Across three seeds on Stacked MNIST:

| Method | Recon MSE ↓ | FID ↓ | Modes ↑ |
|---|---:|---:|---:|
| Practical SiLD | $0.003539\pm0.000017$ | $7.951\pm0.262$ | 1000 |
| CleanAE + LDM | $0.003542\pm0.000036$ | $8.199\pm0.137$ | 1000 |
| VAE-LDM | $0.006284\pm0.000052$ | $16.152\pm0.098$ | 1000 |

This control provides a more precise conclusion. Replacing the stochastic KL-constrained VAE with a deterministic autoencoder explains most of the improvement over VAE-LDM on Stacked MNIST. Relative to CleanAE+LDM, low-noise denoising gives essentially identical reconstruction and a modest additional FID improvement, from $8.199$ to $7.951$.

We therefore do not claim that low-noise denoising is the sole source of the improvement. Rather, it provides a principled score-motivated KL-free objective and yields a further modest benefit beyond deterministic clean reconstruction.

We thank the reviewer for these comments. We will incorporate the clarified theoretical scope, practical algorithm, corrected operator-level statement, and new controlled experiment into the final manuscript.



## Response to Reviewer TuCJ

We thank the reviewer for the careful assessment and constructive questions. We address the concerns below.

### W1/Q1. What is MMD regularization in this setting?

MMD is an optional Stage-1 regularizer applied to the learned latent distribution; it is not part of the core SiLD definition or the current theoretical analysis. Let

$$
z_i=E_\phi(x_i)
$$

be encoded samples and let $u_i$ be samples from the chosen reference latent distribution. The minibatch MMD term has the standard form

$$
\widehat{\mathrm{MMD}}^2 =
\frac{1}{B(B-1)}
\sum_{i\neq j}k(z_i,z_j) +
\frac{1}{B(B-1)}
\sum_{i\neq j}k(u_i,u_j) -
\frac{2}{B^2}
\sum_{i,j}k(z_i,u_j),
$$

where $k$ is the kernel used in the implementation. The regularized Stage-1 objective is

$$
\mathcal L_{\mathrm{Stage\,1}} =
\mathcal L_{\mathrm{denoise}}
+
\alpha_{\mathrm{MMD}}
\widehat{\mathrm{MMD}}^2.
$$

The purpose of this term is to study how additional global smoothing of the latent distribution affects the reconstruction–generation trade-off. The unregularized row remains the core Practical SiLD method.

We will state the exact reference distribution, kernel, bandwidth, and minibatch estimator used in the implementation. We also clarify that $\alpha_{\mathrm{MMD}}$ is unrelated to the theoretical ridge coefficient $\lambda$ in Algorithm 1.

### W2/Q2. Hyperparameter choices and sensitivity

We agree that the submitted manuscript did not clearly distinguish structural parameters from optional empirical hyperparameters.

Practical Stage 1 uses a fixed low-noise VP corruption,

$$
x_{t_{\mathrm{geom}}} =
\sqrt{\bar\alpha_{t_{\mathrm{geom}}}}\,x_0
+
\sqrt{1-\bar\alpha_{t_{\mathrm{geom}}}}\,\epsilon,
\qquad
t_{\mathrm{geom}}=0.005.
$$

After freezing the encoder and decoder, Stage 2 trains a standard latent diffusion model with

$$
t\sim\operatorname{Unif}[t_{\mathrm{cut}},T].
$$

The originally reported SiLD setting used

$$
[t_{\mathrm{cut}},T]=[0.005,0.999],
$$

whereas the LDM baseline used $[0.001,0.999]$. We reran SiLD using the matched LDM interval while holding all other settings fixed:

| Practical SiLD Stage-2 interval | FID ↓ | Modes ↑ |
|---|---:|---:|
| $[0.001,0.999]$ | $7.951\pm0.262$ | 1000/1000 |
| $[0.005,0.999]$ | $7.968\pm0.304$ | 1000/1000 |

The mean FID differs by only $0.017$, far below the run-to-run variation. Thus, the Stage-2 lower cutoff does not account for the improvement over LDM.

For MMD, Table 2 reports the complete sweep

$$
\alpha_{\mathrm{MMD}}\in\{0.1,1.0,5.0\},
$$

rather than only the best value:

| MMD coefficient | FID ↓ | Recon MSE ↓ |
|---:|---:|---:|
| none | 46.22 | 0.00238 |
| $0.1$ | 43.56 | 0.00229 |
| $1.0$ | 42.52 | 0.00210 |
| $5.0$ | 45.59 | 0.00207 |

This sweep shows that stronger regularization is not monotonically better and that the generation metric is sensitive to $\alpha_{\mathrm{MMD}}$. We present these rows as a characterization of Practical SiLD, not as evidence of parameter insensitivity.

The theoretical ridge parameter $\lambda$ has a different role: it controls the norm and finite-sample variance of the analyzable random-feature estimator and introduces the standard finite-$\lambda$ regularization bias.



### W3. Architecture constraints

The theoretical analysis does require constrained architectures. Stage 1 uses a conservative-form network satisfying

$$
f_1(x;\theta) =
-\nabla_x\Phi_{\mathrm{net}}(x;\theta),
$$

which matches the conservative structure of the dominant small-noise score. Stage 2 is modeled by a random-feature network so that residual-score estimation reduces to tractable ridge regression on the learned manifold.

These architectural choices are essential to the current proof, but they are not imposed on the real-data implementation. Practical SiLD uses a deterministic CNN/MLP encoder–decoder followed by a standard time-conditioned latent diffusion model. The present theorems analyze the conservative/random-feature realization and do not directly establish convergence of the deep implementation; their correspondence is at the level of the geometry-before-density mechanism.



### Q3. Disentangling MMD from SiLD

Nothing in MMD is specific to SiLD; it can also be applied to an LDM encoder. We agree that the submitted SiLD+MMD versus standard-LDM comparison changes both the representation-learning procedure and the regularization strategy. Therefore, the unmatched MMD rows cannot establish that the FID improvement is specifically caused by SiLD.

Table 2 was intended primarily as a within-method ablation of Practical SiLD. The unregularized SiLD row improves reconstruction MSE over LDM,

$$
0.00238 \quad \text{vs.} \quad 0.00265,
$$

while obtaining a higher FID,

$$
46.22 \quad \text{vs.} \quad 43.58.
$$

The MMD rows show that additional latent smoothing can improve generation while retaining the reconstruction advantage. They should not be interpreted as an isolated cross-method comparison.

The submitted table does contain a regularization-matched GAN+EMA comparison:

| Method | FID ↓ | Recon MSE ↓ |
|---|---:|---:|
| LDM + GAN | 40.69 | 0.00103 |
| SiLD + GAN | 39.57 | 0.00098 |

This result shows a modest SiLD advantage under the same GAN+EMA strategy, but we will not use the unmatched MMD rows as the sole evidence for SiLD’s performance advantage.

We will clarify the purpose of Table 2 as follows:

> The unregularized row evaluates the core score-induced representation objective, while the MMD and GAN rows characterize its compatibility with optional practical regularizers and the resulting reconstruction–generation trade-off.


## Response to Reviewer xRk5

We thank the reviewer for recognizing the paper’s attempt to connect score singularity, neural-network training dynamics, and latent diffusion design. We address the concerns below.

### W1/Q1. Connection between the theoretical architecture and the practical implementation

We agree that the submitted manuscript did not clearly distinguish the analyzable SiLD construction from the deep implementation used in the real-data experiments.

**Analyzable SiLD** is the ambient-space construction studied in the theory. Stage 1 uses a conservative-form network to learn an explicit projection map, Stage 2 uses a random-feature residual-score estimator, and an additional high-noise head is introduced for the full-path sampling guarantee.

**Practical SiLD**, used for MNIST, CelebA, CelebA-HQ, and molecules, instead:

1. trains a deterministic encoder–decoder to reconstruct clean data from a fixed low-noise VP corruption;
2. freezes the encoder and decoder;
3. trains a standard time-conditioned latent diffusion model on the learned codes.

Thus, Practical SiLD does not instantiate the projection, residual-score, and high-noise components as separate random-feature heads. The current theorems analyze the conservative/random-feature realization and do not directly establish convergence of the CNN/MLP implementation. Their correspondence is at the level of the geometry-before-density mechanism rather than exact architectural equivalence.

The connection for Practical Stage 1 follows from the VP denoising identity. For

$$
x_t =
\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,
$$

the population-optimal denoiser satisfies

$$
\sqrt{\bar\alpha_t}\,
\mathbb E[x_0\mid x_t] =
x_t+
(1-\bar\alpha_t)s^*(x_t,t).
$$

Together with the small-noise score decomposition, this implies that low-noise noisy-to-clean training approximates the manifold projection. Practical Stage 2 then learns the density in the frozen learned coordinates using standard latent diffusion.

We will describe the real-data method explicitly as Practical SiLD and make clear that the theoretical and practical realizations share a mechanism, not an identical architecture.

### W2. Separating low-noise denoising from removing KL

The reviewer correctly notes that the submitted comparison conflated low-noise denoising with replacing the stochastic KL-VAE by a deterministic autoencoder.

We therefore ran an additional CleanAE+LDM control. CleanAE+LDM uses deterministic clean-to-clean reconstruction and the same Stage-2 pipeline as Practical SiLD. Across three seeds on Stacked MNIST:

| Method | Recon MSE ↓ | FID ↓ | Modes ↑ |
|---|---:|---:|---:|
| Practical SiLD | $0.003539\pm0.000017$ | $7.951\pm0.262$ | 1000 |
| CleanAE + LDM | $0.003542\pm0.000036$ | $8.199\pm0.137$ | 1000 |
| VAE-LDM | $0.006284\pm0.000052$ | $16.152\pm0.098$ | 1000 |

This control gives a more precise conclusion. Replacing the stochastic KL-constrained VAE with a deterministic autoencoder accounts for most of the improvement over VAE-LDM on Stacked MNIST. Relative to CleanAE+LDM, low-noise denoising yields essentially identical reconstruction and a modest additional FID improvement, from $8.199$ to $7.951$.

We therefore do not claim that low-noise denoising is the sole source of the empirical gain. Rather, it provides a score-motivated KL-free representation objective and gives an additional modest benefit beyond deterministic clean reconstruction.

### Q2. Sensitivity to the noise scales and stage separation

Practical Stage 1 uses a fixed low-noise VP corruption,

$$
x_{t_{\mathrm{geom}}} =
\sqrt{\bar\alpha_{t_{\mathrm{geom}}}}\,x_0
+
\sqrt{1-\bar\alpha_{t_{\mathrm{geom}}}}\,\epsilon,
\qquad
t_{\mathrm{geom}}=0.005.
$$

After freezing the encoder and decoder, Stage 2 samples

$$
t\sim\operatorname{Unif}[t_{\mathrm{cut}},T].
$$

The originally reported SiLD configuration used

$$
[t_{\mathrm{cut}},T]=[0.005,0.999],
$$

whereas LDM used $[0.001,0.999]$. We reran Practical SiLD with the matched LDM interval while keeping all other settings fixed:

| Practical SiLD Stage-2 interval | FID ↓ | Modes ↑ |
|---|---:|---:|
| $[0.001,0.999]$ | $7.951\pm0.262$ | 1000/1000 |
| $[0.005,0.999]$ | $7.968\pm0.304$ | 1000/1000 |

The mean FID differs by only $0.017$, far below run-to-run variation, and both settings recover all 1000 modes. This rules out the lower Stage-2 cutoff as the source of the improvement over LDM.

We also clarify that Practical SiLD does not use an adaptively selected “switch time” within one training trajectory. Stage 1 is trained first, the encoder and decoder are frozen, and Stage 2 is then trained separately. The stage separation is therefore an explicit algorithmic design rather than a tuned transition point during joint optimization.

### W3/Q3. CelebA regularization and the claim that score matching replaces KL

Table 2 is intended to characterize Practical SiLD under different regularization choices, rather than to claim that the unregularized variant must outperform standard LDM on every metric.

The core unregularized SiLD variant uses only low-noise denoising, without KL, MMD, or GAN regularization. It improves reconstruction MSE over LDM,

$$
0.00238 \quad \text{vs.} \quad 0.00265,
$$

while giving a higher FID,

$$
46.22 \quad \text{vs.} \quad 43.58.
$$

This reflects a reconstruction–generation trade-off. MMD and GAN are optional empirical extensions used to study latent smoothness and finite-capacity diffusibility; they are not required to define SiLD. Under matched GAN+EMA regularization, SiLD obtains FID $39.57$ compared with $40.69$ for LDM, while retaining slightly lower reconstruction error.

This observation does not contradict the theory. The theoretical results concern projection alignment and intrinsic-dimensional residual-score estimation in the analyzable model; they do not predict that an unregularized finite-capacity CNN must achieve the best FID on every dataset.

We will refine the broad claim to:

> The score-induced objective is sufficient to learn a geometry-preserving latent representation without KL regularization; auxiliary distributional or perceptual regularization may further improve finite-capacity diffusibility and generation quality.

### W4. Experimental breadth and modern latent-diffusibility baselines

We agree that the image experiments are limited in resolution and are not intended as a comprehensive state-of-the-art latent-diffusion benchmark. The main contribution is the theoretical collapse-and-refine mechanism and its controlled empirical validation.

To compare with a recent latent-diffusibility method, we implemented FreqWarm on the matched LDM baseline on Stacked MNIST, keeping the autoencoder, architecture, training budget, and evaluation protocol fixed:

| Method | FID ↓ | Epsilon loss ↓ | Modes ↑ |
|---|---:|---:|---:|
| Practical SiLD | **$7.968\pm0.304$** | $0.39215\pm0.00061$ | 1000/1000 |
| LDM | $16.193\pm0.165$ | **$0.37037\pm0.01547$** | 1000/1000 |
| LDM + FreqWarm | $16.326\pm0.419$ | $0.37175\pm0.01331$ | 1000/1000 |

FreqWarm does not improve the matched LDM in this setting, whereas Practical SiLD retains a substantially lower FID. This supports the distinction between the mechanisms: FreqWarm modifies diffusion training on a fixed latent representation, while SiLD changes how the representation itself is learned from small-noise score geometry.

We do not claim universal superiority from this single benchmark. We will add the recent latent-diffusibility literature, report this controlled comparison, and explicitly state the limited resolution and scope of the current image experiments.

We thank the reviewer for these comments. We will incorporate the clarified theory–practice relationship, controlled experiments, refined claims, and empirical limitations into the final manuscript.