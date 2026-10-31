"""
Synthetic data generators for manifold diffusion experiments.

Supports:
  1. Linear subspace manifold (MoG in k-dim subspace of R^d)
  2. Nonlinear circle manifold (S^1 embedded in R^d)
  3. Swiss roll manifold (2D manifold in R^d)
  4. Multi-component nonlinearly lifted manifold (K disjoint Swiss rolls
     embedded in R^D via a fixed random Tanh-MLP)
"""

import torch
import numpy as np


def random_orthonormal_basis(d, k, seed=None):
    """Generate a random orthonormal basis A in R^{d x k}."""
    rng = np.random.RandomState(seed)
    M = rng.randn(d, k)
    A, _ = np.linalg.qr(M)
    return torch.tensor(A[:, :k], dtype=torch.float32)


class LinearManifoldDataset:
    """
    Data on a k-dim linear subspace of R^d.
    z ~ MoG in R^k, x0 = A @ z, xt = x0 + sigma_t * eps.
    """

    def __init__(self, d=100, k=5, n_components=3, seed=42):
        self.d = d
        self.k = k
        self.n_components = n_components
        self.seed = seed

        self.A = random_orthonormal_basis(d, k, seed=seed)  # (d, k)

        # MoG parameters in latent space
        rng = np.random.RandomState(seed)
        self.means = torch.tensor(
            rng.randn(n_components, k) * 3.0, dtype=torch.float32
        )
        self.weights = torch.ones(n_components) / n_components
        self.std = 0.5

    def sample_clean(self, n):
        """Sample n clean data points x0 = A @ z."""
        # Sample component indices
        idx = torch.multinomial(self.weights, n, replacement=True)
        z = torch.randn(n, self.k) * self.std
        for c in range(self.n_components):
            mask = idx == c
            z[mask] += self.means[c]
        x0 = z @ self.A.T  # (n, d)
        return x0, z

    def sample_noisy(self, n, sigma_t):
        """Sample noisy data xt = x0 + sigma_t * eps."""
        x0, z = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, z

    def get_projector(self):
        """Return the orthogonal projector P = A A^T."""
        return self.A @ self.A.T  # (d, d)

    def true_score(self, xt, x0, sigma_t):
        """Compute the true denoising score: -eps/sigma_t = -(xt - x0)/sigma_t^2."""
        return -(xt - x0) / (sigma_t ** 2)


class CircleManifoldDataset:
    """
    Data on S^1 (circle) embedded in R^d via first 2 coordinates.
    Non-uniform density: two Gaussian clusters at theta=0 and theta=pi.
    """

    def __init__(self, d=20, seed=42):
        self.d = d
        self.seed = seed

    def sample_clean(self, n):
        """Sample n points on the circle with non-uniform density."""
        rng = np.random.RandomState(self.seed)
        # Two clusters at theta=0 and theta=pi
        cluster_probs = [0.45, 0.45, 0.10]  # two peaks + uniform background
        assignments = rng.choice(3, size=n, p=cluster_probs)

        theta = np.zeros(n)
        for i in range(n):
            if assignments[i] == 0:
                theta[i] = rng.normal(0, 0.3)
            elif assignments[i] == 1:
                theta[i] = rng.normal(np.pi, 0.3)
            else:
                theta[i] = rng.uniform(0, 2 * np.pi)

        x0 = np.zeros((n, self.d))
        x0[:, 0] = np.cos(theta)
        x0[:, 1] = np.sin(theta)
        return torch.tensor(x0, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32)

    def sample_noisy(self, n, sigma_t):
        x0, theta = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, theta

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)


class SwissRollDataset:
    """
    2D Swiss roll manifold embedded in R^d.
    """

    def __init__(self, d=50, seed=42):
        self.d = d
        self.seed = seed

    def sample_clean(self, n):
        rng = np.random.RandomState(self.seed)
        t = 1.5 * np.pi * (1 + 2 * rng.rand(n))
        height = rng.rand(n) * 10

        x0 = np.zeros((n, self.d))
        x0[:, 0] = t * np.cos(t) / 10.0
        x0[:, 1] = height / 10.0
        x0[:, 2] = t * np.sin(t) / 10.0
        return torch.tensor(x0, dtype=torch.float32), torch.tensor(
            np.stack([t, height], axis=1), dtype=torch.float32
        )

    def sample_noisy(self, n, sigma_t):
        x0, latent = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, latent

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)


class MultiManifoldLiftedDataset:
    """
    K disjoint 2D Swiss-roll components, lifted to R^D via a *fixed* random
    nonlinear MLP (Tanh activations). Each component lives in a different
    region of the intrinsic 2D space, so the components are disconnected.

    This is the testbed for the "VAE provably fails, ULD provably wins"
    story. Because we know the lift map exactly, we can:
      - Compute the distance from any sampled point to the *true* manifold
        by gradient-descending the inverse map (project_to_manifold).
      - Compute mode-coverage exactly (which component each sample is
        closest to).
      - Estimate intrinsic dim of generated samples.

    VAE intuition for why it fails: the KL-to-N(0,I) prior smears latent
    mass across the gaps between components, and the Gaussian posterior
    cannot concentrate on a thin 2D manifold inside R^D.

    ULD intuition for why it works: at small t0, the Tweedie estimator
    E[x0 | x_{t0}] → projection onto M, and the bottleneck encoder is
    forced to map noisy points back to the nearest manifold point.
    """

    def __init__(self, D=128, K=3, intrinsic_dim=2,
                 hidden_dim=256, n_lift_layers=3,
                 component_separation=4.0, n_pool=20000, seed=42):
        self.D = D
        self.K = K
        self.intrinsic_dim = intrinsic_dim
        self.hidden_dim = hidden_dim
        self.n_lift_layers = n_lift_layers
        self.component_separation = component_separation
        self.seed = seed

        # ----- 1. Build the fixed random nonlinear lift  R^k -> R^D -----
        # We use small fixed weights so the lift is approximately
        # bi-Lipschitz on the support — this keeps the manifold smooth
        # and prevents degenerate folds.
        rng = np.random.RandomState(seed)
        layers = []
        in_dim = intrinsic_dim
        dims = [in_dim] + [hidden_dim] * (n_lift_layers - 1) + [D]
        self.lift_weights = []
        self.lift_biases = []
        for a, b in zip(dims[:-1], dims[1:]):
            # Glorot-ish init, smallish to keep things bounded.
            # Cast everything to float32 explicitly — np.sqrt produces
            # float64 and would silently upcast.
            W = (rng.randn(a, b) / np.sqrt(a)).astype(np.float32)
            bvec = (rng.randn(b) * 0.1).astype(np.float32)
            self.lift_weights.append(torch.tensor(W))
            self.lift_biases.append(torch.tensor(bvec))

        # ----- 2. Pre-sample the per-component intrinsic distributions ----
        # Each component is a Swiss roll over (t, h) with t in a different
        # range so the components are disconnected in intrinsic space (and
        # therefore in the lifted ambient space too).
        self.component_ranges = []
        base_offset = 0.0
        for k in range(K):
            t_lo = base_offset
            t_hi = base_offset + 1.0
            self.component_ranges.append((t_lo, t_hi))
            base_offset += 1.0 + component_separation  # gap between components

        # Pool of clean samples for fast batched sampling
        self._pool_x, self._pool_z, self._pool_comp = self._build_pool(n_pool, seed)

        # Pre-compute lifted "manifold dictionary" for fast distance queries
        # (a dense set of intrinsic points evaluated through the lift).
        self._build_manifold_dictionary(n_per_comp=4000, seed=seed + 1)

        # Ground-truth coverage (per-component sampling probabilities)
        # — derived from the pool. Used by evaluate_samples for KL(gen || true).
        gt_counts = torch.bincount(self._pool_comp, minlength=K).float()
        self.true_coverage = gt_counts / gt_counts.sum()

    # ------------------------------------------------------------------
    # Lift map and helpers
    # ------------------------------------------------------------------

    def _lift(self, z_intrinsic):
        """Apply the fixed random nonlinear MLP: R^k -> R^D."""
        h = z_intrinsic
        for i, (W, b) in enumerate(zip(self.lift_weights, self.lift_biases)):
            h = h @ W + b
            if i < len(self.lift_weights) - 1:
                h = torch.tanh(h)
        return h

    def _sample_component_intrinsic(self, n_per_comp, seed):
        """Sample n_per_comp 2D points from each of the K Swiss-roll components."""
        rng = np.random.RandomState(seed)
        zs = []
        comps = []
        for k, (t_lo, t_hi) in enumerate(self.component_ranges):
            t = 1.5 * np.pi * (t_lo + (t_hi - t_lo) * rng.rand(n_per_comp))
            h = rng.rand(n_per_comp) * 2.0 - 1.0  # height in [-1, 1]
            # Standard Swiss roll parameterization in 2D, then we add the
            # height as a second intrinsic coordinate.
            x_int = t * np.cos(t) / 6.0
            y_int = h
            zs.append(np.stack([x_int, y_int], axis=1))
            comps.append(np.full(n_per_comp, k, dtype=np.int64))
        z_int = np.concatenate(zs, axis=0).astype(np.float32)
        comp = np.concatenate(comps, axis=0)
        return z_int, comp

    def _build_pool(self, n_pool, seed):
        n_per = n_pool // self.K
        z_int, comp = self._sample_component_intrinsic(n_per, seed)
        z_int_t = torch.tensor(z_int)
        x = self._lift(z_int_t)
        return x, z_int_t, torch.tensor(comp)

    def _build_manifold_dictionary(self, n_per_comp, seed):
        """Dense pre-sampled lifted manifold for distance-to-M queries."""
        z_int, comp = self._sample_component_intrinsic(n_per_comp, seed)
        z_int_t = torch.tensor(z_int)
        with torch.no_grad():
            x_dict = self._lift(z_int_t)
        self._dict_x = x_dict
        self._dict_comp = torch.tensor(comp)

    # ------------------------------------------------------------------
    # Public dataset interface (matches other manifold datasets)
    # ------------------------------------------------------------------

    @property
    def d(self):
        return self.D

    @property
    def pca_explained_variance(self):
        with torch.no_grad():
            X = self._pool_x - self._pool_x.mean(dim=0, keepdim=True)
            U, S, V = torch.svd(X)
            return (S ** 2) / (X.shape[0] - 1)

    def sample_clean(self, n):
        idx = torch.randint(0, self._pool_x.shape[0], (n,))
        return self._pool_x[idx], self._pool_z[idx]

    def sample_noisy(self, n, sigma_t):
        x0, z = self.sample_clean(n)
        eps = torch.randn_like(x0)
        return x0 + sigma_t * eps, x0, eps, z

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)

    # ------------------------------------------------------------------
    # Evaluation metrics (the whole point of using a synthetic dataset)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def manifold_distance(self, x_samples, batch_size=512):
        """
        For each row in x_samples (n, D), return the Euclidean distance to
        the nearest point in the dense lifted manifold dictionary, plus
        the index of the assigned component.

        Returns:
            dists  : (n,) min distance to lifted manifold M
            comps  : (n,) which component each sample is closest to
        """
        n = x_samples.shape[0]
        dists = torch.empty(n)
        comps = torch.empty(n, dtype=torch.long)
        dict_x = self._dict_x  # (M, D)
        dict_comp = self._dict_comp  # (M,)
        for start in range(0, n, batch_size):
            chunk = x_samples[start:start + batch_size]
            # (B, M) pairwise distances
            d2 = torch.cdist(chunk, dict_x)
            mind, argd = d2.min(dim=1)
            dists[start:start + chunk.shape[0]] = mind
            comps[start:start + chunk.shape[0]] = dict_comp[argd]
        return dists, comps

    @torch.no_grad()
    def mode_coverage(self, x_samples):
        """Fraction of samples assigned to each component (length K)."""
        _, comps = self.manifold_distance(x_samples)
        counts = torch.bincount(comps, minlength=self.K).float()
        return counts / counts.sum()

    @torch.no_grad()
    def evaluate_samples(self, x_samples):
        """
        One-call summary: distance-to-manifold + per-component coverage +
        KL(generated coverage || ground-truth coverage).
        """
        dists, comps = self.manifold_distance(x_samples)
        coverage = torch.bincount(comps, minlength=self.K).float()
        coverage = coverage / coverage.sum().clamp(min=1)
        eps = 1e-8
        kl_to_true = (coverage * (coverage.clamp(min=eps).log()
                                  - self.true_coverage.clamp(min=eps).log())).sum()
        return {
            'mean_dist_to_M': dists.mean().item(),
            'median_dist_to_M': dists.median().item(),
            'p95_dist_to_M': dists.quantile(0.95).item(),
            'max_dist_to_M': dists.max().item(),
            'mode_coverage': coverage.tolist(),
            'true_coverage': self.true_coverage.tolist(),
            'mode_kl_to_true': kl_to_true.item(),
        }
