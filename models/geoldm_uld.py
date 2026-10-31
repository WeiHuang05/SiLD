"""
GeoLDM-ULD: Geometric Latent Diffusion with ULD's denoising AE objective.

Combines GeoLDM's architecture (Xu et al., ICML 2023) with ULD's
denoising autoencoder training for Stage 1.

Key design from GeoLDM paper:
  - Latent space is POINT-STRUCTURED: z = (z_x ∈ R^{N×3}, z_h ∈ R^{N×k})
    where z_x is equivariant (positions) and z_h is invariant (features).
  - Encoder/Decoder/Score-net are ALL equivariant EGNNs.
  - Diffusion runs in the point-structured latent space.
  - All intermediate states live on zero-center-of-gravity subspace.

Our contribution: replace GeoLDM's KL-reg/ES-reg Stage 1 with ULD's
denoising AE at small t0. This fixes the KL instability they reported
(Appendix E) and provides a principled alternative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ========================================================================
# EGNN Layers (following Satorras et al. 2021, used by GeoLDM)
# ========================================================================

class EGCLLayer(nn.Module):
    """
    Equivariant Graph Convolutional Layer.
    Updates h (invariant) and x (equivariant) via message passing.
    Optionally conditions on time embedding for the score network.
    """
    def __init__(self, h_dim, time_dim=0):
        super().__init__()
        edge_in = 2 * h_dim + 1 + time_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim), nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * h_dim + time_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, 1, bias=False),
        )
        # Attention for edge reweighting
        self.att_mlp = nn.Sequential(
            nn.Linear(h_dim, 1), nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.01)

    def forward(self, h, x, mask, t_embed=None):
        """
        h: (B, N, h_dim)
        x: (B, N, 3)
        mask: (B, N)
        t_embed: (B, h_dim) optional time embedding
        """
        B, N, _ = h.shape

        dx = x.unsqueeze(2) - x.unsqueeze(1)       # (B, N, N, 3)
        d2 = (dx ** 2).sum(-1, keepdim=True)         # (B, N, N, 1)
        # Normalize relative directions by distance (GeoLDM/EDM convention).
        # Clamp BEFORE sqrt to avoid inf gradients at zero distance.
        dist = d2.clamp(min=1e-8).sqrt()

        hi = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)

        edge_feat = [hi, hj, d2]
        if t_embed is not None:
            t_exp = t_embed.unsqueeze(1).unsqueeze(1).expand(-1, N, N, -1)
            edge_feat.append(t_exp)

        m_ij = self.edge_mlp(torch.cat(edge_feat, dim=-1))

        # Edge mask
        edge_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        eye = torch.eye(N, device=h.device).unsqueeze(0)
        edge_mask = edge_mask * (1 - eye)
        m_ij = m_ij * edge_mask.unsqueeze(-1)

        # Attention weights
        att = self.att_mlp(m_ij) * edge_mask.unsqueeze(-1)

        # Coordinate update (equivariant)
        w = self.coord_mlp(m_ij)
        # Normalize by (d+1) as in GeoLDM
        coord_diff = dx / (dist + 1)
        n_edges = edge_mask.sum(dim=2, keepdim=True).clamp(min=1)  # (B, N, 1)
        x_update = (coord_diff * w).sum(dim=2) / n_edges  # (B, N, 3) / (B, N, 1)
        x_new = x + torch.clamp(x_update, -5.0, 5.0)

        # Node update (invariant)
        agg = (att * m_ij).sum(dim=2) / n_edges  # (B, N, h_dim) / (B, N, 1)
        node_in = [h, agg]
        if t_embed is not None:
            node_in.append(t_embed.unsqueeze(1).expand(-1, N, -1))
        h_new = h + self.node_mlp(torch.cat(node_in, dim=-1))

        h_new = h_new * mask.unsqueeze(-1)
        x_new = x_new * mask.unsqueeze(-1)

        return h_new, x_new


def remove_mean(x, mask):
    """Center positions to zero center-of-gravity (translation invariance)."""
    n_atoms = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
    mean = (x * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / n_atoms
    return (x - mean) * mask.unsqueeze(-1)


# ========================================================================
# Geometric Autoencoder
# ========================================================================

class GeoEncoder(nn.Module):
    """
    EGNN encoder: (atom_types, positions) → (z_x, z_h).
    z_x ∈ R^{N×3} equivariant latent positions.
    z_h ∈ R^{N×k} invariant latent features.
    Shallow encoder (1-2 layers) as recommended by GeoLDM.
    """
    def __init__(self, n_atom_types, h_dim=256, latent_h_dim=1, n_layers=2):
        super().__init__()
        self.latent_h_dim = latent_h_dim
        self.type_embed = nn.Linear(n_atom_types, h_dim)
        self.layers = nn.ModuleList([EGCLLayer(h_dim) for _ in range(n_layers)])
        self.project_h = nn.Linear(h_dim, latent_h_dim)

    def forward(self, types, positions, mask):
        """
        Returns: z_x (B, N, 3), z_h (B, N, k)
        """
        h = self.type_embed(types)
        x = positions.clone()
        for layer in self.layers:
            h, x = layer(h, x, mask)
        z_h = self.project_h(h) * mask.unsqueeze(-1)
        z_x = remove_mean(x, mask)
        return z_x, z_h


class GeoDecoder(nn.Module):
    """
    EGNN decoder: (z_x, z_h) → (atom_types, positions).
    Takes point-structured latent and refines through EGNN layers.
    """
    def __init__(self, n_atom_types, h_dim=256, latent_h_dim=1, n_layers=4):
        super().__init__()
        self.unproject_h = nn.Linear(latent_h_dim, h_dim)
        self.layers = nn.ModuleList([EGCLLayer(h_dim) for _ in range(n_layers)])
        self.type_out = nn.Linear(h_dim, n_atom_types)

    def forward(self, z_x, z_h, mask):
        """
        z_x: (B, N, 3), z_h: (B, N, k)
        Returns: types (B, N, n_types), positions (B, N, 3)
        """
        h = self.unproject_h(z_h)
        x = z_x.clone()
        for layer in self.layers:
            h, x = layer(h, x, mask)
        types = self.type_out(h)
        positions = remove_mean(x, mask)
        return types, positions


# ========================================================================
# Equivariant Latent Score Network (time-conditional EGNN)
# ========================================================================

class EquivariantScoreNet(nn.Module):
    """
    Time-conditional EGNN that predicts noise ε = (ε_x, ε_h) in the
    point-structured latent space.

    Input: (z_x,t, z_h,t, t)
    Output: (eps_x, eps_h)
    """
    def __init__(self, latent_h_dim=1, h_dim=256, n_layers=9, time_embed_dim=128):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.input_h = nn.Linear(latent_h_dim, h_dim)
        self.layers = nn.ModuleList([
            EGCLLayer(h_dim, time_dim=h_dim) for _ in range(n_layers)
        ])
        self.output_h = nn.Linear(h_dim, latent_h_dim)

    def sinusoidal_embedding(self, t):
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        if t.dim() == 0:
            t = t.unsqueeze(0)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, z_x, z_h, t, mask):
        """
        z_x: (B, N, 3), z_h: (B, N, k), t: (B,), mask: (B, N)
        Returns: eps_x (B, N, 3), eps_h (B, N, k)
        """
        t_embed = self.time_mlp(self.sinusoidal_embedding(t))  # (B, h_dim)
        h = self.input_h(z_h)
        x = z_x.clone()

        x_init = x.clone()
        for layer in self.layers:
            h, x = layer(h, x, mask, t_embed=t_embed)

        eps_x = remove_mean(x - x_init, mask)  # equivariant output
        eps_h = self.output_h(h) * mask.unsqueeze(-1)  # invariant output
        return eps_x, eps_h


# ========================================================================
# GeoLDM-ULD: Full Model
# ========================================================================

class GeoLDMULD(nn.Module):
    """
    GeoLDM architecture + ULD denoising AE training.

    Stage 1: Denoising AE at small t0 (replaces KL-reg/ES-reg)
    Stage 2: Equivariant latent diffusion on (z_x, z_h)
    """
    def __init__(self, n_atom_types, max_atoms, h_dim=256, latent_h_dim=1,
                 n_enc_layers=2, n_dec_layers=4, n_score_layers=9, t0=0.05,
                 noise_schedule_type='polynomial'):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.max_atoms = max_atoms
        self.latent_h_dim = latent_h_dim
        self.t0 = t0
        self.T = 0.999
        self.noise_schedule_type = noise_schedule_type

        # VP-SDE parameters
        self.beta_min = 0.1
        self.beta_max = 20.0

        # Polynomial schedule (GeoLDM/EDM): alpha_bar = 1 - 2s - (1-2s)*t^2
        # with s = precision constant.
        self.poly_s = 1e-5
        self.poly_power = 2.0

        # Stage 1: Equivariant AE
        self.encoder = GeoEncoder(n_atom_types, h_dim, latent_h_dim, n_enc_layers)
        self.decoder = GeoDecoder(n_atom_types, h_dim, latent_h_dim, n_dec_layers)

        # Stage 2: Equivariant score network
        self.score_net = EquivariantScoreNet(
            latent_h_dim, h_dim, n_score_layers,
        )

        # Latent normalization stats
        self.register_buffer('zh_mean', torch.zeros(latent_h_dim))
        self.register_buffer('zh_std', torch.ones(latent_h_dim))
        self.register_buffer('zx_std', torch.ones(1))  # global scale for z_x

    def noise_schedule(self, t):
        t_t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        if self.noise_schedule_type == 'polynomial':
            # Polynomial alpha_bar decaying from ~1 at t=0 to ~0 at t=T,
            # matching GeoLDM/EDM. alpha_bar = (1 - 2s) * (1 - t^2) + noise_floor
            alpha_bar = (1.0 - 2.0 * self.poly_s) * (1.0 - t_t.pow(self.poly_power)) \
                        + self.poly_s
            alpha_bar = alpha_bar.clamp(min=1e-5, max=1.0 - 1e-5)
            sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
            return alpha_bar, sigma
        else:
            # VP-SDE (original)
            log_mean = -0.25 * t_t**2 * (self.beta_max - self.beta_min) - 0.5 * t_t * self.beta_min
            alpha_bar = torch.exp(2.0 * log_mean)
            sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
            return alpha_bar, sigma

    # ------------------------------------------------------------------
    # Stage 1: ULD denoising AE
    # ------------------------------------------------------------------
    def train_stage1(self, dataset, n_steps=30000, lr=1e-4, batch_size=64):
        """Denoising AE at small t0 — our ULD objective."""
        device = next(self.parameters()).device
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)

        for step in range(n_steps):
            x_flat, _ = dataset.sample_clean(batch_size)
            x_flat = x_flat.to(device)
            types, positions, mask = self._flat_to_struct(x_flat)

            # Add small noise (t0) to positions and types
            pos_noisy = positions + self.t0 * torch.randn_like(positions)
            pos_noisy = remove_mean(pos_noisy, mask)
            types_noisy = types + self.t0 * 0.1 * torch.randn_like(types)

            # Encode → decode
            z_x, z_h = self.encoder(types_noisy, pos_noisy, mask)
            types_hat, pos_hat = self.decoder(z_x, z_h, mask)

            # Reconstruction loss
            loss_pos = F.mse_loss(pos_hat * mask.unsqueeze(-1),
                                  positions * mask.unsqueeze(-1))
            loss_types = F.cross_entropy(
                types_hat[mask > 0.5], types[mask > 0.5].argmax(-1)
            )
            loss = loss_pos + loss_types

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            if (step + 1) % 500 == 0:
                print(f'  S1 [{step+1}/{n_steps}] loss={loss.item():.4f} '
                      f'pos={loss_pos.item():.4f} types={loss_types.item():.4f}')

        # Compute latent normalization stats for z_h
        self._compute_latent_stats(dataset, batch_size, device)

    def _compute_latent_stats(self, dataset, batch_size, device):
        zhs, zxs = [], []
        with torch.no_grad():
            for _ in range(10):
                x, _ = dataset.sample_clean(min(batch_size, 1000))
                x = x.to(device)
                types, pos, mask = self._flat_to_struct(x)
                z_x, z_h = self.encoder(types, pos, mask)
                zhs.append(z_h[mask > 0.5])
                zxs.append(z_x[mask > 0.5])
        zhs = torch.cat(zhs, dim=0)
        zxs = torch.cat(zxs, dim=0)
        self.zh_mean = zhs.mean(0)
        self.zh_std = zhs.std(0).clamp(min=1e-6)
        # Global scalar std for z_x (keep per-component isotropy for equivariance)
        self.zx_std = zxs.std().unsqueeze(0).clamp(min=1e-6)
        print(f'  z_h: mean={self.zh_mean.tolist()}, std={self.zh_std.tolist()}')
        print(f'  z_x: global std={self.zx_std.item():.4f}')

    def freeze_stage1(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Stage 2: Equivariant latent diffusion
    # ------------------------------------------------------------------
    def train_stage2(self, dataset, n_steps=200000, lr=1e-4, batch_size=64,
                     min_snr_gamma=5.0):
        """Latent diffusion on point-structured (z_x, z_h)."""
        device = next(self.parameters()).device
        opt = torch.optim.Adam(self.score_net.parameters(), lr=lr)

        for step in range(n_steps):
            x_flat, _ = dataset.sample_clean(batch_size)
            x_flat = x_flat.to(device)
            types, positions, mask = self._flat_to_struct(x_flat)

            with torch.no_grad():
                z_x0, z_h0 = self.encoder(types, positions, mask)
                # Normalize: z_h by per-dim mean/std, z_x by global scalar
                z_h0 = ((z_h0 - self.zh_mean) / self.zh_std) * mask.unsqueeze(-1)
                z_x0 = z_x0 / self.zx_std  # preserves equivariance (scalar rescale)
                z_x0 = remove_mean(z_x0, mask)

            # Sample t ∈ [t0, T]
            B = z_x0.shape[0]
            t = torch.rand(B, device=device) * (self.T - self.t0) + self.t0
            alpha_bar, sigma = self.noise_schedule(t)
            a = torch.sqrt(alpha_bar).view(B, 1, 1)
            s = sigma.view(B, 1, 1)

            # Forward diffusion on both z_x and z_h
            eps_x = torch.randn_like(z_x0)
            eps_x = remove_mean(eps_x, mask)  # noise must be zero-mean for positions
            eps_h = torch.randn_like(z_h0) * mask.unsqueeze(-1)

            z_x_t = a * z_x0 + s * eps_x
            z_h_t = a * z_h0 + s * eps_h

            # Predict noise
            eps_x_pred, eps_h_pred = self.score_net(z_x_t, z_h_t, t, mask)

            # Loss with Min-SNR-γ weighting
            snr = (alpha_bar / sigma.pow(2).clamp(min=1e-8)).clamp(min=1e-8)
            w = torch.minimum(snr, torch.full_like(snr, min_snr_gamma)) / snr
            w = w.view(B, 1, 1)

            loss_x = (w * ((eps_x_pred - eps_x) * mask.unsqueeze(-1)).pow(2)).mean()
            loss_h = (w * ((eps_h_pred - eps_h) * mask.unsqueeze(-1)).pow(2)).mean()
            loss = loss_x + loss_h

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
            opt.step()

            if (step + 1) % 1000 == 0:
                print(f'  S2 [{step+1}/{n_steps}] loss={loss.item():.6f} '
                      f'x={loss_x.item():.6f} h={loss_h.item():.6f}')

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n_samples, n_atoms_dist, n_steps=200, device='cpu'):
        """
        DDIM sampling in equivariant latent space, then decode.

        n_atoms_dist: dict {n_atoms: count} or list of n_atoms per sample.
        """
        self.eval()

        # Sample number of atoms per molecule
        if isinstance(n_atoms_dist, dict):
            atoms_list = []
            total = sum(n_atoms_dist.values())
            for na, count in n_atoms_dist.items():
                n_this = int(round(count / total * n_samples))
                atoms_list.extend([na] * n_this)
            while len(atoms_list) < n_samples:
                atoms_list.append(max(n_atoms_dist.keys()))
            atoms_list = atoms_list[:n_samples]
        else:
            atoms_list = list(n_atoms_dist)

        # Build masks
        N = self.max_atoms
        mask = torch.zeros(n_samples, N, device=device)
        for i, na in enumerate(atoms_list):
            mask[i, :na] = 1.0

        # Start from noise
        z_x = torch.randn(n_samples, N, 3, device=device)
        z_x = remove_mean(z_x, mask)
        z_h = torch.randn(n_samples, N, self.latent_h_dim, device=device)
        z_h = z_h * mask.unsqueeze(-1)

        ts = torch.linspace(self.T, self.t0, n_steps + 1, device=device)

        for i in range(n_steps):
            t_cur, t_next = ts[i], ts[i+1]
            t_batch = torch.full((n_samples,), t_cur.item(), device=device)

            alpha_cur, sigma_cur = self.noise_schedule(t_cur)
            alpha_next, sigma_next = self.noise_schedule(t_next)

            eps_x_pred, eps_h_pred = self.score_net(z_x, z_h, t_batch, mask)

            # DDIM step
            a_cur = torch.sqrt(alpha_cur)
            z_x0_pred = (z_x - sigma_cur * eps_x_pred) / a_cur.clamp(min=1e-6)
            z_h0_pred = (z_h - sigma_cur * eps_h_pred) / a_cur.clamp(min=1e-6)

            if i == n_steps - 1:
                z_x = remove_mean(z_x0_pred, mask)
                z_h = z_h0_pred * mask.unsqueeze(-1)
            else:
                a_next = torch.sqrt(alpha_next)
                z_x = a_next * z_x0_pred + sigma_next * eps_x_pred
                z_x = remove_mean(z_x, mask)
                z_h = (a_next * z_h0_pred + sigma_next * eps_h_pred) * mask.unsqueeze(-1)

        # Denormalize z_h and z_x
        z_h = z_h * self.zh_std + self.zh_mean
        z_x = z_x * self.zx_std
        z_x = remove_mean(z_x, mask)

        # Decode
        types_logits, positions = self.decoder(z_x, z_h, mask)

        self.train()
        return types_logits, positions, mask, atoms_list

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _flat_to_struct(self, x_flat):
        from data.molecule_3d_data import MAX_ATOMS, DIMS_PER_ATOM, N_ATOM_TYPES
        B = x_flat.shape[0]
        x = x_flat.view(B, MAX_ATOMS, DIMS_PER_ATOM)
        types = x[:, :, :N_ATOM_TYPES]
        positions = x[:, :, N_ATOM_TYPES:]
        mask = (types.sum(-1) > 0.5).float()
        return types, positions, mask
