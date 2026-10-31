"""
E(n) Equivariant Graph Neural Network (EGNN) layers for 3D molecule
generation, following Satorras et al. 2021.

Provides:
  - EGNNLayer: single equivariant message-passing layer
  - EquivariantEncoder: EGNN stack → global mean pool → invariant latent z
  - EquivariantDecoder: z → broadcast → EGNN stack → (positions, types)

These are used by the 3D ULD and 3D LDM models to build E(3)-equivariant
autoencoders. The latent z is invariant, so the score network in latent
space is a standard MLP (LatentScoreNet from two_stage_latent.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EGNNLayer(nn.Module):
    """
    Single E(n)-equivariant message passing layer.

    Updates node features h (invariant) and positions x (equivariant):
      m_ij = φ_e(h_i, h_j, ||x_i - x_j||²)
      x_i' = x_i + Σ_j (x_i - x_j) · ψ(m_ij)
      h_i' = h_i + φ_h(h_i, Σ_j m_ij)

    Fully connected (all pairs) — suitable for small molecules (N ≤ 29).
    """
    def __init__(self, h_dim):
        super().__init__()
        # Edge MLP: (h_i, h_j, d²) → message
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * h_dim + 1, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
        )
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
        # Coordinate update MLP (outputs scalar weight per edge)
        self.coord_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 1, bias=False),
        )
        # Small init for coordinate updates (stability)
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.01)

    def forward(self, h, x, mask=None):
        """
        h: (B, N, h_dim) node features
        x: (B, N, 3) positions
        mask: (B, N) binary mask (1 = real atom, 0 = padding)
        Returns updated (h', x').
        """
        B, N, _ = h.shape

        # Pairwise differences and squared distances
        dx = x.unsqueeze(2) - x.unsqueeze(1)       # (B, N, N, 3)
        d2 = (dx ** 2).sum(-1, keepdim=True)         # (B, N, N, 1)

        # Edge features
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, h_dim)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, h_dim)
        edge_feat = torch.cat([hi, hj, d2], dim=-1)  # (B, N, N, 2*h_dim+1)
        m_ij = self.edge_mlp(edge_feat)               # (B, N, N, h_dim)

        # Mask edges involving padding atoms
        if mask is not None:
            edge_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # (B, N, N)
            # Zero out self-loops too
            eye = torch.eye(N, device=h.device).unsqueeze(0)
            edge_mask = edge_mask * (1 - eye)
            m_ij = m_ij * edge_mask.unsqueeze(-1)

        # Coordinate update (equivariant) — normalized + clamped for stability
        w = self.coord_mlp(m_ij)                      # (B, N, N, 1)
        n_edges = edge_mask.sum(dim=2, keepdim=True).clamp(min=1) if mask is not None \
                  else torch.full((B, N, 1), N - 1, device=h.device)
        x_update = (dx * w).sum(dim=2) / n_edges      # (B, N, 3)
        x_update = torch.clamp(x_update, -5.0, 5.0)   # prevent explosion
        x_new = x + x_update

        # Node feature update (invariant) — normalized
        agg = m_ij.sum(dim=2) / n_edges                # (B, N, h_dim)
        h_input = torch.cat([h, agg], dim=-1)          # (B, N, 2*h_dim)
        h_new = h + self.node_mlp(h_input)

        # Zero out padding
        if mask is not None:
            h_new = h_new * mask.unsqueeze(-1)
            x_new = x_new * mask.unsqueeze(-1)

        return h_new, x_new


class EquivariantEncoder(nn.Module):
    """
    EGNN stack → global mean pool → invariant latent z ∈ R^{latent_dim}.

    Input: atom types (B, N, n_types) one-hot + positions (B, N, 3)
    Output: z (B, latent_dim) — invariant under E(3).
    """
    def __init__(self, n_types, h_dim=128, latent_dim=64, n_layers=4):
        super().__init__()
        self.type_embed = nn.Linear(n_types, h_dim)
        self.layers = nn.ModuleList([EGNNLayer(h_dim) for _ in range(n_layers)])
        self.pool_fc = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, latent_dim),
        )

    def forward(self, types, positions, mask):
        """
        types: (B, N, n_types) one-hot
        positions: (B, N, 3)
        mask: (B, N) binary
        Returns: z (B, latent_dim) invariant.
        """
        h = self.type_embed(types)  # (B, N, h_dim)
        x = positions
        for layer in self.layers:
            h, x = layer(h, x, mask)
        # Global mean pool over non-padding atoms → invariant
        h_masked = h * mask.unsqueeze(-1)
        n_atoms = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        z = h_masked.sum(dim=1) / n_atoms  # (B, h_dim)
        return self.pool_fc(z)  # (B, latent_dim)


class NodeLevelEncoder(nn.Module):
    """
    EGNN stack → per-node invariant features → flatten.

    Instead of global pooling, each atom keeps its own z_i ∈ R^{node_dim}.
    The full latent is the concatenation: z = [z_1, ..., z_N] ∈ R^{N * node_dim}.
    This preserves much more structural information than global pooling.

    Output: z (B, max_atoms * node_dim) — invariant under E(3).
    """
    def __init__(self, n_types, h_dim=64, node_dim=16, n_layers=4, max_atoms=29):
        super().__init__()
        self.max_atoms = max_atoms
        self.node_dim = node_dim
        self.latent_dim = max_atoms * node_dim  # total flat latent size
        self.type_embed = nn.Linear(n_types, h_dim)
        self.layers = nn.ModuleList([EGNNLayer(h_dim) for _ in range(n_layers)])
        self.project = nn.Linear(h_dim, node_dim)

    def forward(self, types, positions, mask):
        h = self.type_embed(types)
        x = positions
        for layer in self.layers:
            h, x = layer(h, x, mask)
        z_nodes = self.project(h)  # (B, N, node_dim)
        z_nodes = z_nodes * mask.unsqueeze(-1)  # zero out padding
        return z_nodes.view(-1, self.latent_dim)  # (B, N*node_dim)


class NodeLevelDecoder(nn.Module):
    """
    Flat latent → reshape to per-node features → EGNN → (types, positions).

    The per-node z_i gives the decoder local structural information for
    each atom. Positions are initialized randomly (centered at origin)
    and refined by EGNN layers using the z_i-informed node features.
    """
    def __init__(self, n_types, h_dim=64, node_dim=16, n_layers=4, max_atoms=29):
        super().__init__()
        self.max_atoms = max_atoms
        self.node_dim = node_dim
        self.latent_dim = max_atoms * node_dim
        self.unproject = nn.Linear(node_dim, h_dim)
        self.coord_init = nn.Sequential(
            nn.Linear(node_dim, 32), nn.SiLU(),
            nn.Linear(32, 3),
        )
        self.layers = nn.ModuleList([EGNNLayer(h_dim) for _ in range(n_layers)])
        self.type_out = nn.Linear(h_dim, n_types)

    def forward(self, z_flat, mask):
        B = z_flat.shape[0]
        z_nodes = z_flat.view(B, self.max_atoms, self.node_dim)  # (B, N, node_dim)
        h = self.unproject(z_nodes)        # (B, N, h_dim)
        x = self.coord_init(z_nodes)       # (B, N, 3)
        # Center positions
        n_atoms = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        x_masked = x * mask.unsqueeze(-1)
        com = x_masked.sum(dim=1, keepdim=True) / n_atoms
        x = x - com

        for layer in self.layers:
            h, x = layer(h, x, mask)

        types = self.type_out(h)
        return types, x


class NodeLevelVAEEncoder(nn.Module):
    """Same as NodeLevelEncoder but outputs per-node (mu, logvar)."""
    def __init__(self, n_types, h_dim=64, node_dim=16, n_layers=4, max_atoms=29):
        super().__init__()
        self.max_atoms = max_atoms
        self.node_dim = node_dim
        self.latent_dim = max_atoms * node_dim
        self.type_embed = nn.Linear(n_types, h_dim)
        self.layers = nn.ModuleList([EGNNLayer(h_dim) for _ in range(n_layers)])
        self.proj_mu = nn.Linear(h_dim, node_dim)
        self.proj_logvar = nn.Linear(h_dim, node_dim)

    def forward(self, types, positions, mask):
        h = self.type_embed(types)
        x = positions
        for layer in self.layers:
            h, x = layer(h, x, mask)
        mu = self.proj_mu(h) * mask.unsqueeze(-1)
        logvar = self.proj_logvar(h) * mask.unsqueeze(-1)
        return mu.view(-1, self.latent_dim), logvar.view(-1, self.latent_dim)

    def encode_deterministic(self, types, positions, mask):
        mu, _ = self.forward(types, positions, mask)
        return mu


class EquivariantDecoder(nn.Module):
    """
    Invariant latent z → EGNN refinement → (positions, type_logits).

    Strategy: broadcast z to each atom slot via MLP → initial h and x,
    then refine with EGNN layers. The decoder is equivariant: if you
    rotate the initial x, the output positions rotate the same way.
    Since the initial x is generated from z (invariant), the output
    positions are determined up to rotation — which is the correct
    behavior for unconditional 3D generation.
    """
    def __init__(self, latent_dim, n_types, h_dim=128, n_layers=4,
                 max_atoms=29):
        super().__init__()
        self.max_atoms = max_atoms
        self.h_dim = h_dim
        self.fc_h = nn.Sequential(
            nn.Linear(latent_dim, h_dim * max_atoms),
        )
        self.fc_x = nn.Sequential(
            nn.Linear(latent_dim, 3 * max_atoms),
        )
        self.layers = nn.ModuleList([EGNNLayer(h_dim) for _ in range(n_layers)])
        self.type_out = nn.Linear(h_dim, n_types)
        self.coord_out = nn.Linear(3, 3)  # optional linear refinement

    def forward(self, z, mask):
        """
        z: (B, latent_dim) invariant
        mask: (B, N) binary
        Returns: types (B, N, n_types) logits, positions (B, N, 3)
        """
        B = z.shape[0]
        N = self.max_atoms

        h = self.fc_h(z).view(B, N, self.h_dim)
        x = self.fc_x(z).view(B, N, 3)
        # Center positions
        n_atoms = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        x_masked = x * mask.unsqueeze(-1)
        com = x_masked.sum(dim=1, keepdim=True) / n_atoms
        x = x - com

        for layer in self.layers:
            h, x = layer(h, x, mask)

        types = self.type_out(h)        # (B, N, n_types)
        positions = self.coord_out(x)   # (B, N, 3)
        return types, positions


class VAEEquivariantEncoder(nn.Module):
    """Same EGNN encoder but outputs (mu, logvar) for VAE training."""
    def __init__(self, n_types, h_dim=128, latent_dim=64, n_layers=4):
        super().__init__()
        self.type_embed = nn.Linear(n_types, h_dim)
        self.layers = nn.ModuleList([EGNNLayer(h_dim) for _ in range(n_layers)])
        self.pool_mu = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, latent_dim),
        )
        self.pool_logvar = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, latent_dim),
        )

    def forward(self, types, positions, mask):
        h = self.type_embed(types)
        x = positions
        for layer in self.layers:
            h, x = layer(h, x, mask)
        h_masked = h * mask.unsqueeze(-1)
        n_atoms = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = h_masked.sum(dim=1) / n_atoms
        return self.pool_mu(pooled), self.pool_logvar(pooled)

    def encode_deterministic(self, types, positions, mask):
        mu, _ = self.forward(types, positions, mask)
        return mu
