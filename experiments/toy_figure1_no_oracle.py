"""Figure 1 without the oracle: same toy run, only observable diagnostics.

The published Figure 1 monitors three quantities that require the ground-truth
manifold: the loss split by TRUE_A, the analytic MoG score, and the response
along the true normal directions. A practitioner has none of these.

This script reruns the identical experiment (same data, net, schedule, no SVD,
rank-free idempotence penalty) and records only quantities computable from the
model and the data:

  M = W W^T               the learned operator; collapse means M -> a projector
  spec(M)                 its eigenvalues -- the gap *estimates* the intrinsic
                          dimension instead of consuming it
  d_hat                   largest-eigengap rank estimate (no threshold, no k)
  P_hat                   projector onto the model's own top-d_hat eigenvectors
  DSM residual split      ||r P_hat||^2 and ||r (I-P_hat)||^2, which sum exactly
                          to the observed DSM loss
  normal response         learned linear response along the model's OWN inferred
                          normal directions, vs the -alpha/h prediction (h is a
                          known hyperparameter, not ground truth)

TRUE_A / TRUE_MEANS are used only to synthesize data and, optionally, to print
a post-hoc accuracy check that no panel depends on.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'logs', 'figure1_no_oracle')
os.makedirs(OUT_DIR, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--regularizer', choices=['none', 'l2', 'projection'],
                default='projection')
ap.add_argument('--reg-weight', type=float, default=10.0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--log-every', type=int, default=10)
args = ap.parse_args()

d, k_true, m = 100, 5, 200
sigma_t = 0.1
h_t = sigma_t ** 2
batch_size = 4096
lr_stage1, lr_stage2 = 0.001, 0.005
steps_stage1 = steps_stage2 = 1000

torch.manual_seed(args.seed)

# ---- ground truth: used ONLY to synthesize data -------------------------
TRUE_A, _ = torch.linalg.qr(torch.randn(d, k_true))
TRUE_P = TRUE_A @ TRUE_A.T
TRUE_MEANS = torch.tensor([[2.5, 0, 0, 0, 0], [-2.5, 2.0, 0, 0, 0],
                           [0, -2.5, 0, 0, 0]])
TRUE_WTS = torch.tensor([0.3, 0.3, 0.4])


def get_batch(n):
    idx = torch.multinomial(TRUE_WTS, n, replacement=True)
    z = TRUE_MEANS[idx] + torch.randn(n, k_true) * 0.5
    x0 = z @ TRUE_A.T
    noise = torch.randn(n, d)
    return x0 + sigma_t * noise, -noise / sigma_t, z, x0


class DecomposedScoreNet(nn.Module):
    def __init__(self, d, m, h_t):
        super().__init__()
        self.h_t = h_t
        self.W = nn.Parameter(torch.randn(d, m) * 0.02)
        self.V = nn.Parameter(torch.eye(m))
        self.bias = nn.Parameter(torch.zeros(1, m))

    def forward(self, x):
        u = x @ self.W + self.bias
        s_nonlin = (F.relu(u) @ self.V.T) @ self.W.T
        s_lin = ((x @ self.W) / self.h_t) @ self.W.T
        return s_nonlin + s_lin - x / self.h_t

    @torch.no_grad()
    def linear_part(self, x):
        return ((x @ self.W) / self.h_t) @ self.W.T - x / self.h_t


model = DecomposedScoreNet(d, m, h_t)

# fixed held-out probe batch so the curves are comparable across steps
torch.manual_seed(args.seed + 999)
HO_X, HO_TGT, _, HO_X0 = get_batch(4096)
torch.manual_seed(args.seed)


@torch.no_grad()
def observables():
    """Everything here is computable without the ground-truth manifold."""
    M = (model.W @ model.W.T).double()
    lam, U = torch.linalg.eigh(M)
    lam = lam.flip(0).clamp(min=0)          # descending
    U = U.flip(1)
    gaps = lam[:-1] - lam[1:]
    d_hat = int(torch.argmax(gaps[:d // 2]).item()) + 1      # largest-eigengap
    n_above_half = int((lam > 0.5).sum().item())
    P_hat = (U[:, :d_hat] @ U[:, :d_hat].T).float()

    pred = model(HO_X)
    r = pred - HO_TGT
    denom = r.numel()
    r_tan = r @ P_hat
    r_nrm = r - r_tan
    Mf = M.float()
    proj_defect = torch.linalg.matrix_norm(Mf @ Mf - Mf) / \
        torch.linalg.matrix_norm(Mf).clamp(min=1e-12)
    return {
        'dsm_holdout': (r.pow(2).sum() / denom).item(),
        'res_tangent': (r_tan.pow(2).sum() / denom).item(),
        'res_normal': (r_nrm.pow(2).sum() / denom).item(),
        'd_hat': d_hat,
        'n_eig_above_half': n_above_half,
        'proj_defect': proj_defect.item(),
        'lam': lam.float().numpy(),
        'U': U.float().numpy(),
    }


hist = {k: [] for k in ('step', 'dsm_train', 'dsm_holdout', 'res_tangent',
                        'res_normal', 'd_hat', 'n_eig_above_half',
                        'proj_defect')}
spectra = {}


def log(step):
    o = observables()
    hist['step'].append(step)
    for k in ('dsm_holdout', 'res_tangent', 'res_normal', 'd_hat',
              'n_eig_above_half', 'proj_defect'):
        hist[k].append(o[k])
    return o


print(f'd={d}, true k={k_true} (data synthesis only), width m={m}, '
      f'sigma={sigma_t}, h={h_t}')
print(f'regularizer={args.regularizer} weight={args.reg_weight} '
      f'seed={args.seed}\n')

o0 = log(0)
hist['dsm_train'].append(float('nan'))
spectra['init'] = o0['lam']
print(f"step    0 | holdout {o0['dsm_holdout']:8.3f} | d_hat {o0['d_hat']:3d} "
      f"| projector defect {o0['proj_defect']:.4f}")

model.V.requires_grad = False
model.bias.requires_grad = False
opt1 = torch.optim.Adam([model.W], lr=lr_stage1)
for s in range(1, steps_stage1 + 1):
    xt, target, _, _ = get_batch(batch_size)
    dsm = F.mse_loss(model(xt), target)
    if args.regularizer == 'projection':
        P = model.W @ model.W.T
        reg = (P @ P - P).pow(2).sum()
    elif args.regularizer == 'l2':
        reg = model.W.pow(2).sum()
    else:
        reg = torch.zeros((), device=model.W.device)
    loss = dsm + args.reg_weight * reg
    opt1.zero_grad(); loss.backward(); opt1.step()
    if s % args.log_every == 0:
        o = log(s)
        hist['dsm_train'].append(dsm.item())
        if s % 200 == 0:
            print(f'step {s:4d} | holdout {o["dsm_holdout"]:8.3f} | '
                  f'tan {o["res_tangent"]:8.3f} | nrm {o["res_normal"]:8.3f} | '
                  f'd_hat {o["d_hat"]:3d} | #lam>.5 {o["n_eig_above_half"]:3d} | '
                  f'defect {o["proj_defect"]:.4f}')

spectra['stage1_end'] = observables()['lam']

with torch.no_grad():
    proj_x = get_batch(2000)[0] @ model.W
    model.bias.copy_(-proj_x.mean(0) + torch.randn(1, m) * proj_x.std(0))

model.W.requires_grad = False
model.V.requires_grad = True
model.bias.requires_grad = True
opt2 = torch.optim.Adam([model.V, model.bias], lr=lr_stage2)
print()
for s in range(1, steps_stage2 + 1):
    xt, target, _, _ = get_batch(batch_size)
    dsm = F.mse_loss(model(xt), target)
    opt2.zero_grad(); dsm.backward(); opt2.step()
    if s % args.log_every == 0:
        o = log(steps_stage1 + s)
        hist['dsm_train'].append(dsm.item())
        if s % 200 == 0:
            print(f'step {steps_stage1 + s:4d} | holdout {o["dsm_holdout"]:8.3f} | '
                  f'tan {o["res_tangent"]:8.3f} | nrm {o["res_normal"]:8.3f} | '
                  f'd_hat {o["d_hat"]:3d} | #lam>.5 {o["n_eig_above_half"]:3d} | '
                  f'defect {o["proj_defect"]:.4f}')

fin = observables()
spectra['stage2_end'] = fin['lam']

# ---- normal-direction response, using the MODEL's own inferred normals ----
U_fin = torch.tensor(fin['U'])
d_hat = fin['d_hat']
normal_dir = U_fin[:, -1]                       # smallest-eigenvalue direction
tangent_dir = U_fin[:, 0]                       # largest-eigenvalue direction
alphas = torch.linspace(-0.5, 0.5, 400)
with torch.no_grad():
    base, _, _, _ = get_batch(400)
    pts = base + alphas.unsqueeze(1) * normal_dir.unsqueeze(0)
    lin = model.linear_part(pts)
    normal_response = (lin @ normal_dir).numpy()
    # learned score profile along the model's own leading tangent direction
    scan = torch.linspace(-6, 6, 300).unsqueeze(1) * tangent_dir.unsqueeze(0)
    learned_profile = (model(scan) @ tangent_dir).numpy()
    # empirical-Bayes (Tweedie) reference built ONLY from held-out clean data
    ref_x0 = get_batch(20000)[3]
    logits = -(torch.cdist(scan, ref_x0) ** 2) / (2 * sigma_t ** 2)
    wts = torch.softmax(logits, dim=1)
    emp_x0 = wts @ ref_x0
    emp_profile = (((emp_x0 - scan) / h_t) @ tangent_dir).numpy()

# post-hoc accuracy check -- printed only, no panel depends on it
true_rank_err = float(torch.linalg.matrix_norm(
    (model.W @ model.W.T) - TRUE_P).item())
print(f'\n[post-hoc only] d_hat={d_hat} (true k={k_true})  '
      f'||WW^T - P_true||_F = {true_rank_err:.4f}')

np.savez(os.path.join(OUT_DIR, f'fig1_no_oracle_seed{args.seed}.npz'),
         alphas=alphas.numpy(), normal_response=normal_response,
         scan_coord=torch.linspace(-6, 6, 300).numpy(),
         learned_profile=learned_profile, emp_profile=emp_profile,
         h_t=h_t, final_d_hat=d_hat, k_true=k_true,
         steps_stage1=steps_stage1,
         **{f'spec_{k}': v for k, v in spectra.items()},
         **{k: np.asarray(v, dtype=float) for k, v in hist.items()})
with open(os.path.join(OUT_DIR, f'fig1_no_oracle_seed{args.seed}.json'), 'w') as f:
    json.dump({'config': vars(args), 'd': d, 'k_true_data_synthesis_only': k_true,
               'm': m, 'sigma_t': sigma_t, 'h_t': h_t,
               'final_d_hat': d_hat,
               'final_proj_defect': fin['proj_defect'],
               'final_holdout_dsm': fin['dsm_holdout'],
               'final_res_tangent': fin['res_tangent'],
               'final_res_normal': fin['res_normal'],
               'posthoc_WWt_to_Ptrue_fro': true_rank_err}, f, indent=2)
print(f'Saved npz + json to {OUT_DIR}')
