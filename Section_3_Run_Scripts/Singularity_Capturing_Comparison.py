import os
import math
import numpy as np
import torch
from QGPINNs_Engine import ParabolicPINNSolver, NET_DTYPE, device, seed_everything, write_rows

ALPHA = 0.5
EPOCHS = 10000
SEED = 42
STRATEGY = 'gradient_ratio_v2'
VARIANTS = ('z_on', 'z_off')
R_VALUES = (1.0, 2.0, 3.0, 4.0)
ADAPTIVE_INITIAL_LAMBDA = 1.0
USE_LBFGS = True
OUT_DIR = 'z_capture_r_ablation'

N_EDGES = 4
L = 1.0
K = 2.0 * math.pi
NU = 0.5
REACTION = 10.0

PINN_PTS = 200
N_T = 100
GRADING_R = 2.0

AMPLITUDES = {
    0: 2.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
}

ARCH = dict(hidden_layers=4, hidden_dim=128, use_fourier=True,
            fourier_dim=128, fourier_sigma=2.0, fourier_sampling='gaussian')

class TreeGraph:
    nodes = [0, 1, 2, 3, 4]
    edges = [
        (0, 1, L),
        (1, 2, L),
        (1, 3, L),
        (3, 4, L),
    ]

class EdgePhysics:
    def __init__(self, edge_idx, alpha):
        self.edge_idx = edge_idx
        self.alpha = alpha
        self.nu = NU
        self.k = K
        self.c = AMPLITUDES[edge_idx]

    def F(self, x, t, u, u_x, u_xx, dt_alpha_u, f_target):
        return dt_alpha_u + u * u_x - self.nu * u_xx + REACTION * u - f_target

    def get_f_target(self, x, t, L_):
        t_safe = t.clamp(min=1e-10)
        dt_alpha_term = self.c * torch.sin(self.k * x) * math.gamma(self.alpha + 1.0)
        return (
                dt_alpha_term
                + self.c * self.c * self.k * torch.sin(self.k * x) * torch.cos(self.k * x)
                * t_safe.pow(2.0 * self.alpha)
                + self.nu * self.c * self.k * self.k * torch.sin(self.k * x) * t_safe.pow(self.alpha)
                + REACTION * self.c * torch.sin(self.k * x) * t_safe.pow(self.alpha)
        )

    def get_ic(self, x):
        return torch.zeros_like(x)

def exact_u_np(edge_idx, x_np, t_np):
    x = np.asarray(x_np).reshape(-1)
    t = np.asarray(t_np).reshape(-1)
    return AMPLITUDES[edge_idx] * np.sin(K * x) * t ** ALPHA

def make_val_funcs():
    return [
        lambda x, t, idx=i: exact_u_np(idx, x, t)
        for i in range(N_EDGES)
    ]

def verify_mms():
    t = np.linspace(0.0, 1.0, 101)
    t_pow = t ** ALPHA
    c = AMPLITUDES
    q = K
    left_flux = (-c[0] * q * math.cos(q * L) + c[1] * q + c[2] * q) * t_pow
    right_flux = (-c[2] * q * math.cos(q * L) + c[3] * q) * t_pow
    node1 = (
        c[0] * np.sin(q * L) * t_pow - c[1] * np.sin(0.0) * t_pow,
        c[0] * np.sin(q * L) * t_pow - c[2] * np.sin(0.0) * t_pow,
    )
    node3 = c[2] * np.sin(q * L) * t_pow - c[3] * np.sin(0.0) * t_pow
    residuals = {
        'node1_continuity': float(np.max(np.abs(node1[0]))),
        'node1_branch_continuity': float(np.max(np.abs(node1[1]))),
        'node1_kirchhoff': float(np.max(np.abs(left_flux))),
        'node3_continuity': float(np.max(np.abs(node3))),
        'node3_kirchhoff': float(np.max(np.abs(right_flux))),
    }
    ok = all(v <= 1e-12 for v in residuals.values())
    if not ok:
        raise RuntimeError('MMS verification failed.')
    return residuals

def build_solver(alpha=ALPHA, seed=SEED, fixed_lambda=None, arch=None, grading_r=GRADING_R, use_z=False):
    seed_everything(seed)
    graph = TreeGraph()
    physics_list = [EdgePhysics(i, alpha) for i in range(N_EDGES)]
    solver = ParabolicPINNSolver(graph, physics_list)
    bc_types = {0: 'dirichlet', 2: 'dirichlet', 4: 'dirichlet'}
    bc_values = {
        node: (lambda t: torch.zeros_like(t))
        for node in bc_types
    }
    solver.set_constraints('soft', bc_types=bc_types, bc_values=bc_values)
    solver.set_validation(make_val_funcs())
    solver.set_mesh(pinn_pts=PINN_PTS, n_t=N_T, grading_factor=grading_r)
    if arch is not None:
        solver.set_architecture(**arch)
    if use_z:
        solver.set_singularity_capture(enabled=True, xi_init=alpha, xi_loss_adaptive=True)
    solver.compile()
    if fixed_lambda is not None:
        solver.lambda_bc = float(fixed_lambda)
    return solver

def calibrate_strategies(r, seed=SEED):
    calib_solver = build_solver(seed=seed, fixed_lambda=ADAPTIVE_INITIAL_LAMBDA, grading_r=r)
    cfg_full = calib_solver.calibrate_lambda_controllers(window_epochs=EPOCHS, seed=seed)
    return {
        'gradient_ratio_v2': cfg_full['gradient_ratio_v2'],
        'bdmm_v2': cfg_full['bdmm_v2'],
    }

def build_and_train(variant, r, lam_cfg):
    use_z = (variant == 'z_on')
    solver = build_solver(seed=SEED, fixed_lambda=ADAPTIVE_INITIAL_LAMBDA, arch=ARCH,
                          grading_r=r, use_z=use_z)
    rows, global_rel, mean_rel = solver.run_strategy(
        epochs=EPOCHS, strategy=STRATEGY, initial_lambda=ADAPTIVE_INITIAL_LAMBDA,
        lam_cfg=lam_cfg, use_lbfgs=USE_LBFGS
    )
    return solver, rows, global_rel, mean_rel

def pick_best_per_variant(summaries):
    best = {}
    for variant in VARIANTS:
        cand = [s for s in summaries if s['variant'] == variant]
        best[variant] = min(cand, key=lambda s: s['global_rel_l2'])
    return best

def main():
    verify_mms()
    os.makedirs(OUT_DIR, exist_ok=True)

    summaries = []
    for r in R_VALUES:
        strategy_cfg = calibrate_strategies(r=r, seed=SEED)
        lam_cfg = strategy_cfg[STRATEGY]
        for variant in VARIANTS:
            solver, rows, global_rel, mean_rel = build_and_train(variant, r, lam_cfg)
            write_rows(os.path.join(OUT_DIR, f'{variant}_r{r:g}_history.csv'), rows)
            summaries.append({
                'variant': variant,
                'r': r,
                'global_rel_l2': global_rel,
                'mean_l2': mean_rel,
            })

    write_rows(os.path.join(OUT_DIR, 'summary.csv'), summaries)
    best_per_variant = pick_best_per_variant(summaries)

if __name__ == '__main__':
    seed_everything(SEED)
    main()