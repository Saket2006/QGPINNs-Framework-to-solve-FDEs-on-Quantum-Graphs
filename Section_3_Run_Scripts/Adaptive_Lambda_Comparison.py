import os
import csv
import math
import numpy as np
import torch
from QGPINNs_Engine import ParabolicPINNSolver, NET_DTYPE, device, seed_everything, write_rows

ALPHA = 0.5
EPOCHS = 10000
SEED = 42
RUN_MODE = 'strategies'
STRATEGIES = ('dual_v2', 'gradient_ratio_v2', 'bdmm_v2', 'fixed')
FIXED_LAMBDAS = (1, 100.0)
ADAPTIVE_INITIAL_LAMBDA = 1.0
USE_LBFGS = True
OUT_DIR = 'adaptive_lambda_calibrated'

N_EDGES = 4
L = 1.0
K = 6.0 * math.pi
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
        gamma_coeff = math.gamma(3.0) / math.gamma(3.0 - self.alpha)
        return (
                self.c * torch.sin(self.k * x)
                * gamma_coeff * t_safe.pow(2.0 - self.alpha)
                + self.c * self.c * self.k * torch.sin(self.k * x) * torch.cos(self.k * x) * t_safe.pow(4.0)
                + self.nu * self.c * self.k * self.k * torch.sin(self.k * x) * t_safe.pow(2.0)
                + REACTION * self.c * torch.sin(self.k * x) * t_safe.pow(2.0)
        )

    def get_ic(self, x):
        return torch.zeros_like(x)

def exact_u_np(edge_idx, x_np, t_np):
    x = np.asarray(x_np).reshape(-1)
    t = np.asarray(t_np).reshape(-1)
    return AMPLITUDES[edge_idx] * np.sin(K * x) * t ** 2

def make_val_funcs():
    return [
        lambda x, t, idx=i: exact_u_np(idx, x, t)
        for i in range(N_EDGES)
    ]

def verify_mms():
    t = np.linspace(0.0, 1.0, 101)
    t2 = t ** 2
    c = AMPLITUDES
    q = K
    left_flux = (-c[0] * q * math.cos(q * L) + c[1] * q + c[2] * q) * t2
    right_flux = (-c[2] * q * math.cos(q * L) + c[3] * q) * t2
    node1 = (
        c[0] * np.sin(q * L) * t2 - c[1] * np.sin(0.0) * t2,
        c[0] * np.sin(q * L) * t2 - c[2] * np.sin(0.0) * t2,
    )
    node3 = c[2] * np.sin(q * L) * t2 - c[3] * np.sin(0.0) * t2
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

def build_solver(alpha=ALPHA, seed=SEED, fixed_lambda=None, arch=None):
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
    solver.set_mesh(pinn_pts=PINN_PTS, n_t=N_T, grading_factor=GRADING_R)
    if arch is not None:
        solver.set_architecture(**arch)
    solver.compile()
    if fixed_lambda is not None:
        solver.lambda_bc = float(fixed_lambda)
    return solver

def calibrate_strategies(seed=SEED):
    calib_solver = build_solver(seed=seed, fixed_lambda=ADAPTIVE_INITIAL_LAMBDA)
    cfg_full = calib_solver.calibrate_lambda_controllers(window_epochs=EPOCHS, seed=seed)
    return {
        'dual_v2': {
            'bdmm_v2': cfg_full['bdmm_v2'],
            'gradient_ratio_v2': cfg_full['gradient_ratio_v2'],
        },
        'gradient_ratio_v2': cfg_full['gradient_ratio_v2'],
        'bdmm_v2': cfg_full['bdmm_v2'],
    }

def train_experiment(strategy, initial_lambda, lam_cfg=None):
    solver = build_solver(seed=SEED, fixed_lambda=initial_lambda)
    rows, global_rel, mean_rel = solver.run_strategy(
        epochs=EPOCHS, strategy=strategy, initial_lambda=initial_lambda, lam_cfg=lam_cfg, use_lbfgs=USE_LBFGS
    )
    return solver, rows, global_rel, mean_rel

def main():
    verify_mms()
    os.makedirs(OUT_DIR, exist_ok=True)
    summaries = []

    strategy_cfg = calibrate_strategies(seed=SEED)

    if RUN_MODE == 'fixed_lambda_sweep':
        for lam in FIXED_LAMBDAS:
            solver, rows, global_rel, mean_rel = train_experiment('fixed', lam)
            write_rows(os.path.join(OUT_DIR, f'fixed_{lam:g}_history.csv'), rows)
            summaries.append({
                'strategy': 'fixed',
                'initial_lambda': lam,
                'final_lambda': solver.lambda_bc,
                'global_rel_l2': global_rel,
                'mean_l2': mean_rel,
            })
    else:
        for strategy in STRATEGIES:
            if strategy == 'fixed':
                for lam in FIXED_LAMBDAS:
                    solver, rows, global_rel, mean_rel = train_experiment('fixed', lam)
                    summaries.append({
                        'strategy': 'fixed',
                        'initial_lambda': lam,
                        'final_lambda': solver.lambda_bc,
                        'global_rel_l2': global_rel,
                        'mean_l2': mean_rel,
                    })
            else:
                initial = ADAPTIVE_INITIAL_LAMBDA
                solver, rows, global_rel, mean_rel = train_experiment(strategy, initial, lam_cfg=strategy_cfg[strategy])
                write_rows(os.path.join(OUT_DIR, f'{strategy}_history.csv'), rows)
                summaries.append({
                    'strategy': strategy,
                    'initial_lambda': initial,
                    'final_lambda': solver.lambda_bc,
                    'global_rel_l2': global_rel,
                    'mean_l2': mean_rel,
                })

    write_rows(os.path.join(OUT_DIR, 'summary.csv'), summaries)

if __name__ == '__main__':
    seed_everything(SEED)
    main()