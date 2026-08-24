import sys
import os
import gc
import random
import traceback
import numpy as np
import torch
from scipy.special import gamma

sys.path.insert(0, '/kaggle/working')
from PINN_Solver import EllipticPINNSolver, device

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BranchedTadpoleGraph:
    nodes = (0, 1, 2, 3)

    def __init__(self, L0=2.0, L1=1.0, L2=1.0, L3=1.0):
        self.edges = [(0, 0, L0), (0, 1, L1), (1, 2, L2), (0, 3, L3)]

B3_VAL = 1.6 * (2.0 ** 0.6) - 1.0

def get_A2_for_dirichlet(target_val=4.0):
    return target_val - 5.6

def get_exact_u(edge_idx, x_np, A2=0.0):
    if edge_idx == 0:
        return x_np ** 1.6 - (2.0 ** 0.6) * x_np
    elif edge_idx == 1:
        return x_np ** 1.6 + x_np
    elif edge_idx == 2:
        return x_np ** 1.6 + A2 * x_np ** 2 + 2.6 * x_np + 2.0
    elif edge_idx == 3:
        return x_np ** 1.6 + B3_VAL * x_np
    return np.zeros_like(x_np)

def get_exact_du(edge_idx, x_np, A2=0.0):
    if edge_idx == 0:
        return 1.6 * x_np ** 0.6 - (2.0 ** 0.6)
    elif edge_idx == 1:
        return 1.6 * x_np ** 0.6 + 1.0
    elif edge_idx == 2:
        return 1.6 * x_np ** 0.6 + 2.0 * A2 * x_np + 2.6
    elif edge_idx == 3:
        return 1.6 * x_np ** 0.6 + B3_VAL
    return np.zeros_like(x_np)

def get_exact_D_alpha(edge_idx, x_np, A2=0.0):
    term_16 = (gamma(2.6) / gamma(1.1)) * (x_np ** 0.1)
    if edge_idx == 2:
        term_2 = A2 * (gamma(3.0) / gamma(1.5)) * (x_np ** 0.5)
        return term_16 + term_2
    return term_16

def get_exact_D_beta(edge_idx, x_np, A2=0.0, B3_VAL=0.0):
    term_16 = (gamma(2.6) / gamma(2.1)) * (x_np ** 1.1)
    if edge_idx == 0:
        return term_16 - (2.0 ** 0.6) * (gamma(2.0) / gamma(1.5)) * (x_np ** 0.5)
    elif edge_idx == 1:
        return term_16 + 1.0 * (gamma(2.0) / gamma(1.5)) * (x_np ** 0.5)
    elif edge_idx == 2:
        term_2 = A2 * (gamma(3.0) / gamma(2.5)) * (x_np ** 1.5)
        term_1 = 2.6 * (gamma(2.0) / gamma(1.5)) * (x_np ** 0.5)
        return term_16 + term_2 + term_1
    elif edge_idx == 3:
        return term_16 + B3_VAL * (gamma(2.0) / gamma(1.5)) * (x_np ** 0.5)

class NonlinearEllipticPhysics:
    def __init__(self, alpha=1.5, beta=0.5, V=1.0, edge_idx=0, A2=0.0):
        self.alpha = alpha
        self.beta = beta
        self.V = V
        self.edge_idx = edge_idx
        self.A2 = A2
        self.B3_VAL = 1.6 * (2.0 ** 0.6) - 1.0

    def F(self, x, u, u_x, d_beta_u, f_target):
        V_tensor = self.V if torch.is_tensor(self.V) else torch.tensor(self.V, dtype=x.dtype, device=x.device)
        return f_target - V_tensor * u * u_x + (d_beta_u) ** 2

    def get_f_target(self, x_reg, x_nsig, L, Da_fd, Db_fd):
        u_sp = get_exact_u(self.edge_idx, x_nsig, self.A2)
        u_x_sp = get_exact_du(self.edge_idx, x_nsig, self.A2)
        D_alpha_u_sp = get_exact_D_alpha(self.edge_idx, x_nsig, self.A2)
        D_beta_u_sp = get_exact_D_beta(self.edge_idx, x_nsig, self.A2, self.B3_VAL)
        return D_alpha_u_sp + self.V * u_sp * u_x_sp - (D_beta_u_sp) ** 2

def clear_gpu(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def is_oom_error(e):
    msg = str(e).lower()
    return 'out of memory' in msg or 'cuda oom' in msg or isinstance(e, getattr(torch.cuda, 'OutOfMemoryError', ()))

PTS_PER_UNIT = 200
EPOCHS = 10000
N_STARTS = 3
PROBE_EPOCHS = 500
LEAF_VAL = 4.0
A2_VAL = get_A2_for_dirichlet(LEAF_VAL)
LEAF_ANCHOR_VAL = 1.0 + B3_VAL

VARIANTS = {
    'Variant 1': dict(strategy='fixed', use_singularity_capture=False,
                     use_fourier=False, fourier_dim=64, fourier_sigma=1.0),
    'Variant 2': dict(strategy='dual_v2', use_singularity_capture=False,
                    use_fourier=False, fourier_dim=64, fourier_sigma=1.0),
    'Variant 3': dict(strategy='dual_v2', use_singularity_capture=True,
                         use_fourier=False, fourier_dim=64, fourier_sigma=1.0),
    'Variant 4': dict(strategy='dual_v2', use_singularity_capture=True,
                                 use_fourier=True, fourier_dim=128, fourier_sigma=1.0),
}

OUT_DIR = os.path.join('/kaggle/working', "tadpole_variants")

def build_solver(cfg):
    graph = BranchedTadpoleGraph()
    physics_list = [NonlinearEllipticPhysics(1.5, 0.5, 1.0, i, A2_VAL) for i in range(4)]
    solver = EllipticPINNSolver(graph, physics_list)
    solver.set_frac_scheme('L21sigma')

    solver.set_architecture(
        hidden_layers=4,
        hidden_dim=128,
        use_fourier=cfg['use_fourier'],
        fourier_dim=cfg['fourier_dim'],
        fourier_sigma=cfg['fourier_sigma']
    )
    solver.set_mesh(pts_per_unit=PTS_PER_UNIT, grading_factor=2)

    solver.set_singularity_capture(
        enabled=cfg['use_singularity_capture'],
        xi_init=1.5 if cfg['use_singularity_capture'] else None,
        xi_loss_adaptive=True
    )
    solver.set_lr(lr=0.0005, min_lr=1e-06, xi_lr_scale=0.1, xi_lr_scale_phase2=0.01)
    solver.set_constraints(
        constraint_mode='soft',
        bc_types={2: 'dirichlet', 3: 'dirichlet'},
        bc_values={2: LEAF_VAL, 3: LEAF_ANCHOR_VAL}
    )
    exact_funcs = [lambda x, L=None, i=i, A=A2_VAL: get_exact_u(i, x, A) for i in range(4)]
    solver.set_validation(exact_funcs)
    return solver, exact_funcs

def run_variant(run_name, cfg, seed=42):
    seed_everything(seed)
    clear_gpu()

    result = dict(run_name=run_name, cfg=cfg, oom=False, error=None,
                  errors_rand=None, errors_colloc=None)

    solver = None
    try:
        solver, exact_funcs = build_solver(cfg)
        solver.compile()

        lam_cfg = None
        if cfg['strategy'] != 'fixed':
            cfg_full = solver.calibrate_lambda_controllers(window_epochs=EPOCHS, seed=seed)
            if cfg['strategy'] == 'dual_v2':
                lam_cfg = {
                    'bdmm_v2': cfg_full['bdmm_v2'],
                    'gradient_ratio_v2': cfg_full['gradient_ratio_v2'],
                }
            else:
                lam_cfg = cfg_full.get(cfg['strategy'])

        solver.train_multistart(epochs=EPOCHS, strategy=cfg['strategy'], use_lbfgs=True,
                                n_starts=N_STARTS, seed_base=seed, probe_epochs=PROBE_EPOCHS,
                                lam_cfg=lam_cfg)

        errors_rand, errors_colloc = solver.report_l2(exact_funcs)
        result['errors_rand'] = [float(e) for e in errors_rand]
        result['errors_colloc'] = [float(e) for e in errors_colloc]
        os.makedirs(OUT_DIR, exist_ok=True)
        ckpt = {
            'all_nets': {i: m['net'].state_dict() for i, m in solver.models.items()},
            'xi_raw': solver.xi_raw.detach().cpu() if cfg['use_singularity_capture'] else None,
            'lambda_bc': solver.lambda_bc,
            'l2_colloc': result['errors_colloc'],
            'l2_rand': result['errors_rand'],
            'config': cfg,
        }
        path = os.path.join(OUT_DIR, f"{run_name}.pt")
        torch.save(ckpt, path)

    except Exception as e:
        if is_oom_error(e):
            result['oom'] = True
        result['error'] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        clear_gpu(solver)

    return result

if __name__ == '__main__':
    results = []
    for name, cfg in VARIANTS.items():
        results.append(run_variant(name, cfg))