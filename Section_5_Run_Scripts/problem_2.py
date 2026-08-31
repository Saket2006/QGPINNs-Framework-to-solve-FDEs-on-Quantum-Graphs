import sys
import os
import gc
import math
import random
import traceback
import numpy as np
import torch
from scipy.special import gamma


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Engine'))
from QGPINNs_Engine import ParabolicPINNSolver, device


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
        self.edges = [
            (0, 0, L0),
            (0, 1, L1),
            (1, 2, L2),
            (0, 3, L3),
        ]


ALPHA = 0.5
V_COEFF = 1.0

GAMMA_A1 = float(gamma(ALPHA + 1.0))

EDGE_COEFS = {
    0: dict(a=1.0, b=1.0),
    1: dict(a=1.0, b=1.0),
    2: dict(a=-1.0, b=-1.0),
    3: dict(a=-1.0, b=1.0),
}


def _S(edge_idx, x):
    c = EDGE_COEFS[edge_idx]
    return c['a'] * np.sin(np.pi * x) + c['b'] * np.cos(np.pi * x)


def _dS(edge_idx, x):
    c = EDGE_COEFS[edge_idx]
    return np.pi * (c['a'] * np.cos(np.pi * x) - c['b'] * np.sin(np.pi * x))


L0, L1, L2, L3 = 2.0, 1.0, 1.0, 1.0
G2 = float(_dS(2, L2))
G3 = float(_dS(3, L3))


def get_exact_u(edge_idx, x_np, t_np, A2=0.0):
    return (np.asarray(t_np) ** ALPHA) * _S(edge_idx, x_np)


def get_exact_du(edge_idx, x_np, t_np, A2=0.0):
    return (np.asarray(t_np) ** ALPHA) * _dS(edge_idx, x_np)


def get_exact_dt_alpha(edge_idx, x_np):
    return GAMMA_A1 * _S(edge_idx, x_np)


class NonlinearParabolicPhysics:
    def __init__(self, alpha=ALPHA, V=V_COEFF, edge_idx=0):
        self.alpha = alpha
        self.V = V
        self.edge_idx = edge_idx
        self.a = EDGE_COEFS[edge_idx]['a']
        self.b = EDGE_COEFS[edge_idx]['b']
        self.gamma_a1 = float(gamma(alpha + 1.0))

    def get_ic(self, x):
        return torch.zeros_like(x)

    def F(self, x, t, u, u_x, u_xx, dt_alpha_u, f_target):
        V_tensor = self.V if torch.is_tensor(self.V) else torch.tensor(self.V, dtype=x.dtype, device=x.device)
        return dt_alpha_u - u_xx + V_tensor * u * u_x - f_target

    def get_f_target(self, x, t, L):
        S = self.a * torch.sin(math.pi * x) + self.b * torch.cos(math.pi * x)
        Sx = math.pi * (self.a * torch.cos(math.pi * x) - self.b * torch.sin(math.pi * x))
        t_a = t.clamp(min=0.0).pow(self.alpha)
        dt_alpha_analytic = self.gamma_a1 * S
        u_sp = t_a * S
        u_x_sp = t_a * Sx
        u_xx_sp = -(math.pi ** 2) * t_a * S
        return dt_alpha_analytic - u_xx_sp + self.V * u_sp * u_x_sp


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


PTS_PER_UNIT = 80
EPOCHS = 10000
N_STARTS = 3
PROBE_EPOCHS = 500
T_MAX = 1.0
N_T = 50


def build_solver(scheme, aux_pts=None):
    graph = BranchedTadpoleGraph(L0, L1, L2, L3)
    physics_list = [NonlinearParabolicPhysics(ALPHA, V_COEFF, i) for i in range(4)]
    solver = ParabolicPINNSolver(graph, physics_list)
    solver.set_architecture(hidden_layers=4, hidden_dim=128, use_fourier=True, fourier_dim=128, fourier_sigma=1.0)

    if scheme == 'L21_aux':
        solver.set_frac_scheme(scheme, aux_pts=aux_pts, aux_grading_factor=4)
    else:
        solver.set_frac_scheme(scheme)

    solver.set_mesh(pinn_pts=PTS_PER_UNIT, grading_factor=2, t_max=T_MAX, n_t=N_T)
    solver.set_singularity_capture(enabled=True, xi_init=ALPHA, xi_loss_adaptive=True)
    solver.set_lr(lr=0.0005, min_lr=1e-06, xi_lr_scale=0.1, xi_lr_scale_phase2=0.01)
    solver.set_constraints(
        constraint_mode='soft',
        bc_types={2: 'neumann', 3: 'neumann'},
        bc_values={
            2: lambda t: G2 * t.clamp(min=0.0).pow(ALPHA),
            3: lambda t: G3 * t.clamp(min=0.0).pow(ALPHA),
        })
    exact_funcs = [lambda x, t, i=i: get_exact_u(i, x, t) for i in range(4)]
    solver.set_validation(exact_funcs)
    return solver, exact_funcs


def run_one(run_name, scheme, aux_pts=None, seed=42):
    seed_everything(seed)
    clear_gpu()

    result = dict(run_name=run_name, scheme=scheme, aux_pts=aux_pts,
                  oom=False, error=None, errors=None)

    solver = None
    try:
        solver, exact_funcs = build_solver(scheme, aux_pts=aux_pts)
        solver.compile()

        strategy = 'dual'
        cfg_full = solver.calibrate_lambda_controllers(window_epochs=EPOCHS, seed=seed)
        lam_cfg = cfg_full.get(strategy)

        solver.train_multistart(epochs=EPOCHS, strategy=strategy, use_lbfgs=True,
                                n_starts=N_STARTS, seed_base=seed, probe_epochs=PROBE_EPOCHS,
                                lam_cfg=lam_cfg)

        errors = solver.report_l2(exact_funcs)
        result['errors'] = [float(e) for e in errors]

        out_dir = os.path.join('/kaggle/working', "parabolic_variants_checkpoints")
        os.makedirs(out_dir, exist_ok=True)

        ckpt = {
            'all_nets': {i: m['net'].state_dict() for i, m in solver.models.items()},
            'xi_raw': solver.xi_raw.detach().cpu(),
            'lambda_bc': getattr(solver, 'lambda_bc', None),
            'l2_errors': result['errors'],
            'scheme': scheme,
            'aux_pts': aux_pts,
        }

        safe_run_name = run_name.replace(" ", "_").lower()
        path = os.path.join(out_dir, f"{safe_run_name}.pt")
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
    configs = [
        ('50 Auxiliary Points', 'L21_aux', 50),
        ('25 Auxiliary Points', 'L21_aux', 25),
        ('Baseline', 'L21sigma', None)
    ]

    results = []
    for run_name, scheme, aux_pts in configs:
        results.append(run_one(run_name, scheme, aux_pts))