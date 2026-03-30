# Kaggle: start notebook cell with `%%writefile run_parabolic_inverse.py`
import sys
sys.path.insert(0, "/kaggle/working")

import numpy as np
import torch
from PINN_Solver import run_parabolic_inverse

# -------- experiment controls --------
ALPHA = 0.5
TRUE_NU = 1.0
N_EDGES = 3
L = 1.0
PHI_COEFFS = [1.0, 1.0, -2.0]

SCHEME = "L1"
R_GRADING = 2.0
EPOCHS = 12000
SEED = 123
N_POINTS_PER_EDGE = 80
NOISE_STD = 0.01
DATA_WEIGHT = 20.0
PARAM_BOUNDS = {"nu": (0.1, 2.0)}
ALPHA_BOUNDS = (0.2, 0.95)


class StarGraph:
    nodes = list(range(N_EDGES + 1))
    edges = [(0, i + 1, L) for i in range(N_EDGES)]


class EdgePhysics:
    def __init__(self, alpha, coeff):
        self.alpha = alpha
        self.nu = 1.0
        self._c = coeff

    def F(self, x, t, u, u_x, u_xx, dt_alpha_u, f_target):
        return dt_alpha_u - self.nu * u_xx - f_target

    def get_f_target(self, x, t, L_):
        a = self.alpha if torch.is_tensor(self.alpha) else torch.tensor(self.alpha, dtype=t.dtype, device=t.device)
        g1 = torch.exp(torch.lgamma(a + 1.0))
        t_a = torch.pow(t.clamp(min=1e-10), a)
        return (g1 + self.nu * (np.pi ** 2) * t_a) * self._c * torch.sin(np.pi * x)

    def get_ic(self, x):
        return torch.zeros_like(x)


def exact_u(edge_idx, x_np, t_np, alpha=ALPHA):
    return (t_np ** alpha) * PHI_COEFFS[edge_idx] * np.sin(np.pi * x_np.flatten())


if __name__ == "__main__":
    print(f"Inverse parabolic: epochs={EPOCHS}, points/edge={N_POINTS_PER_EDGE}, noise={NOISE_STD}")
    physics = [EdgePhysics(ALPHA, c) for c in PHI_COEFFS]
    for p in physics:
        p.nu = float(TRUE_NU)

    solver, est = run_parabolic_inverse(
        StarGraph(),
        physics,
        exact_u,
        scheme=SCHEME,
        r_grading=R_GRADING,
        epochs=EPOCHS,
        n_points_per_edge=N_POINTS_PER_EDGE,
        noise_std=NOISE_STD,
        data_weight=DATA_WEIGHT,
        seed=SEED,
        parameter_names=("nu",),
        include_alpha=True,
        param_bounds=PARAM_BOUNDS,
        alpha_bounds=ALPHA_BOUNDS,
    )

    print("\nEstimated (per edge):", est)
    print("True nu / alpha:", TRUE_NU, ALPHA)

    exact_fns = [lambda x, t, i=i: exact_u(i, x, t) for i in range(N_EDGES)]
    errs = solver.report_l2(exact_fns)
    if errs:
        print(f"Mean L2: {np.mean(errs):.4e}")
