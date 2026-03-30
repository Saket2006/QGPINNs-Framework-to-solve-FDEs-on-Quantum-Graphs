# Kaggle: start notebook cell with `%%writefile run_elliptic_inverse.py`
import sys
sys.path.insert(0, "/kaggle/working")

import numpy as np
from PINN_Solver import run_elliptic_inverse

# -------- experiment controls --------
TRUE_ALPHA = 1.6
TRUE_BETA = 0.7
TRUE_REACTION = 1.0
N_EDGES = 3
L = 1.0
PHI_COEFFS = [1.0, 1.0, -2.0]

EPOCHS = 6000
SEED = 123
N_POINTS_PER_EDGE = 80
NOISE_STD = 0.01
DATA_WEIGHT = 20.0
PARAM_BOUNDS = {"beta": (0.4, 1.0), "reaction": (0.5, 1.5)}
INCLUDE_ALPHA = False
ALPHA_BOUNDS = (1.05, 1.95)
PTS_PER_UNIT = 220
GRADING_FACTOR = 1.5


class StarGraph:
    nodes = list(range(N_EDGES + 1))
    edges = [(0, i + 1, L) for i in range(N_EDGES)]


class EllipticPhysics:
    def __init__(self, alpha, beta, reaction, edge_coeff=1.0):
        self.alpha = alpha
        self.beta = beta
        self.reaction = reaction
        self.edge_coeff = edge_coeff

    def F(self, x, u, du, d_beta_u, f_target):
        # FIX: Changed `- f_target` to `+ f_target` to match the solver's residual formula
        return d_beta_u + self.reaction * u + f_target

    def get_f_target(self, x_np, L_, Da_np, Db_np):
        x = x_np.reshape(-1)
        u = self.edge_coeff * np.sin(np.pi * x)
        du = self.edge_coeff * np.pi * np.cos(np.pi * x)
        d_beta_u = Db_np @ u
        d_alpha_u = Da_np @ du
        return d_alpha_u - d_beta_u - self.reaction * u


def _exact_edge(i):
    return lambda x: PHI_COEFFS[i] * np.sin(np.pi * x)


if __name__ == "__main__":
    print(f"Inverse elliptic: epochs={EPOCHS}, points/edge={N_POINTS_PER_EDGE}, noise={NOISE_STD}")
    physics_list = [
        EllipticPhysics(TRUE_ALPHA, TRUE_BETA, TRUE_REACTION, edge_coeff=PHI_COEFFS[i])
        for i in range(N_EDGES)
    ]
    exact_funcs = [_exact_edge(i) for i in range(N_EDGES)]

    solver, est = run_elliptic_inverse(
        StarGraph(),
        physics_list,
        exact_funcs,
        parameter_names=("beta", "reaction"),
        include_alpha=INCLUDE_ALPHA,
        param_bounds=PARAM_BOUNDS,
        alpha_bounds=ALPHA_BOUNDS,
        data_weight=DATA_WEIGHT,
        n_points_per_edge=N_POINTS_PER_EDGE,
        noise_std=NOISE_STD,
        seed=SEED,
        epochs=EPOCHS,
        pts_per_unit=PTS_PER_UNIT,
        grading_factor=GRADING_FACTOR,
    )

    print("\nEstimated:", est)
    print("True beta / reaction:", TRUE_BETA, TRUE_REACTION)

    val_list = [lambda x, L, i=i: exact_funcs[i](x) for i in range(N_EDGES)]
    solver.report_l2(val_list)
