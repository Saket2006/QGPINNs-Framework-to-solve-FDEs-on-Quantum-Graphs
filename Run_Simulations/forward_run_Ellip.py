import sys

sys.path.insert(0, "/kaggle/working")

import numpy as np
from PINN_Solver import run_elliptic_forward

# -------- experiment controls --------
ALPHA = 1.6
BETA = 0.7
REACTION = 1.0
N_EDGES = 3
L = 1.0

# 1. ADD COEFFICIENTS TO SATISFY KIRCHHOFF'S LAW
PHI_COEFFS = [1.0, 1.0, -2.0]

EPOCHS = 8000
N_STARTS = 3
PROBE_EPOCHS = 600
SEED_BASE = 0
PTS_PER_UNIT = 250
GRADING_FACTOR = 1.5


class StarGraph:
    nodes = list(range(N_EDGES + 1))
    edges = [(0, i + 1, L) for i in range(N_EDGES)]


class EllipticPhysics:
    def __init__(self, alpha, beta, reaction, coeff):
        self.alpha = alpha
        self.beta = beta
        self.reaction = reaction
        self._c = coeff  # 2. ADD EDGE COEFFICIENT

    def F(self, x, u, du, d_beta_u, f_target):
        # 3. FIX THE SIGN ERROR: Change `- f_target` to `+ f_target`
        return d_beta_u + self.reaction * u + f_target

    def get_f_target(self, x_np, L_, Da_np, Db_np):
        x = x_np.reshape(-1)
        # 4. APPLY COEFFICIENT TO THE EXACT SOLUTION
        u = self._c * np.sin(np.pi * x)
        du = self._c * np.pi * np.cos(np.pi * x)

        d_beta_u = Db_np @ u
        d_alpha_u = Da_np @ du
        return d_alpha_u - d_beta_u - self.reaction * u


# 5. UPDATE EXACT FUNCTION TO HANDLE EDGE INDEX
def exact_u(edge_idx, x_np, L_):
    return PHI_COEFFS[edge_idx] * np.sin(np.pi * x_np)


if __name__ == "__main__":
    print(f"Forward elliptic: epochs={EPOCHS}, starts={N_STARTS}, probe={PROBE_EPOCHS}")

    # 6. INSTANTIATE PHYSICS PER EDGE
    physics_list = [EllipticPhysics(ALPHA, BETA, REACTION, c) for c in PHI_COEFFS]
    exact_funcs = [lambda x, L_, i=i: exact_u(i, x, L_) for i in range(N_EDGES)]

    run_elliptic_forward(
        StarGraph(),
        physics_list,  # Pass the list
        exact_funcs,  # Pass the list of validation functions
        epochs=EPOCHS,
        n_starts=N_STARTS,
        probe_epochs=PROBE_EPOCHS,
        seed_base=SEED_BASE,
        pts_per_unit=PTS_PER_UNIT,
        grading_factor=GRADING_FACTOR,
    )
