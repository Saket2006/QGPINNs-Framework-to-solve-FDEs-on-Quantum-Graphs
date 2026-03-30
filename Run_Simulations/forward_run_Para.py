# Kaggle: start notebook cell with `%%writefile run_parabolic_forward.py`
import sys
sys.path.insert(0, "/kaggle/working")

import numpy as np
import torch
from PINN_Solver import run_parabolic_forward_sweep, parabolic_sweep_error_plot

# -------- experiment controls --------
ALPHA = 0.5
N_EDGES = 3
L = 1.0
PHI_COEFFS = [1.0, 1.0, -2.0]
R_VALUES = [1.0, 2.0, 4.0]
SCHEME = "L21sigma"

EPOCHS = 12000
N_STARTS = 3
PROBE_EPOCHS = 800
SEED_BASE = 0
PINN_PTS = 100
N_T = 100
T_MAX = 1.0


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
    print(f"Forward parabolic: epochs={EPOCHS}, starts={N_STARTS}, probe={PROBE_EPOCHS}, scheme={SCHEME}")
    physics = [EdgePhysics(ALPHA, c) for c in PHI_COEFFS]
    results = run_parabolic_forward_sweep(
        StarGraph(),
        physics,
        exact_u,
        r_values=tuple(R_VALUES),
        scheme=SCHEME,
        epochs=EPOCHS,
        n_starts=N_STARTS,
        probe_epochs=PROBE_EPOCHS,
        seed_base=SEED_BASE,
        pinn_pts=PINN_PTS,
        n_t=N_T,
        t_max=T_MAX,
    )

    print("\n r    mean_L2")
    print("-" * 20)
    for r in R_VALUES:
        if results.get(r):
            print(f" {r:>3.1f}  {np.mean(results[r]):.4e}")

    parabolic_sweep_error_plot(results, R_VALUES, ALPHA, SCHEME, N_EDGES)
