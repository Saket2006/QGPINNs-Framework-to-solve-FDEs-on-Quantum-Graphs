import sys
import os

if '/' not in sys.path:
    sys.path.append('/')
from pinn_parabolic_engine import ParabolicPINNSolver
import numpy as np
import torch
from scipy.special import gamma


class Graph:
    def __init__(self):
        self.edges = [(0, 1, 1.0)]
        self.nodes = [0, 1]


class BurgersPhysics:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def F(self, x, t, u, u_x, u_xx, dt_alpha_u, f_target):
        return dt_alpha_u + f_target + (u * u_x) - 0.1 * u_xx

    def get_f_target(self, x, t, L):
        g_val = gamma(3 - self.alpha)
        term1 = -(2 * torch.exp(x) * torch.pow(t, 2 - self.alpha)) / g_val
        term2 = -torch.exp(2 * x) * torch.pow(t, 4)
        term3 = 0.1 * torch.exp(x) * torch.pow(t, 2)
        return term1 + term2 + term3

    def get_ic(self, x):
        return torch.zeros_like(x)


def exact_u(x, t):
    return np.exp(x) * t ** 2


if __name__ == "__main__":

    SCHEME = "L21sigma"

    my_bc_types = {0: "dirichlet", 1: "dirichlet"}
    my_bc_values = {
        0: lambda t: t ** 2,
        1: lambda t: np.e * t ** 2,
    }

    solver = ParabolicPINNSolver(graph=Graph(), physics=BurgersPhysics(alpha=0.5))

    solver.set_frac_scheme(scheme=SCHEME, sigma=None)

    solver.set_architecture(
        hidden_layers=4,
        hidden_dim=60,
        use_fourier=True,
        fourier_dim=64,
        fourier_sigma=1.0,
        #fourier_sampling="sobol"
    ).set_mesh(
        mesh_type="power_law",
        pinn_pts=100,
        anchor_pts=0,
        grading_factor=1,
        n_t=100
    ).set_constraints(
        constraint_mode="hard",
        bc_types=my_bc_types,
        bc_values=my_bc_values
    )

    solver.set_ntk_balancing(enabled=True)
    solver.ntk_every = 200
    solver.set_rad_resampling(enabled=False)
    solver.train(epochs=10000, strategy="dual", use_lbfgs=True)
    errors = solver.report_l2(exact_u, eval_times=[1.0])
    solver.plot_results(exact_u, t_vals=[0.5, 0.75, 1.0])
