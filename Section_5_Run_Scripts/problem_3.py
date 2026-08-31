import sys
import os
import random
import math
import torch
import numpy as np

sys.path.insert(0, '/kaggle/working')
from QGPINNs_Engine import ParabolicPINNSolver, DEFAULT_PARABOLIC_ARCH, device, NET_DTYPE

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

class DrainageNetworkGraph:
    nodes = (0, 1, 2, 3, 4, 5)
    edges = [
        (1, 3, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
        (0, 4, 1.0),
        (4, 5, 1.0),
    ]

class NonlinearBTEPhysics:
    def __init__(self, alpha, L_phys=20.0, T_phys=10.0, h0=5.0, D_phys=25.0):
        self.alpha = alpha
        self.L = L_phys
        self.T = T_phys
        self.h0 = h0
        self.D = D_phys
        self.g = 9.81

        self.term1 = (2.0 / 3.0) * math.sqrt(self.g)
        self.h0_term = self.h0 ** 1.5

        self.Lam_adv = (self.T ** alpha) / self.L
        self.Lam_diff = (self.T ** alpha) * self.D / (self.L ** 2)

    def V(self, u):
        total_depth = torch.clamp(u + self.h0, min=1e-5)
        return self.term1 * (total_depth ** 1.5 - self.h0_term)

    def get_ic(self, x_hat):
        if torch.is_tensor(x_hat):
            return torch.zeros_like(x_hat)
        return np.zeros_like(x_hat)

    def get_f_target(self, x_hat, t_hat, L=None):
        if torch.is_tensor(x_hat):
            return torch.zeros_like(x_hat)
        return np.zeros_like(x_hat)

    def F(self, x_hat, t_hat, u, u_x, u_xx, d_alpha_u, f_target):
        V_dyn = self.V(u)
        total_depth = torch.clamp(u + self.h0, min=1e-5)
        V_prime = math.sqrt(self.g) * (total_depth ** 0.5)

        conservative_adv = (V_dyn * u_x) + (u * V_prime * u_x)
        diffusion = u_xx

        return d_alpha_u + self.Lam_adv * conservative_adv - self.Lam_diff * diffusion - f_target

    def get_flux(self, u, u_x):
        V_dyn = self.V(u)
        return self.Lam_adv * (V_dyn * u) - self.Lam_diff * u_x

if __name__ == "__main__":
    seed_everything(42)

    EPOCHS = 15000
    ALPHA_TARGET = 0.85
    L_PHYS = 20.0
    T_PHYS = 10.0
    PINN_PTS = 200
    N_T = 200

    graph = DrainageNetworkGraph()
    physics_list = [
        NonlinearBTEPhysics(alpha=ALPHA_TARGET, L_phys=L_PHYS, T_phys=T_PHYS, h0=5.0, D_phys=25.0)
        for _ in graph.edges
    ]

    solver = ParabolicPINNSolver(graph, physics_list)
    solver.set_frac_scheme('L21sigma')

    solver.set_architecture(hidden_layers=4, hidden_dim=128, use_fourier=True, fourier_dim=128)

    solver.set_mesh(mesh_type='power_law', pinn_pts=PINN_PTS, n_t=N_T,
                     grading_factor=2.0, t_max=1.0)

    def inlet_surge(t_hat):
        surge = 2.0 * torch.sin(math.pi * t_hat) ** 2
        return surge.to(NET_DTYPE).to(device)

    bc_types_map = {
        0: 'dirichlet',
        1: 'dirichlet',
        2: 'dirichlet',
        5: 'neumann'
    }

    bc_values_map = {
        0: lambda t: inlet_surge(t),
        1: lambda t: inlet_surge(t),
        2: lambda t: inlet_surge(t),
        5: lambda t: torch.zeros_like(t)
    }

    solver.set_constraints('soft', bc_types=bc_types_map, bc_values=bc_values_map)

    solver.set_dense_constraint_enforcement(
        enabled=True,
        factor=4,
        only_custom_flux=True,
    )

    solver.set_singularity_capture(enabled=True, xi_init=ALPHA_TARGET, xi_loss_adaptive=True)

    solver.set_lr(lr=0.0008, min_lr=1e-05)
    solver.lambda_junc = 1.0
    solver.lambda_soft_clip = 2500.0
    solver.BDMM_Clip = 10000.0

    solver.compile()

    solver.train(epochs=EPOCHS, strategy='dual', use_lbfgs=True)

    os.makedirs('weights', exist_ok=True)
    weight_dict = {}
    for edge_idx, m in solver.models.items():
        weight_dict[f'edge_{edge_idx}_net'] = m['net'].state_dict()
        weight_dict[f'edge_{edge_idx}_nodes'] = m['nodes']

    if getattr(solver, 'use_singularity_capture', False):
        weight_dict['xi_raw'] = solver.xi_raw.data

    save_path = f'weights/pinn_nonlinear_nd_a{int(ALPHA_TARGET * 100)}.pt'
    torch.save(weight_dict, save_path)