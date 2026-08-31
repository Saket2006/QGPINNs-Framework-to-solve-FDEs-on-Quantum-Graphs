import os
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Engine'))
from QGPINNs_Engine import (
    ParabolicPINNSolver, seed_everything, device, NET_DTYPE, DEFAULT_PARABOLIC_ARCH
)

SEED = 42
ALPHA = 0.75
V0 = 0
A_AMP = 1.0
LAM = 0.20
ETA = 1.0
T_MAX = 5.0

C_LOW, C_HIGH = 0.6, 1.4

HIDDEN_LAYERS = 4
HIDDEN_DIM = 128
FOURIER_DIM = 128
FOURIER_SIGMA = 4.0
PINN_PTS = 60
N_T = 80
GRADING_FACTOR = 2.0
FRAC_SCHEME = 'L21sigma'
EPOCHS = 12000
N_STARTS = 1
PROBE_EPOCHS = 0

OUT_PATH = 'telegraph_model_nd(0.75).pt'

seed_everything(SEED)

class MetricGraph:
    def __init__(self, edges):
        self.edges = edges
        self.nodes = sorted(set(n for e in edges for n in e[:2]))

_edges_raw = [
    (0, 1, 1.0),
    (0, 4, 1.2),
    (1, 2, 1.0),
    (1, 3, 1.1),
    (1, 4, 0.9),
    (2, 3, 1.0),
    (3, 4, 0.8),
    (3, 6, 1.3),
    (3, 8, 1.4),
    (4, 5, 1.1),
    (5, 10, 1.0),
    (5, 11, 1.2),
    (5, 12, 1.1),
    (6, 7, 0.7),
    (6, 8, 0.9),
    (8, 9, 1.0),
    (8, 13, 1.3),
    (9, 10, 0.8),
    (11, 12, 0.9),
    (12, 13, 1.0),
]

edges = []
for u, v, L in _edges_raw:
    if v == V0 and u != V0:
        u, v = v, u
    edges.append((u, v, float(L)))

graph = MetricGraph(edges)
pulse_edge_idx = {i for i, (u, v, L) in enumerate(edges) if u == V0}
assert pulse_edge_idx == {0, 1}
assert len(edges) == 20 and len(graph.nodes) == 14

rng = np.random.default_rng(SEED)
c_list = rng.uniform(C_LOW, C_HIGH, size=len(edges))

class TelegraphEdgePhysics:
    def __init__(self, alpha, eta, c, t_max, pulse=False, A=1.0, lam=0.15):
        self.alpha = alpha
        self.eta = eta
        self.c = c
        self.t_max = t_max
        self.pulse = pulse
        self.A = A
        self.lam = lam

        self.Lam_damp = eta * (t_max ** (2.0 - alpha))
        self.Lam_wave = (c ** 2) * (t_max ** 2)

    def get_ic(self, x):
        if self.pulse:
            return self.A * torch.exp(-(x / self.lam) ** 2)
        return torch.zeros_like(x)

    def get_f_target(self, x, t, L):
        return torch.zeros_like(x)

    def F(self, X, T, u, u_x, u_xx, dt_alpha_u, f_target):
        u_t = torch.autograd.grad(u, T, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, T, grad_outputs=torch.ones_like(u_t),
                                   create_graph=True, retain_graph=True)[0]

        res = u_tt + self.Lam_damp * dt_alpha_u - self.Lam_wave * u_xx - f_target
        T_max_sq = self.t_max ** 2
        return res / T_max_sq

    def get_flux(self, u, u_x):
        return (self.c ** 2) * u_x

physics_list = [
    TelegraphEdgePhysics(alpha=ALPHA, eta=ETA, c=float(c_list[i]), t_max=T_MAX,
                          pulse=(i in pulse_edge_idx), A=A_AMP, lam=LAM)
    for i in range(len(edges))
]

class TelegraphPINNSolver(ParabolicPINNSolver):
    def compute_losses(self, epoch=0):
        lp, ln = super().compute_losses(epoch)
        lv = torch.zeros((), dtype=NET_DTYPE, device=device)
        ic_pulse_loss = torch.zeros((), dtype=NET_DTYPE, device=device)

        for i, m in self.models.items():
            X0 = m['x_grid'].clone().detach().requires_grad_(True)
            T0 = torch.zeros_like(X0).requires_grad_(True)
            u0 = self.predict(i, X0, T0)
            u_t0 = torch.autograd.grad(u0, T0, grad_outputs=torch.ones_like(u0),
                                        create_graph=True)[0]

            u_t0_phys = u_t0 / m['physics'].t_max
            lv = lv + torch.mean(u_t0_phys ** 2)

            if getattr(m['physics'], 'pulse', False):
                target_ic = m['physics'].get_ic(X0)
                ic_pulse_loss = ic_pulse_loss + 10.0 * torch.mean((u0 - target_ic) ** 2)

        return lp, ln + lv + ic_pulse_loss

arch = dict(hidden_layers=HIDDEN_LAYERS, hidden_dim=HIDDEN_DIM, use_fourier=True,
            fourier_dim=FOURIER_DIM, fourier_sigma=FOURIER_SIGMA, fourier_sampling='sobol')

solver = TelegraphPINNSolver(graph, physics_list)
solver.set_architecture(**arch)
solver.set_mesh(mesh_type='power_law', pinn_pts=PINN_PTS, n_t=N_T,
                grading_factor=GRADING_FACTOR, t_max=1.0)
solver.set_constraints('soft')
solver.set_frac_scheme(FRAC_SCHEME)
solver.set_singularity_capture(enabled=False)
solver.set_lr(lr=5e-4, min_lr=1e-5)
solver.compile()

solver.train_multistart(epochs=EPOCHS, strategy='dual', use_lbfgs=True,
                         n_starts=N_STARTS, seed_base=SEED, probe_epochs=PROBE_EPOCHS)

checkpoint = dict(
    edges=edges,
    c_list=c_list.tolist(),
    eta=ETA,
    alpha=ALPHA,
    v0=V0,
    pulse_edge_idx=sorted(pulse_edge_idx),
    A=A_AMP,
    lam=LAM,
    t_max_phys=T_MAX,
    t_max_mesh=1.0,
    arch=arch,
    mesh=dict(mesh_type='power_law', pinn_pts=PINN_PTS, n_t=N_T,
              grading_factor=GRADING_FACTOR, t_max=1.0),
    frac_scheme=FRAC_SCHEME,
    seed=SEED,
    singularity_capture=bool(getattr(solver, 'use_singularity_capture', False)),
    xi_raw=(solver.xi_raw.detach().cpu() if getattr(solver, 'use_singularity_capture', False) else None),
    net_states={i: m['net'].state_dict() for i, m in solver.models.items()},
)
torch.save(checkpoint, OUT_PATH)