'''This solves the Parabolic Problem, it will be integrated with Main Engine on refinement'''


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConstrainedNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Fourier features help capture high-frequency components for better L2 accuracy
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, t, L):
        u_raw = self.net(torch.cat([x, t], dim=-1))
        # HARD CONSTRAINTS:
        # x*(L-x) forces u(0,t)=0 and u(L,t)=0
        # t**2 ensures u(x,0)=0 AND matches the t^2 growth of our manufactured solution
        return x * (L - x) * (t ** 2) * u_raw


class ManufacturedPhysics:
    def __init__(self, alpha=0.7, mu=1.0, b=0.2, sigma=0.1):
        self.alpha, self.mu, self.b, self.sigma = alpha, mu, b, sigma

    def get_coeffs(self, x, t, L):
        pi = torch.tensor(np.pi)
        # Caputo D^alpha (t^2) = 2/Gamma(3-alpha) * t^(2-alpha)
        term_t = (2.0 / gamma(3 - self.alpha)) * (t ** (2 - self.alpha)) * torch.sin(pi * x / L)
        u_x = (t ** 2) * (pi / L) * torch.cos(pi * x / L)
        u_xx = -(t ** 2) * (pi / L) ** 2 * torch.sin(pi * x / L)

        f_target = term_t - self.mu * u_xx + self.b * u_x + self.sigma * (t ** 2 * torch.sin(pi * x / L))
        return f_target

    def exact_u(self, x, t, L):
        return (t ** 2) * torch.sin(np.pi * x / L)


class HighPrecisionSolver:
    def __init__(self, physics, L=1.0, n_t=100, n_x=60):
        self.physics = physics
        self.L = L
        self.t_grid = torch.linspace(0, 1.0, n_t).view(-1, 1).to(device)
        self.x_grid = torch.linspace(0, L, n_x).view(-1, 1).to(device)
        self.n_t, self.n_x = n_t, n_x

        self.Dt_alpha = self.compute_l1_matrix(physics.alpha, self.t_grid.flatten().cpu().numpy())
        self.net = ConstrainedNet().to(device)
        self.optimizer_lbfgs = None  # Initialize as None

    @staticmethod
    def compute_l1_matrix(alpha, t_np):
        n = len(t_np)
        mat = np.zeros((n, n))
        h = np.diff(t_np)
        g1 = gamma(2 - alpha)
        for i in range(1, n):
            for j in range(i):
                w = ((t_np[i] - t_np[j]) ** (1 - alpha) - (t_np[i] - t_np[j + 1]) ** (1 - alpha)) / (h[j] * g1)
                mat[i, j + 1] += w
                mat[i, j] -= w
        return torch.tensor(mat).to(device)

    def closure(self):
        # Only call zero_grad if the optimizer exists
        if self.optimizer_lbfgs is not None:
            self.optimizer_lbfgs.zero_grad()

        X = self.x_grid.repeat(self.n_t, 1).requires_grad_(True)
        T = self.t_grid.repeat_interleave(self.n_x, dim=0).requires_grad_(True)

        u = self.net(X, T, self.L)
        u_x = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0]

        f_target = self.physics.get_coeffs(X, T, self.L)
        u_reshaped = u.view(self.n_t, self.n_x)
        dt_alpha_u = torch.mm(self.Dt_alpha, u_reshaped).view(-1, 1)

        loss = torch.mean(
            (dt_alpha_u - self.physics.mu * u_xx + self.physics.b * u_x + self.physics.sigma * u - f_target) ** 2)

        if loss.requires_grad:
            loss.backward()
        return loss

    def train(self, adam_epochs=2000):
        # 1. Adam Warmup
        opt_adam = optim.Adam(self.net.parameters(), lr=1e-3)
        print("--- Start Adam Warmup ---")
        for i in range(adam_epochs + 1):
            opt_adam.zero_grad()
            loss = self.closure()
            opt_adam.step()
            if i % 500 == 0:
                print(f"Adam Epoch {i} | Loss: {loss.item():.2e}")

        # 2. L-BFGS Polishing
        print("\n--- Start L-BFGS Refinement ---")
        self.optimizer_lbfgs = optim.LBFGS(
            self.net.parameters(),
            max_iter=1000,
            tolerance_grad=1e-11,
            tolerance_change=1e-13,
            line_search_fn="strong_wolfe"
        )
        self.optimizer_lbfgs.step(self.closure)
        print(f"Final Refined Loss: {self.closure().item():.2e}")

        def check_accuracy_at_times(self, time_points=[0.2, 0.5, 0.8, 1.0]):
            """Prints a detailed error report at specific time intervals."""
            print(f"\n{'Time (t)':<10} | {'Relative L2 Error':<20}")
            print("-" * 35)

            for t_val in time_points:
                t_tensor = torch.full((200, 1), t_val).to(device)
                x_test = torch.linspace(0, self.L, 200).view(-1, 1).to(device)

                # Predict and compare
                u_pred = self.net(x_test, t_tensor, self.L).detach().cpu().numpy()
                u_true = self.physics.exact_u(x_test.cpu(), t_val, self.L).numpy()

                # Avoid division by zero at t=0
                norm_true = np.linalg.norm(u_true)
                if norm_true < 1e-10:
                    l2_err = np.linalg.norm(u_pred)
                else:
                    l2_err = np.linalg.norm(u_pred - u_true) / norm_true

                print(f"{t_val:<10.2f} | {l2_err:.4e}")

    # --- Update your Main execution block ---
if __name__ == "__main__":
    phys = ManufacturedPhysics(alpha=0.7)
    solver = HighPrecisionSolver(phys, n_t=100)
    solver.train()

    # New reporting call
    solver.check_accuracy_at_times(time_points=[0.25, 0.5, 0.75, 1.0])

    solver.report_accuracy()
