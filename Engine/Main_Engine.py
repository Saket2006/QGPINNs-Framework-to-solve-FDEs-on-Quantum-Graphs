'''This Engine solves the elliptic problem, I will update this engine to solve the 
                Parabolic Problem on refinement'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import networkx as nx
import copy

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#The model size is pre-determined for now, will make it user defined.
class PINN_Net(nn.Module):
    def __init__(self, hidden_dim=64, fourier_dim=40):
        super().__init__()
        B = torch.randn(1, fourier_dim) * 25.0
        self.register_buffer('B', B)
        self.layers = nn.Sequential(
            nn.Linear(fourier_dim * 2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        proj = x @ self.B
        xf = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.layers(xf)

class GraphPINNSolver:
    def __init__(self, graph, physics, strategy="bdmm",
                 mesh_type="graded", fixed_lambda=1.0,
                 pts_per_unit=250, grading_factor=1.5,
                 bc_types=None, bc_values=None):
        self.graph = graph
        self.physics = physics
        self.strategy = strategy
        self.mesh_type = mesh_type
        self.lambda_node = fixed_lambda
        self.pts_per_unit = pts_per_unit
        self.grading_factor = grading_factor
        self.bc_types = bc_types if bc_types else {}
        self.bc_values = bc_values if bc_values else {}

        self.mu = 1.0
        self.eta = 1e-2

        self.models = {}
        self._init_edges()

    @staticmethod
    def compute_l1_matrix(alpha, x_np):
        n = len(x_np)
        mat = np.zeros((n, n))
        if alpha <= 0: return torch.tensor(mat).to(device)
        h = np.diff(x_np)
        g1 = gamma(2 - alpha)
        for i in range(1, n):
            for j in range(i):
                w = ((x_np[i] - x_np[j]) ** (1 - alpha) - (x_np[i] - x_np[j + 1]) ** (1 - alpha)) / (h[j] * g1)
                mat[i, j + 1] += w
                mat[i, j] -= w
        return torch.tensor(mat).to(device)

    def _generate_mesh(self, L, n_pts):
        if self.mesh_type == "graded":
            xi = np.linspace(-1, 1, n_pts)
            return (np.sign(xi) * np.abs(xi) ** self.grading_factor + 1) * 0.5 * L

        else:
            return np.linspace(0, L, n_pts)

    def _init_edges(self):
        for i, (u, v, L) in enumerate(self.graph.edges):
            n_pts = max(int(self.pts_per_unit * L), 2)
            x_np = self._generate_mesh(L, n_pts)

            self.models[i] = {
                'net': PINN_Net().to(device),
                'L': L, 'n': n_pts, 'nodes': (u, v),
                'x_grid': torch.tensor(x_np).view(-1, 1).to(device),
                'Da': self.compute_l1_matrix(self.physics.alpha - 1.0, x_np),
                'Db': self.compute_l1_matrix(self.physics.beta, x_np),
                'f_target': None
            }
            f_target = self.physics.get_f_target(x_np, L, self.models[i]['Da'].cpu().numpy(),
                                                 self.models[i]['Db'].cpu().numpy())
            self.models[i]['f_target'] = torch.tensor(f_target).view(-1, 1).to(device)

    def resample_mesh(self):
        for i, m in self.models.items():
            new_x = np.sort(np.random.uniform(0, m['L'], m['n']))
            new_x[0], new_x[-1] = 0.0, m['L']
            m['x_grid'] = torch.tensor(new_x).view(-1, 1).to(device)
            m['Da'] = self.compute_l1_matrix(self.physics.alpha - 1.0, new_x)
            m['Db'] = self.compute_l1_matrix(self.physics.beta, new_x)
            f_new = self.physics.get_f_target(new_x, m['L'], m['Da'].cpu().numpy(), m['Db'].cpu().numpy())
            m['f_target'] = torch.tensor(f_new).view(-1, 1).to(device)

    def compute_losses(self):
        lp, ln = 0, 0
        for i, m in self.models.items():
            xt = m['x_grid'].clone().detach().requires_grad_(True)
            u = m['net'](xt)
            du = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True, allow_unused=True)[0]
            if du is None: du = torch.zeros_like(u)

            d_beta_u = torch.mm(m['Db'], u)
            d_alpha_u = torch.mm(m['Da'], du)
            res = d_alpha_u - self.physics.F(xt, u, du, d_beta_u, m['f_target'])
            lp += torch.mean(res ** 2)

        for node in self.graph.nodes:
            inc = []
            for i, m in self.models.items():
                if node == m['nodes'][0]: inc.append((i, 0.0, 1.0))
                if node == m['nodes'][1]: inc.append((i, m['L'], -1.0))
            if not inc: continue

            if len(inc) == 1:
                idx, xv, s = inc[0]
                xi = torch.tensor([[xv]], device=device, requires_grad=True)
                u_pred = self.models[idx]['net'](xi)
                if self.bc_types.get(node, "dirichlet") == "dirichlet":
                    ln += torch.mean((u_pred - self.bc_values.get(node, 0.0)) ** 2)
                else:
                    du_p = torch.autograd.grad(u_pred, xi, create_graph=True, allow_unused=True)[0]
                    ln += torch.mean(
                        ((du_p if du_p is not None else torch.zeros_like(xi)) - self.bc_values.get(node, 1.0)) ** 2)
            else:
                preds = [self.models[idx]['net'](torch.tensor([[xv]], device=device)) for idx, xv, _ in inc]
                for j in range(1, len(preds)): ln += torch.mean((preds[0] - preds[j]) ** 2)

                current_flux = 0
                for idx, xv, s in inc:
                    xi = torch.tensor([[xv]], device=device, requires_grad=True)
                    u_val = self.models[idx]['net'](xi)
                    grad = torch.autograd.grad(u_val, xi, torch.ones_like(u_val), create_graph=True, allow_unused=True)[
                        0]
                    if grad is not None: current_flux += grad * s
                ln += torch.mean(current_flux ** 2)
        return lp, ln

    def train(self, epochs=2000, use_lbfgs=False, adaptive_every=None, settling_epochs=100):
        params = [p for m in self.models.values() for p in m['net'].parameters()]
        optimizer = optim.Adam(params, lr=5e-4)

        best_loss = float('inf')
        best_models_state = None

        print(f"Strategy: {self.strategy}")
        print(f"--- Stage 1: Adam (Dynamic Weights) for {epochs} epochs ---")

        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            lp, ln = self.compute_losses()


            if self.strategy == "gradient_ratio" and epoch % adaptive_every == 0:
                g_p = torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)
                g_n = torch.autograd.grad(ln, params, retain_graph=True, allow_unused=True)
                std_p = torch.sqrt(torch.var(torch.cat([g.view(-1) for g in g_p if g is not None])) + 1e-9)
                std_n = torch.sqrt(torch.var(torch.cat([g.view(-1) for g in g_n if g is not None])) + 1e-9)
                self.lambda_node = 0.9 * self.lambda_node + 0.1 * (std_p / std_n).item()

            current_weight = self.mu if self.strategy == "bdmm" else self.lambda_node
            loss = lp + current_weight * ln


            if loss.item() < best_loss:
                best_loss = loss.item()
                best_models_state = {i: copy.deepcopy(m['net'].state_dict()) for i, m in self.models.items()}

            loss.backward()
            optimizer.step()

            if self.strategy == "bdmm":
                self.mu += self.eta * ln.item()

            if epoch % 500 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.2e} | PDE: {lp.item():.2e} | Node: {ln.item():.2e}")


        if best_models_state:
            for i, m in self.models.items():
                m['net'].load_state_dict(best_models_state[i])


        fixed_weight = self.mu if self.strategy == "bdmm" else self.lambda_node
        print(f"\n--- Stage 2: Settling (Adam with Fixed Weight: {fixed_weight:.2f}) ---")

        for _ in range(settling_epochs):
            optimizer.zero_grad()
            lp, ln = self.compute_losses()
            loss = lp + fixed_weight * ln
            loss.backward()
            optimizer.step()


        if use_lbfgs:
            print("--- Stage 3: L-BFGS Refining ---")
            lbfgs = optim.LBFGS(params, max_iter=1000,
                                line_search_fn="strong_wolfe",
                                tolerance_change=1e-9)

            def closure():
                lbfgs.zero_grad()
                lp_l, ln_l = self.compute_losses()
                loss_l = lp_l + fixed_weight * ln_l
                loss_l.backward()
                return loss_l

            lbfgs.step(closure)

            final_lp, final_ln = self.compute_losses()
            print(f"Final Convergence | Loss: {(final_lp + fixed_weight * final_ln).item():.2e}")

    def plot_graph_topology(self):
        G = nx.Graph()
        for u, v, L in self.graph.edges: G.add_edge(u, v, weight=L)
        plt.figure(figsize=(5, 3))
        nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
        plt.title("Metric Graph Setup")
        plt.show()

    def report_l2(self, exact_func):
        print("\n--- Error Report ---")
        for i, m in self.models.items():
            x = m['x_grid'].cpu().numpy().flatten()
            u_pred = m['net'](m['x_grid']).detach().cpu().numpy().flatten()
            u_true = exact_func(x, m['L'])
            rel_l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
            print(f"  Edge {i} {m['nodes']}: Rel L2 = {rel_l2:.4e}")

    def post_process(self, u_exact_func=None):
        num_edges = len(self.models)
        fig, axes = plt.subplots(num_edges, 1, figsize=(8, 3 * num_edges))
        if num_edges == 1: axes = [axes]
        for i, m in self.models.items():
            xt = torch.linspace(0, m['L'], 200, device=device).view(-1, 1)
            u_p = m['net'](xt).detach().cpu().numpy().flatten()
            x_np = xt.cpu().numpy().flatten()
            axes[i].plot(x_np, u_p, 'r-', label='PINN')
            if u_exact_func:
                axes[i].plot(x_np, u_exact_func(x_np, m['L']), 'k--', alpha=0.5, label='Exact')
            axes[i].set_title(f"Edge {i}")
            axes[i].legend();
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout();
        plt.show()


