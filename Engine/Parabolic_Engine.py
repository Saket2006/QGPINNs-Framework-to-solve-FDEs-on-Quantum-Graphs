import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FourierEmbedding(nn.Module):
    def __init__(self, in_dim=2, embed_dim=64, sigma=1.0, sampling="sobol"):
        super().__init__()
        n_freqs = embed_dim // 2
        if sampling == "sobol":
            sobol = torch.quasirandom.SobolEngine(dimension=in_dim, scramble=True)
            u = sobol.draw(n_freqs).double()
            u = torch.clamp(u, 1e-6, 1 - 1e-6)
            B = torch.erfinv(2 * u - 1) * (2 ** 0.5) * sigma
            B = B.T
        else:
            B = torch.randn(in_dim, n_freqs) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class PINN_Net(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, hidden_layers=3,
                 use_fourier=False, fourier_dim=64, fourier_sigma=1.0,
                 fourier_sampling="sobol"):
        super().__init__()
        self.use_fourier = use_fourier

        if use_fourier:
            self.fourier = FourierEmbedding(in_dim, fourier_dim, fourier_sigma, fourier_sampling)
            first_in = fourier_dim
        else:
            self.fourier = None
            first_in = in_dim

        layers = []
        layers.append(nn.Linear(first_in, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_fourier:
            x = self.fourier(x)
        return self.net(x)


class ParabolicPINNSolver:
    def __init__(self, graph, physics):
        self.graph = graph
        self.physics = physics
        self.compiled = False
        self.lambda_bc = 1.0
        self.models = {}

        self.use_anchors = False
        self.anchor_X = None
        self.anchor_T = None
        self.anchor_U = None
        self.lambda_data_schedule = None

        self.use_fourier = False
        self.fourier_dim = 64
        self.fourier_sigma = 1.0
        self.fourier_sampling = "sobol"

        self.use_causal = False
        self.causal_eps = 1.0

        self.use_time_windowing = False
        self.window_schedule = None
        self.current_t_max = None

        self.use_rad = False
        self.rad_every = 1000
        self.rad_k = 1

        self.use_ntk_balance = False
        self.ntk_every = 200

        self.lr = 5e-4
        self.scheduler_patience = 500
        self.scheduler_min_lr = 1e-6

        self.frac_scheme = "L1"
        self.l21_sigma = 0.5

    def set_architecture(self, hidden_layers=3, hidden_dim=64,
                         use_fourier=False, fourier_dim=64, fourier_sigma=1.0,
                         fourier_sampling="sobol"):
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.use_fourier = use_fourier
        self.fourier_dim = fourier_dim
        self.fourier_sigma = fourier_sigma
        self.fourier_sampling = fourier_sampling
        return self

    def set_mesh(self, mesh_type="power_law",
                 pinn_pts=100, anchor_pts=0,
                 grading_factor=1.0, t_max=1.0, n_t=100):
        self.mesh_type = mesh_type
        self.pinn_pts = pinn_pts
        self.anchor_pts = anchor_pts
        self.grading_factor = grading_factor
        self.t_max = t_max
        self.n_t = n_t
        return self

    def set_frac_scheme(self, scheme="L1", sigma=None):
        assert scheme in ("L1", "L21sigma"), "scheme must be 'L1' or 'L21sigma'"
        self.frac_scheme = scheme
        if scheme == "L21sigma":
            alpha = self.physics.alpha if hasattr(self.physics, 'alpha') else 0.5
            self.l21_sigma = (1.0 - alpha / 2.0) if sigma is None else float(sigma)
        return self

    def set_constraints(self, constraint_mode="soft", bc_types=None, bc_values=None):
        self.constraint_mode = constraint_mode.lower()
        self.bc_types = bc_types if bc_types else {}
        self.bc_values = bc_values if bc_values else {}
        return self

    def set_lambda_data_schedule(self, schedule):
        self.lambda_data_schedule = schedule
        return self

    def set_causal_training(self, enabled=True, eps=1.0):
        self.use_causal = enabled
        self.causal_eps = eps
        return self

    def set_time_windowing(self, enabled=True, schedule=None):
        self.use_time_windowing = enabled
        if schedule is not None:
            self.window_schedule = sorted(schedule)
        return self

    def set_rad_resampling(self, enabled=True, every=1000, k=1):
        self.use_rad = enabled
        self.rad_every = every
        self.rad_k = k
        return self

    def set_ntk_balancing(self, enabled=True):
        self.use_ntk_balance = enabled
        return self

    @staticmethod
    def _compute_l1_matrix_vectorized(alpha, grid_np):
        n = len(grid_np)
        if alpha <= 0:
            return torch.zeros(n, n, dtype=torch.float64).to(device)

        g1 = gamma(2 - alpha)
        h = np.diff(grid_np)

        I, J = np.tril_indices(n, k=-1)

        tau_ij  = grid_np[I] - grid_np[J]
        tau_ij1 = grid_np[I] - grid_np[J + 1]
        h_j = h[J]

        w = (tau_ij ** (1 - alpha) - tau_ij1 ** (1 - alpha)) / (h_j * g1)

        mat = np.zeros((n, n))
        np.add.at(mat, (I, J + 1), w)
        np.add.at(mat, (I, J),    -w)

        return torch.tensor(mat, dtype=torch.float64).to(device)

    @staticmethod
    def _compute_l21sigma_matrix(alpha, grid_np, sigma):
        n = len(grid_np)
        g2a = gamma(2 - alpha)
        tau = np.diff(grid_np)

        W = np.zeros((n, n))

        for row in range(1, n):
            t_nsig = (1.0 - sigma) * grid_np[row - 1] + sigma * grid_np[row]

            for k in range(row):
                tau_k = tau[k]
                left  = t_nsig - grid_np[k]
                right = t_nsig - grid_np[k + 1]
                if left <= 0:
                    continue
                right_c = max(right, 0.0)
                A_k = (left ** (1 - alpha) - right_c ** (1 - alpha)) / (tau_k * g2a)
                W[row, k + 1] += A_k
                W[row, k]     -= A_k

        return torch.tensor(W, dtype=torch.float64).to(device)

    @staticmethod
    def compute_l1_matrix(alpha, grid_np):
        return ParabolicPINNSolver._compute_l1_matrix_vectorized(alpha, grid_np)

    @staticmethod
    def compute_l21sigma_matrix(alpha, grid_np, sigma=None):
        if sigma is None:
            sigma = 1.0 - alpha / 2.0
        return ParabolicPINNSolver._compute_l21sigma_matrix(alpha, grid_np, sigma)

    def compile(self):
        t_np = self._generate_mesh(self.t_max, self.n_t)
        self.t_grid = torch.tensor(t_np).view(-1, 1).to(device)

        if hasattr(self.physics, 'alpha'):
            alpha = self.physics.alpha
            if self.frac_scheme == "L21sigma":
                if not hasattr(self, 'l21_sigma') or self.l21_sigma is None:
                    self.l21_sigma = 1.0 - alpha / 2.0
                self.Dt_alpha    = self._compute_l21sigma_matrix(alpha, t_np, self.l21_sigma)
                t_nsig_np        = np.empty(self.n_t)
                t_nsig_np[0]     = t_np[0]
                t_nsig_np[1:]    = (1.0 - self.l21_sigma) * t_np[:-1] + self.l21_sigma * t_np[1:]
                self.t_nsig_grid = torch.tensor(t_nsig_np).view(-1, 1).to(device)
                print(f"[FracScheme] Using L2-1σ  (σ={self.l21_sigma:.4f}, α={alpha})")
            else:
                self.Dt_alpha    = self._compute_l1_matrix_vectorized(alpha, t_np)
                self.t_nsig_grid = None
                print(f"[FracScheme] Using L1  (α={alpha})")

        for i, edge in enumerate(self.graph.edges):
            u_node, v_node, L = edge[0], edge[1], edge[2]
            n_x = max(int(self.pinn_pts * L), 2)
            x_np = self._generate_mesh(L, n_x, is_spatial=True)
            x_grid = torch.tensor(x_np).view(-1, 1).to(device)

            net = PINN_Net(
                in_dim=2,
                hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers,
                use_fourier=self.use_fourier,
                fourier_dim=self.fourier_dim,
                fourier_sigma=self.fourier_sigma,
                fourier_sampling=self.fourier_sampling,
            ).to(device)

            T_mesh = self.t_grid.repeat_interleave(n_x, dim=0)
            if self.frac_scheme == "L21sigma":
                T_mesh_nsig = self.t_nsig_grid.repeat_interleave(n_x, dim=0)
            else:
                T_mesh_nsig = T_mesh

            self.models[i] = {
                'net':         net,
                'L':           L,
                'n_x':         n_x,
                'nodes':       (u_node, v_node),
                'x_grid':      x_grid,
                'X_mesh':      x_grid.repeat(self.n_t, 1),
                'T_mesh':      T_mesh,
                'T_mesh_nsig': T_mesh_nsig,
            }

            self.models[i]['f_target'] = self.physics.get_f_target(
                self.models[i]['X_mesh'],
                self.models[i]['T_mesh_nsig'],
                L
            ).view(-1, 1).to(device)

        if self.anchor_pts > 0:
            if self.frac_scheme == "L21sigma":
                self._run_l21sigma_inverse_solver()
            else:
                self._run_l1_inverse_solver()

            if self.lambda_data_schedule is None:
                total = getattr(self, '_planned_epochs', 10000)
                p1, p2 = total // 4, total // 2
                self.lambda_data_schedule = (
                    lambda ep, p1=p1, p2=p2:
                    0.1 if ep < p1 else 0.01 if ep < p2 else 0.0
                )

        if self.use_time_windowing and self.window_schedule is None:
            total = getattr(self, '_planned_epochs', 10000)
            frac = [0.2, 0.4, 0.6, 0.8, 1.0]
            self.window_schedule = [
                (int(total * frac[i]), self.t_max * frac[i]) for i in range(5)
            ]

        if self.use_time_windowing:
            self.current_t_max = self.window_schedule[0][1]

        self.compiled = True
        return self

    def _generate_mesh(self, max_val, n_pts, is_spatial=False):
        if is_spatial:
            return np.linspace(0, max_val, n_pts)
        if self.mesh_type == "power_law":
            idx = np.arange(n_pts)
            return max_val * (idx / (n_pts - 1)) ** self.grading_factor
        elif self.mesh_type == "symmetric":
            xi = np.linspace(-1, 1, n_pts)
            return (np.sign(xi) * np.abs(xi) ** self.grading_factor + 1) * 0.5 * max_val
        else:
            return np.linspace(0, max_val, n_pts)

    def predict(self, idx, x, t):
        m = self.models[idx]
        u_raw = m['net'](torch.cat([x, t], dim=-1))

        if self.constraint_mode == "hard":
            u_node, v_node = m['nodes'][0], m['nodes'][1]
            t_left  = self.bc_types.get(u_node, "dirichlet")
            t_right = self.bc_types.get(v_node, "dirichlet")

            v_left  = self._get_bc_value(u_node, t)
            v_right = self._get_bc_value(v_node, t)
            ic      = self.physics.get_ic(x)

            if t_left == "dirichlet" and t_right == "dirichlet":
                t_zero = torch.zeros_like(t)
                bc_l0  = self._get_bc_value(u_node, t_zero)
                bc_r0  = self._get_bc_value(v_node, t_zero)
                ic_l   = self.physics.get_ic(torch.zeros_like(x))
                ic_r   = self.physics.get_ic(torch.ones_like(x) * m['L'])

                compatible = (
                    torch.max(torch.abs(bc_l0 - ic_l)).item() < 1e-6 and
                    torch.max(torch.abs(bc_r0 - ic_r)).item() < 1e-6
                )

                g_x    = v_left + (x / m['L']) * (v_right - v_left)
                dist_x = x * (m['L'] - x)

                if compatible:
                    return g_x + dist_x * u_raw
                else:
                    w_ic = torch.exp(-20.0 * t)
                    return w_ic * ic + (1.0 - w_ic) * (g_x + dist_x * u_raw)
            else:
                return (1 - t) * ic + t * u_raw

        return u_raw

    def _get_bc_value(self, node_idx, t):
        bc_val = self.bc_values.get(node_idx, 0.0)
        if callable(bc_val):
            return bc_val(t)
        return bc_val

    def _rad_resample(self):
        for i, m in self.models.items():
            n_x = m['n_x']
            L   = m['L']
            new_X_rows = []

            for k in range(self.n_t):
                t_k    = self.t_grid[k]
                t_k_ev = self.t_nsig_grid[k] if self.frac_scheme == "L21sigma" and self.t_nsig_grid is not None else t_k
                n_cand = 10 * n_x
                x_cand = torch.rand(n_cand, 1, dtype=torch.float64).to(device) * L
                t_cand = t_k_ev.expand(n_cand, 1)

                with torch.no_grad():
                    dx   = L / (n_x * 10)
                    xp   = torch.clamp(x_cand + dx, 0.0, L)
                    xm   = torch.clamp(x_cand - dx, 0.0, L)
                    u_c  = self.predict(i, x_cand, t_cand)
                    u_xp = self.predict(i, xp, t_cand)
                    u_xm = self.predict(i, xm, t_cand)

                    u_x_c  = (u_xp - u_xm) / (2 * dx)
                    u_xx_c = (u_xp - 2 * u_c + u_xm) / dx ** 2

                    dt_approx = u_c / (t_k.clamp(min=1e-6) ** self.physics.alpha)

                    f_c = self.physics.get_f_target(x_cand, t_cand, L).view(-1, 1)
                    res = self.physics.F(
                        x_cand, t_cand, u_c, u_x_c, u_xx_c, dt_approx, f_c
                    ).abs().flatten()

                prob    = res ** self.rad_k
                prob    = prob / (prob.sum() + 1e-10)
                idx_sel = torch.multinomial(prob, n_x, replacement=True)
                new_X_rows.append(x_cand[idx_sel])

            new_X       = torch.cat(new_X_rows, dim=0).detach()
            m['X_mesh'] = new_X
            m['f_target'] = self.physics.get_f_target(
                m['X_mesh'], m['T_mesh_nsig'], L
            ).view(-1, 1).to(device)

    def compute_losses(self, epoch=0):
        lp, ln = 0, 0

        if self.use_time_windowing and self.window_schedule:
            for ep_thresh, t_max_thresh in reversed(self.window_schedule):
                if epoch >= ep_thresh:
                    self.current_t_max = t_max_thresh
                    break

        for i, m in self.models.items():
            X      = m['X_mesh'].clone().detach().requires_grad_(True)
            T      = m['T_mesh'].clone().detach()
            T_eval = m['T_mesh_nsig'].clone().detach().requires_grad_(True)

            if self.use_time_windowing:
                mask   = (T <= self.current_t_max).flatten()
                if mask.sum() == 0:
                    continue
                X      = X[mask].requires_grad_(True)
                T      = T[mask]
                T_eval = T_eval[mask].requires_grad_(True)

            n_x_m = m['n_x']

            if not self.use_time_windowing:
                u_grid     = self.predict(i, m['X_mesh'], m['T_mesh'])
                u_reshaped = u_grid.view(self.n_t, n_x_m)
                dt_alpha_u = torch.mm(self.Dt_alpha, u_reshaped).view(-1, 1)
            else:
                X_full          = m['X_mesh'].clone()
                T_full          = m['T_mesh'].clone()
                u_full          = self.predict(i, X_full, T_full).view(self.n_t, n_x_m)
                dt_alpha_u_full = torch.mm(self.Dt_alpha, u_full).view(-1, 1)
                mask_full       = (T_full <= self.current_t_max).flatten()
                dt_alpha_u      = dt_alpha_u_full[mask_full]

            u_sp   = self.predict(i, X, T_eval)
            u_x    = torch.autograd.grad(u_sp, X,   torch.ones_like(u_sp), create_graph=True)[0]
            u_xx   = torch.autograd.grad(u_x,  X,   torch.ones_like(u_x),  create_graph=True)[0]

            if self.use_time_windowing:
                f_eval = self.physics.get_f_target(X, T_eval, m['L']).view(-1, 1)
            else:
                f_eval = m['f_target']

            res = self.physics.F(X, T_eval, u_sp, u_x, u_xx, dt_alpha_u, f_eval)

            if self.use_causal and not self.use_time_windowing:
                res_t        = res.view(self.n_t, n_x_m)
                loss_t       = res_t.pow(2).mean(dim=1)
                loss_t       = loss_t[1:]
                cumsum       = torch.cumsum(loss_t, dim=0).roll(1)
                cumsum[0]    = 0.0
                weights      = torch.exp(-self.causal_eps * cumsum).detach()
                lp          += (weights * loss_t).mean()
            elif not self.use_time_windowing:
                res_t  = res.view(self.n_t, n_x_m)
                lp    += torch.mean(res_t[1:] ** 2)
            else:
                t0_mask = (T > 0).flatten()
                if t0_mask.sum() > 0:
                    lp += torch.mean(res[t0_mask] ** 2)

        for i, m in self.models.items():
            if self.constraint_mode == "soft":
                X_ic = m['x_grid']
                T_ic = torch.zeros_like(X_ic)
                u_ic = self.predict(i, X_ic, T_ic)
                ln  += torch.mean((u_ic - self.physics.get_ic(X_ic)) ** 2)

            for node_idx, x_val in [(m['nodes'][0], 0.0), (m['nodes'][1], m['L'])]:
                bc_type = self.bc_types.get(node_idx, "dirichlet")

                if self.constraint_mode == "hard" and bc_type == "dirichlet":
                    continue

                T_bc = self.t_grid.clone().detach().requires_grad_(True)

                if self.use_time_windowing:
                    mask_bc = (T_bc <= self.current_t_max).flatten()
                    if mask_bc.sum() == 0:
                        continue
                    T_bc = T_bc[mask_bc].requires_grad_(True)

                X_bc = torch.full_like(T_bc, x_val).requires_grad_(True)
                u_bc = self.predict(i, X_bc, T_bc)

                if bc_type == "dirichlet":
                    bc_val = self._get_bc_value(node_idx, T_bc)
                    ln    += torch.mean((u_bc - bc_val) ** 2)
                elif bc_type == "neumann":
                    bc_val = self._get_bc_value(node_idx, T_bc)
                    u_x_bc = torch.autograd.grad(
                        u_bc, X_bc, torch.ones_like(u_bc), create_graph=True
                    )[0]
                    ln += torch.mean((u_x_bc - bc_val) ** 2)

        l_data = torch.tensor(0.0).to(device)
        if self.use_anchors and hasattr(self, 'current_epoch'):
            lam = self.lambda_data_schedule(self.current_epoch)
            if lam > 0.0:
                u_pred = self.predict(0, self.anchor_X, self.anchor_T)
                l_data = lam * torch.mean((u_pred - self.anchor_U) ** 2)

        if ln == 0:
            ln = torch.tensor(0.0, requires_grad=True).to(device)

        return lp, ln, l_data

    def update_weights_bdmm(self, ln):
        if ln.item() == 0: return
        self.lambda_bc += 0.5 * ln.item()
        self.lambda_bc = min(self.lambda_bc, 1000.0)

    def update_weights_gradient_ratio(self, lp, ln, params):
        if ln.item() == 0: return
        grad_lp = torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)
        grad_ln = torch.autograd.grad(ln, params, retain_graph=True, allow_unused=True)
        norm_lp = torch.cat([g.view(-1) for g in grad_lp if g is not None]).norm()
        norm_ln = torch.cat([g.view(-1) for g in grad_ln if g is not None]).norm()
        if norm_ln > 1e-8:
            target = norm_lp / (norm_ln + 1e-8)
            self.lambda_bc = 0.9 * self.lambda_bc + 0.1 * target.item()
            self.lambda_bc = float(np.clip(self.lambda_bc, 0.01, 1000.0))

    def update_weights_ntk(self, lp, ln, params):
        if ln.item() == 0:
            return
        grad_lp = torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)
        grad_ln = torch.autograd.grad(ln, params, retain_graph=True, allow_unused=True)

        norm_p = sum(g.pow(2).sum() for g in grad_lp if g is not None)
        norm_n = sum(g.pow(2).sum() for g in grad_ln if g is not None)

        if norm_n > 1e-16:
            target = (norm_p / norm_n).sqrt().item()
            self.lambda_bc = 0.99 * self.lambda_bc + 0.01 * target
            self.lambda_bc = float(np.clip(self.lambda_bc, 0.01, 1000.0))

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        return self

    def train_multistart(self, epochs=3000, strategy="dual", use_lbfgs=True, n_starts=5, seed_base=0):
        best_global_loss = float('inf')
        best_global_weights = None

        for s in range(n_starts):
            seed = seed_base + s
            torch.manual_seed(seed)
            np.random.seed(seed)

            for m in self.models.values():
                for layer in m['net'].net:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                        nn.init.zeros_(layer.bias)

            self.lambda_bc = 1.0
            self.compiled = True

            print(f"\n{'='*50}")
            print(f"Multi-start {s+1}/{n_starts}  (seed={seed})")
            print(f"{'='*50}")

            self.train(epochs=epochs, strategy=strategy, use_lbfgs=False)

            params = [p for m in self.models.values() for p in m['net'].parameters()]
            lp, ln, _ = self.compute_losses(0)
            candidate_loss = lp.item() + ln.item()

            if candidate_loss < best_global_loss:
                best_global_loss = candidate_loss
                best_global_weights = [p.data.clone() for p in params]
                print(f"  New best: {best_global_loss:.2e}")

        params = [p for m in self.models.values() for p in m['net'].parameters()]
        for p, best_p in zip(params, best_global_weights):
            p.data.copy_(best_p)

        if use_lbfgs:
            print("\nFinal L-BFGS refinement on best multi-start solution...")
            lbfgs = optim.LBFGS(params, max_iter=3000, line_search_fn="strong_wolfe")
            epoch = epochs

            def closure():
                lbfgs.zero_grad()
                lp_l, ln_l, _ = self.compute_losses(epoch)
                loss_l = lp_l + self.lambda_bc * ln_l
                loss_l.backward()
                return loss_l

            lbfgs.step(closure)

        return self

    def train(self, epochs=3000, strategy="dual", use_lbfgs=True):
        self._planned_epochs = epochs
        if not self.compiled:
            self.compile()

        params    = [p for m in self.models.values() for p in m['net'].parameters()]
        optimizer = optim.Adam(params, lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs + 1, eta_min=self.scheduler_min_lr
        )

        best_loss    = float('inf')
        best_weights = None

        eval_every = 500
        if hasattr(self, 'validation_func') and hasattr(self, 'validation_times'):
            use_validation = True
        else:
            use_validation = False

        anchor_str = f", anchors={len(self.anchor_X)}" if self.use_anchors else ""
        print(f"Training (Mode: {self.constraint_mode}, Strategy: {strategy}{anchor_str})")

        for epoch in range(epochs + 1):
            self.current_epoch = epoch

            if self.use_causal:
                progress = min(epoch / max(epochs // 2, 1), 1.0)
                self.causal_eps = self.causal_eps * (1.0 - 0.5 * progress)

            if self.use_rad and epoch > 0 and epoch % self.rad_every == 0:
                self._rad_resample()

            if strategy == "dual":
                active_strat = "bdmm" if epoch < (epochs // 2) else "gradient_ratio"
            else:
                active_strat = strategy

            optimizer.zero_grad()
            lp, ln, l_data = self.compute_losses(epoch)

            if epoch > 0 and epoch % 10 == 0:
                if self.use_ntk_balance:
                    if epoch % self.ntk_every == 0:
                        self.update_weights_ntk(lp, ln, params)
                    else:
                        self.update_weights_gradient_ratio(lp, ln, params)
                elif active_strat == "bdmm":
                    self.update_weights_bdmm(ln)
                elif active_strat == "gradient_ratio":
                    self.update_weights_gradient_ratio(lp, ln, params)

            loss = lp + self.lambda_bc * ln + l_data

            if use_validation and epoch % eval_every == 0:
                with torch.no_grad():
                    val_errors = []
                    for i, m in self.models.items():
                        for t_val in self.validation_times:
                            t_t = torch.full((100, 1), t_val).to(device)
                            x_t = torch.linspace(0, m['L'], 100).view(-1, 1).to(device)
                            u_p = self.predict(i, x_t, t_t).detach().cpu().numpy()
                            u_e = self.validation_func(x_t.cpu().numpy(), t_val)
                            err = np.linalg.norm(u_p - u_e) / (np.linalg.norm(u_e) + 1e-10)
                            val_errors.append(err)
                    val_loss = np.mean(val_errors)
                    if val_loss < best_loss:
                        best_loss    = val_loss
                        best_weights = [p.data.clone() for p in params]
            else:
                unweighted_loss = lp.item() + ln.item()
                if unweighted_loss < best_loss:
                    best_loss    = unweighted_loss
                    best_weights = [p.data.clone() for p in params]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if epoch % 500 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                ld_str = (f" | λ_data: {self.lambda_data_schedule(epoch):.2f}"
                          if self.use_anchors else "")
                tw_str = (f" | t_max: {self.current_t_max:.2f}"
                          if self.use_time_windowing else "")
                print(
                    f"Epoch {epoch:5d} | Total Loss: {loss.item():.2e}"
                    f" | Best True Error: {best_loss:.2e}"
                    f" | λ: {self.lambda_bc:.2f}{ld_str}{tw_str}"
                    f" | LR: {current_lr:.1e}"
                )

        if best_weights is not None:
            print(f"\n[Checkpointer] Restoring best Adam weights"
                  f" (True Error: {best_loss:.2e}) for L-BFGS Refinement...")
            for p, best_p in zip(params, best_weights):
                p.data.copy_(best_p)

        if use_lbfgs:
            if best_loss < 1e-3:
                print(f"L-BFGS skipped (Adam loss {best_loss:.2e} already below 1e-3)")
            else:
                print("L-BFGS Refining...")
                pre_lbfgs_weights = [p.data.clone() for p in params]
                pre_lbfgs_loss    = best_loss

                lbfgs = optim.LBFGS(
                    params,
                    max_iter=200,
                    history_size=50,
                    line_search_fn="strong_wolfe",
                    tolerance_change=1e-12,
                    tolerance_grad=1e-10,
                )

                def closure():
                    lbfgs.zero_grad()
                    lp_l, ln_l, l_data_l = self.compute_losses(epochs)
                    loss_l = lp_l + self.lambda_bc * ln_l + l_data_l
                    loss_l.backward()
                    return loss_l

                lbfgs.step(closure)

                with torch.no_grad():
                    lp_post, ln_post, _ = self.compute_losses(epochs)
                    post_lbfgs_loss = lp_post.item() + ln_post.item()

                if post_lbfgs_loss > pre_lbfgs_loss * 2.0:
                    print(f"L-BFGS degraded ({pre_lbfgs_loss:.2e} -> {post_lbfgs_loss:.2e}), restoring Adam weights")
                    for p, w in zip(params, pre_lbfgs_weights):
                        p.data.copy_(w)
                else:
                    print(f"L-BFGS accepted ({pre_lbfgs_loss:.2e} -> {post_lbfgs_loss:.2e})")

    def report_l2(self, exact_func, eval_times=None):
        if eval_times is None:
            eval_times = [1.0]
        errors = []
        for i, m in self.models.items():
            for t_desired in eval_times:
                t_t = torch.full((100, 1), t_desired).to(device)
                x_t = torch.linspace(0, m['L'], 100).view(-1, 1).to(device)
                u_p = self.predict(i, x_t, t_t).detach().cpu().numpy()
                u_e = exact_func(x_t.cpu().numpy(), t_desired)

                norm_exact = np.linalg.norm(u_e)
                if norm_exact < 1e-10:
                    print(f"t={t_desired:.4f}: skipped (exact solution is zero)")
                    continue

                err = np.linalg.norm(u_p - u_e) / norm_exact
                errors.append((t_desired, err))
                print(f"t={t_desired:.4f}: {err:.4e}")

        if errors:
            print(f"Mean: {np.mean([e for _, e in errors]):.4e}")
        return [e for _, e in errors]

    def plot_results(self, exact_func, t_vals=(0.25, 0.5, 0.75, 1.0)):
        fig, axs = plt.subplots(1, len(t_vals), figsize=(4 * len(t_vals), 4))
        if len(t_vals) == 1:
            axs = [axs]
        for idx, t_val in enumerate(t_vals):
            ax = axs[idx]
            for i, m in self.models.items():
                x_plot = np.linspace(0, m['L'], 100)
                x_t    = torch.tensor(x_plot).view(-1, 1).to(device)
                t_t    = torch.full_like(x_t, t_val)
                u_p    = self.predict(i, x_t, t_t).detach().cpu().numpy()
                u_e    = exact_func(x_plot, t_val)
                ax.plot(x_plot, u_e, 'k--', label="Exact"            if i == 0 else "")
                ax.plot(x_plot, u_p, 'r-',  label=f"PINN (Edge {i})" if i == 0 else "")
            ax.set_title(f"t = {t_val}")
            ax.set_xlabel("x")
            if idx == 0: ax.set_ylabel("u(x,t)")
            ax.legend()
        plt.tight_layout()
        plt.show()

    def _solve_tridiag(self, a, b, c, d):
        n = len(b)
        b = b.copy().astype(float)
        d = d.copy().astype(float)
        x = np.zeros(n)
        for i in range(1, n):
            m_ = a[i - 1] / b[i - 1]
            b[i] -= m_ * c[i - 1]
            d[i] -= m_ * d[i - 1]
        x[-1] = d[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
        return x

    def _run_l1_inverse_solver(self):
        print(f"\n[Inverse Solver] L1  (spatial={self.anchor_pts}, temporal={self.n_t})")

        n_x   = self.anchor_pts
        t_np  = self.t_grid.cpu().numpy().flatten()
        n_t   = len(t_np)
        L     = self.graph.edges[0][2]
        x_np  = np.linspace(0, L, n_x)
        dx    = x_np[1] - x_np[0]
        alpha = self.physics.alpha
        g2a   = gamma(2 - alpha)

        def l1_weights(n):
            b = np.zeros(n)
            for k in range(n):
                dt_k  = t_np[k + 1] - t_np[k]
                tau_r = t_np[n] - t_np[k]
                tau_l = t_np[n] - t_np[k + 1]
                b[k]  = (tau_r ** (1 - alpha) - max(tau_l, 0.0) ** (1 - alpha)) / (dt_k * g2a)
            return b

        x_all = torch.tensor(x_np, dtype=torch.float64).view(-1, 1)
        u     = np.zeros((n_t, n_x))
        u[0]  = self.physics.get_ic(x_all).detach().numpy().flatten()
        u[0,  0] = self._eval_bc_scalar(self.graph.edges[0][0], t_np[0])
        u[0, -1] = self._eval_bc_scalar(self.graph.edges[0][1], t_np[0])

        f_all = np.zeros((n_t, n_x))
        for ni in range(n_t):
            t_t       = torch.full_like(x_all, t_np[ni])
            f_all[ni] = self.physics.get_f_target(x_all, t_t, L).detach().numpy().flatten()

        cd = 0.1 / dx ** 2

        for n in range(1, n_t):
            bcl = self._eval_bc_scalar(self.graph.edges[0][0], t_np[n])
            bcr = self._eval_bc_scalar(self.graph.edges[0][1], t_np[n])

            bw   = l1_weights(n)
            b0   = bw[-1]
            hist = np.zeros(n_x)
            for k in range(n - 1):
                hist += bw[k] * (u[k + 1] - u[k])
            f_n = f_all[n]

            ug     = u[n - 1].copy()
            ug[0]  = bcl
            ug[-1] = bcr

            for _ in range(15):
                uo  = ug.copy()
                ux  = np.zeros(n_x)
                uxx = np.zeros(n_x)
                for j in range(1, n_x - 1):
                    ux[j]  = (uo[j + 1] - uo[j - 1]) / (2 * dx)
                    uxx[j] = (uo[j + 1] - 2 * uo[j] + uo[j - 1]) / dx ** 2
                ux[0]   = (uo[1] - uo[0]) / dx
                ux[-1]  = (uo[-1] - uo[-2]) / dx
                uxx[0]  = uxx[1]
                uxx[-1] = uxx[-2]

                ni2  = n_x - 2
                sub  = np.full(ni2 - 1, -cd)
                main = np.full(ni2, b0 + 2 * cd)
                sup  = np.full(ni2 - 1, -cd)
                rhs  = np.array([b0 * u[n - 1, j] - hist[j] - f_n[j] - ug[j] * ux[j]
                                 for j in range(1, n_x - 1)])
                rhs[0]  += cd * bcl
                rhs[-1] += cd * bcr

                un      = np.zeros(n_x)
                un[0]   = bcl
                un[-1]  = bcr
                un[1:-1] = self._solve_tridiag(sub, main, sup, rhs)

                if np.max(np.abs(un - uo)) < 1e-10:
                    ug = un
                    break
                ug = un

            u[n] = ug
            if not np.all(np.isfinite(u[n])):
                u[n] = np.nan_to_num(u[n], nan=0.0, posinf=0.0, neginf=0.0)

        T_m, X_m = np.meshgrid(t_np, x_np, indexing='ij')
        self.anchor_X = torch.tensor(X_m.flatten(), dtype=torch.float64).view(-1, 1).to(device)
        self.anchor_T = torch.tensor(T_m.flatten(), dtype=torch.float64).view(-1, 1).to(device)
        self.anchor_U = torch.tensor(u.flatten(),   dtype=torch.float64).view(-1, 1).to(device)
        self.use_anchors = True
        print(f"  Done: {len(self.anchor_X)} anchors  u∈[{u.min():.4f}, {u.max():.4f}]")

    def _run_l21sigma_inverse_solver(self):
        print(f"\n[Inverse Solver] L2-1σ  (σ={self.l21_sigma:.4f}, spatial={self.anchor_pts}, temporal={self.n_t})")

        n_x   = self.anchor_pts
        t_np  = self.t_grid.cpu().numpy().flatten()
        n_t   = len(t_np)
        L     = self.graph.edges[0][2]
        x_np  = np.linspace(0, L, n_x)
        dx    = x_np[1] - x_np[0]
        alpha = self.physics.alpha
        sigma = self.l21_sigma
        g2a   = gamma(2 - alpha)
        tau   = np.diff(t_np)

        def l21_diagonal_coeff(n):
            return (sigma ** (1 - alpha)) * (tau[n - 1] ** (-alpha)) / g2a

        def l21_history(n, u_all):
            t_nsig = (1.0 - sigma) * t_np[n - 1] + sigma * t_np[n]
            acc    = np.zeros(n_x)

            for k in range(n - 1):
                tau_k = tau[k]
                left  = t_nsig - t_np[k]
                right = t_nsig - t_np[k + 1]
                if left <= 0:
                    continue
                right_c = max(right, 0.0)
                A_k = (left ** (1 - alpha) - right_c ** (1 - alpha)) / (tau_k * g2a)
                acc += A_k * (u_all[k + 1] - u_all[k])

            return acc

        x_all = torch.tensor(x_np, dtype=torch.float64).view(-1, 1)
        u     = np.zeros((n_t, n_x))
        u[0]  = self.physics.get_ic(x_all).detach().numpy().flatten()
        u[0,  0] = self._eval_bc_scalar(self.graph.edges[0][0], t_np[0])
        u[0, -1] = self._eval_bc_scalar(self.graph.edges[0][1], t_np[0])

        cd = self.physics.nu / dx ** 2 if hasattr(self.physics, 'nu') else 0.1 / dx ** 2

        for n in range(1, n_t):
            bcl = self._eval_bc_scalar(self.graph.edges[0][0], t_np[n])
            bcr = self._eval_bc_scalar(self.graph.edges[0][1], t_np[n])

            b0      = l21_diagonal_coeff(n)
            hist    = l21_history(n, u)
            t_nsig  = (1.0 - sigma) * t_np[n - 1] + sigma * t_np[n]

            x_int_t  = torch.tensor(x_np[1:-1], dtype=torch.float64).view(-1, 1)
            t_nsig_t = torch.full((n_x - 2, 1), t_nsig, dtype=torch.float64)
            f_nsig   = self.physics.get_f_target(x_int_t, t_nsig_t, L).detach().numpy().flatten()

            cd_impl = cd * sigma
            cd_expl = cd * (1.0 - sigma)

            ni2       = n_x - 2
            sub       = np.full(ni2 - 1, -cd_impl)
            main      = np.full(ni2,      b0 + 2.0 * cd_impl)
            sup       = np.full(ni2 - 1, -cd_impl)
            uxx_expl  = u[n - 1, 2:] - 2.0 * u[n - 1, 1:-1] + u[n - 1, :-2]

            rhs        = b0 * u[n - 1, 1:-1] - hist[1:-1] + f_nsig + cd_expl * uxx_expl
            rhs[0]    += cd_impl * bcl         + cd_expl * u[n - 1, 0]
            rhs[-1]   += cd_impl * bcr         + cd_expl * u[n - 1, -1]

            un         = np.zeros(n_x)
            un[0]      = bcl
            un[-1]     = bcr
            un[1:-1]   = self._solve_tridiag(sub, main, sup, rhs)

            if not np.all(np.isfinite(un)):
                un = np.nan_to_num(un, nan=0.0, posinf=0.0, neginf=0.0)

            u[n] = un

        T_m, X_m = np.meshgrid(t_np, x_np, indexing='ij')
        self.anchor_X = torch.tensor(X_m.flatten(), dtype=torch.float64).view(-1, 1).to(device)
        self.anchor_T = torch.tensor(T_m.flatten(), dtype=torch.float64).view(-1, 1).to(device)
        self.anchor_U = torch.tensor(u.flatten(),   dtype=torch.float64).view(-1, 1).to(device)
        self.use_anchors = True
        print(f"  Done: {len(self.anchor_X)} anchors  u∈[{u.min():.4f}, {u.max():.4f}]")

    def _eval_bc_scalar(self, node_idx, t_scalar):
        bc = self.bc_values.get(node_idx, 0.0)
        if callable(bc):
            t_t = torch.tensor([[t_scalar]], dtype=torch.float64)
            out = bc(t_t)
            return float(out.item() if isinstance(out, torch.Tensor) else out)
        return float(bc)
