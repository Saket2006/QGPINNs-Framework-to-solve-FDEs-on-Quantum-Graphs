%%writefile Parabolic_Engine.py

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


def _alikhanov_row(alpha, t_full, n):
    sigma = alpha / 2.0
    g2a   = gamma(2.0 - alpha)
    t     = t_full[:n + 1]
    tau   = np.diff(t)
    t_ns  = (1.0 - sigma) * t[n] + sigma * t[n - 1]

    def xi_bar(k):
        left  = t_ns - t[k - 1]
        right = t_ns - t[k]
        if left <= 0.0: return 0.0
        return (left**(1-alpha) - max(right,0.0)**(1-alpha)) / (tau[k-1] * g2a)

    def xi_tilde(k):
        if k < 1 or k > n - 1: return 0.0
        tau_k = tau[k-1]; tau_k1 = tau[k]
        left  = t_ns - t[k-1]; right = t_ns - t[k]
        if left <= 0.0: return 0.0
        right = max(right, 0.0)
        num = (-(tau_k/2.0)*(left**(1-alpha)+right**(1-alpha))
               + (left**(2-alpha)-right**(2-alpha))/(2-alpha))
        return num / (tau_k*(tau_k+tau_k1)*g2a/2.0)

    def rho(k):
        if k < 1 or k >= n: return 0.0
        return tau[k-1]/tau[k]

    if n == 1:
        w0 = (1-sigma)**(1-alpha)*tau[0]**(-alpha)/g2a
        w = np.zeros(2); w[1]=w0; w[0]=-w0; return w

    omega = np.zeros(n)
    omega[0] = xi_bar(n) + rho(n-1)*xi_tilde(n-1)
    for j in range(1, n-1):
        k = n-j; omega[j] = xi_bar(k) + rho(k-1)*xi_tilde(k-1) - xi_tilde(k)
    omega[n-1] = xi_bar(1) - xi_tilde(1)

    w = np.zeros(n+1); w[n] = omega[0]
    for j in range(1, n): w[n-j] -= (omega[j-1]-omega[j])
    w[0] -= omega[n-1]
    return w


def build_l21sigma_matrix_alikhanov(alpha, grid_np):
    n = len(grid_np)
    W = np.zeros((n, n))
    for row in range(1, n):
        w = _alikhanov_row(alpha, grid_np, row)
        W[row, :row+1] = w
    return torch.tensor(W, dtype=torch.float64).to(device)


class ParabolicPINNSolver:
    def __init__(self, graph, physics):
        self.graph     = graph
        self.physics   = physics
        self.compiled  = False
        self.lambda_bc = 1.0
        self.models    = {}

        self.use_fourier      = False
        self.fourier_dim      = 64
        self.fourier_sigma    = 1.0
        self.fourier_sampling = "sobol"

        self.use_causal  = False
        self.causal_eps  = 1.0

        self.use_time_windowing = False
        self.window_schedule    = None
        self.current_t_max      = None

        self.use_rad   = False
        self.rad_every = 1000
        self.rad_k     = 1

        self.use_ntk_balance = False
        self.ntk_every       = 200

        self.lr               = 5e-4
        self.scheduler_min_lr = 1e-6

        self.frac_scheme = "L1"
        self.l21_sigma   = None
        self.t_nsig_grid = None

        self.validation_func  = None
        self.validation_times = None

        self.l2_history = []

        self._inverse_params      = []
        self._inv_x_obs           = None
        self._inv_t_obs           = None
        self._inv_u_obs           = None
        self._inv_lambda_data     = 1.0
        self._inv_param_histories = {}
        self._inv_mode            = False

    def set_inverse_param(self, name, param):
        self._inverse_params.append((name, param))
        self._inv_param_histories[name] = []
        self._inv_mode = True
        return self

    def set_observations(self, x_obs, t_obs, u_obs, lambda_data=1.0):
        self._inv_x_obs       = x_obs
        self._inv_t_obs       = t_obs
        self._inv_u_obs       = u_obs
        self._inv_lambda_data = lambda_data
        return self

    def set_architecture(self, hidden_layers=3, hidden_dim=64,
                         use_fourier=False, fourier_dim=64, fourier_sigma=1.0,
                         fourier_sampling="sobol"):
        self.hidden_layers    = hidden_layers
        self.hidden_dim       = hidden_dim
        self.use_fourier      = use_fourier
        self.fourier_dim      = fourier_dim
        self.fourier_sigma    = fourier_sigma
        self.fourier_sampling = fourier_sampling
        return self

    def set_mesh(self, mesh_type="power_law", pinn_pts=100,
                 grading_factor=1.0, t_max=1.0, n_t=100):
        self.mesh_type      = mesh_type
        self.pinn_pts       = pinn_pts
        self.grading_factor = grading_factor
        self.t_max          = t_max
        self.n_t            = n_t
        return self

    def set_frac_scheme(self, scheme="L1", sigma=None):
        assert scheme in ("L1", "L21sigma")
        self.frac_scheme = scheme
        if scheme == "L21sigma":
            alpha = self.physics.alpha if hasattr(self.physics, 'alpha') else 0.5
            self.l21_sigma = alpha / 2.0 if sigma is None else float(sigma)
        return self

    def set_constraints(self, constraint_mode="soft", bc_types=None, bc_values=None):
        self.constraint_mode = constraint_mode.lower()
        self.bc_types        = bc_types  if bc_types  else {}
        self.bc_values       = bc_values if bc_values else {}
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
        self.use_rad   = enabled
        self.rad_every = every
        self.rad_k     = k
        return self

    def set_ntk_balancing(self, enabled=True):
        self.use_ntk_balance = enabled
        return self

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        return self

    def set_validation(self, func, times):
        self.validation_func  = func
        self.validation_times = list(times)
        return self

    @staticmethod
    def _compute_l1_matrix_vectorized(alpha, grid_np):
        n = len(grid_np)
        if alpha <= 0:
            return torch.zeros(n, n, dtype=torch.float64).to(device)
        g1   = gamma(2 - alpha)
        h    = np.diff(grid_np)
        I, J = np.tril_indices(n, k=-1)
        tau_ij  = grid_np[I] - grid_np[J]
        tau_ij1 = grid_np[I] - grid_np[J + 1]
        w       = (tau_ij**(1-alpha) - tau_ij1**(1-alpha)) / (h[J] * g1)
        mat = np.zeros((n, n))
        np.add.at(mat, (I, J+1),  w)
        np.add.at(mat, (I, J),   -w)
        return torch.tensor(mat, dtype=torch.float64).to(device)

    @staticmethod
    def _compute_l21sigma_matrix(alpha, grid_np, sigma=None):
        return build_l21sigma_matrix_alikhanov(alpha, grid_np)

    @staticmethod
    def compute_l1_matrix(alpha, grid_np):
        return ParabolicPINNSolver._compute_l1_matrix_vectorized(alpha, grid_np)

    @staticmethod
    def compute_l21sigma_matrix(alpha, grid_np, sigma=None):
        return build_l21sigma_matrix_alikhanov(alpha, grid_np)

    def compile(self):
        t_np        = self._generate_mesh(self.t_max, self.n_t)
        self.t_grid = torch.tensor(t_np).view(-1, 1).to(device)

        if hasattr(self.physics, 'alpha'):
            alpha = self.physics.alpha
            if self.frac_scheme == "L21sigma":
                sigma            = alpha / 2.0
                self.l21_sigma   = sigma
                self.Dt_alpha    = build_l21sigma_matrix_alikhanov(alpha, t_np)
                t_nsig_np        = np.empty(self.n_t)
                t_nsig_np[0]     = t_np[0]
                t_nsig_np[1:]    = (1.0-sigma)*t_np[1:] + sigma*t_np[:-1]
                self.t_nsig_grid = torch.tensor(t_nsig_np).view(-1, 1).to(device)
            else:
                self.Dt_alpha    = self._compute_l1_matrix_vectorized(alpha, t_np)
                self.t_nsig_grid = self.t_grid

        for i, edge in enumerate(self.graph.edges):
            u_node, v_node, L = edge[0], edge[1], edge[2]
            n_x    = max(int(self.pinn_pts * L), 2)
            x_np   = self._generate_mesh(L, n_x, is_spatial=True)
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
            T_mesh      = self.t_grid.repeat_interleave(n_x, dim=0)
            T_mesh_nsig = self.t_nsig_grid.repeat_interleave(n_x, dim=0)
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

        if self.use_time_windowing and self.window_schedule is None:
            total = getattr(self, '_planned_epochs', 10000)
            frac  = [0.2, 0.4, 0.6, 0.8, 1.0]
            self.window_schedule = [
                (int(total * frac[k]), self.t_max * frac[k]) for k in range(5)
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
            return (np.sign(xi) * np.abs(xi)**self.grading_factor + 1) * 0.5 * max_val
        else:
            return np.linspace(0, max_val, n_pts)

    def predict(self, idx, x, t):
        m     = self.models[idx]
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
            n_x = m['n_x']; L = m['L']; new_X_rows = []
            for k in range(self.n_t):
                t_k    = self.t_grid[k]
                t_k_ev = self.t_nsig_grid[k] if self.t_nsig_grid is not None else t_k
                n_cand = 10 * n_x
                x_cand = torch.rand(n_cand, 1, dtype=torch.float64).to(device) * L
                t_cand = t_k_ev.expand(n_cand, 1)
                with torch.no_grad():
                    dx     = L / (n_x * 10)
                    xp     = torch.clamp(x_cand + dx, 0.0, L)
                    xm     = torch.clamp(x_cand - dx, 0.0, L)
                    u_c    = self.predict(i, x_cand, t_cand)
                    u_xp   = self.predict(i, xp, t_cand)
                    u_xm   = self.predict(i, xm, t_cand)
                    u_x_c  = (u_xp - u_xm) / (2 * dx)
                    u_xx_c = (u_xp - 2*u_c + u_xm) / dx**2
                    dt_approx = u_c / (t_k.clamp(min=1e-6)**self.physics.alpha)
                    f_c = self.physics.get_f_target(x_cand, t_cand, L).view(-1, 1)
                    res = self.physics.F(x_cand, t_cand, u_c, u_x_c, u_xx_c, dt_approx, f_c).abs().flatten()
                prob    = res**self.rad_k
                prob    = prob / (prob.sum() + 1e-10)
                idx_sel = torch.multinomial(prob, n_x, replacement=True)
                new_X_rows.append(x_cand[idx_sel])
            new_X       = torch.cat(new_X_rows, dim=0).detach()
            m['X_mesh'] = new_X
            m['f_target'] = self.physics.get_f_target(m['X_mesh'], m['T_mesh_nsig'], L).view(-1, 1).to(device)

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
                if mask.sum() == 0: continue
                X      = X[mask].requires_grad_(True)
                T      = T[mask]
                T_eval = T_eval[mask].requires_grad_(True)

            n_x_m = m['n_x']

            if not self.use_time_windowing:
                u_grid     = self.predict(i, m['X_mesh'], m['T_mesh'])
                u_reshaped = u_grid.view(self.n_t, n_x_m)
                dt_alpha_u = torch.mm(self.Dt_alpha, u_reshaped).view(-1, 1)
            else:
                u_full          = self.predict(i, m['X_mesh'], m['T_mesh']).view(self.n_t, n_x_m)
                dt_alpha_u_full = torch.mm(self.Dt_alpha, u_full).view(-1, 1)
                mask_full       = (m['T_mesh'] <= self.current_t_max).flatten()
                dt_alpha_u      = dt_alpha_u_full[mask_full]

            u_sp  = self.predict(i, X, T_eval)
            u_x   = torch.autograd.grad(u_sp, X,    torch.ones_like(u_sp), create_graph=True)[0]
            u_xx  = torch.autograd.grad(u_x,  X,    torch.ones_like(u_x),  create_graph=True)[0]

            f_eval = self.physics.get_f_target(X, T_eval, m['L']).view(-1, 1) \
                     if self.use_time_windowing else m['f_target']

            res = self.physics.F(X, T_eval, u_sp, u_x, u_xx, dt_alpha_u, f_eval)

            if self.use_causal and not self.use_time_windowing:
                res_t     = res.view(self.n_t, n_x_m)
                loss_t    = res_t.pow(2).mean(dim=1)[1:]
                cumsum    = torch.cumsum(loss_t, dim=0).roll(1)
                cumsum[0] = 0.0
                weights   = torch.exp(-self.causal_eps * cumsum).detach()
                lp       += (weights * loss_t).mean()
            elif not self.use_time_windowing:
                lp += torch.mean(res.view(self.n_t, n_x_m)[1:]**2)
            else:
                t0_mask = (T > 0).flatten()
                if t0_mask.sum() > 0:
                    lp += torch.mean(res[t0_mask]**2)

        for i, m in self.models.items():
            if self.constraint_mode == "soft":
                X_ic = m['x_grid']
                T_ic = torch.zeros_like(X_ic)
                u_ic = self.predict(i, X_ic, T_ic)
                ln  += torch.mean((u_ic - self.physics.get_ic(X_ic))**2)
            for node_idx, x_val in [(m['nodes'][0], 0.0), (m['nodes'][1], m['L'])]:
                bc_type = self.bc_types.get(node_idx, "dirichlet")
                if self.constraint_mode == "hard" and bc_type == "dirichlet":
                    continue
                T_bc = self.t_grid.clone().detach().requires_grad_(True)
                if self.use_time_windowing:
                    mask_bc = (T_bc <= self.current_t_max).flatten()
                    if mask_bc.sum() == 0: continue
                    T_bc = T_bc[mask_bc].requires_grad_(True)
                X_bc = torch.full_like(T_bc, x_val).requires_grad_(True)
                u_bc = self.predict(i, X_bc, T_bc)
                if bc_type == "dirichlet":
                    ln += torch.mean((u_bc - self._get_bc_value(node_idx, T_bc))**2)
                elif bc_type == "neumann":
                    u_x_bc = torch.autograd.grad(u_bc, X_bc, torch.ones_like(u_bc), create_graph=True)[0]
                    ln += torch.mean((u_x_bc - self._get_bc_value(node_idx, T_bc))**2)

        if ln == 0:
            ln = torch.tensor(0.0, requires_grad=True, dtype=torch.float64).to(device)

        l_data = torch.tensor(0.0, dtype=torch.float64).to(device)
        if self._inv_mode and self._inv_x_obs is not None:
            for name, param in self._inverse_params:
                self._inv_param_histories[name].append(param.item())
            u_pred = self.predict(0, self._inv_x_obs, self._inv_t_obs)
            l_data = self._inv_lambda_data * torch.mean((u_pred - self._inv_u_obs) ** 2)

        return lp, ln, l_data

    def update_weights_bdmm(self, ln):
        if ln.item() == 0: return
        self.lambda_bc = min(self.lambda_bc + 0.5*ln.item(), 1000.0)

    def update_weights_gradient_ratio(self, lp, ln, params):
        if ln.item() == 0: return
        gp = torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)
        gn = torch.autograd.grad(ln, params, retain_graph=True, allow_unused=True)
        np_ = torch.cat([g.view(-1) for g in gp if g is not None]).norm()
        nn_ = torch.cat([g.view(-1) for g in gn if g is not None]).norm()
        if nn_ > 1e-8:
            self.lambda_bc = float(np.clip(0.9*self.lambda_bc + 0.1*(np_/(nn_+1e-8)).item(), 0.01, 1000.0))

    def update_weights_ntk(self, lp, ln, params):
        if ln.item() == 0: return
        gp = torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)
        gn = torch.autograd.grad(ln, params, retain_graph=True, allow_unused=True)
        sp = sum(g.pow(2).sum() for g in gp if g is not None)
        sn = sum(g.pow(2).sum() for g in gn if g is not None)
        if sn > 1e-16:
            self.lambda_bc = float(np.clip(0.99*self.lambda_bc + 0.01*(sp/sn).sqrt().item(), 0.01, 1000.0))

    def _reinit_weights_only(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed)
        for m in self.models.values():
            for layer in m['net'].net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                    nn.init.zeros_(layer.bias)
        self.lambda_bc = 1.0

    def _reinit_weights(self, seed=None):
        self._reinit_weights_only(seed)

    def _collect_params(self):
        net_params = [p for m in self.models.values() for p in m['net'].parameters()]
        if self._inv_mode:
            return net_params + [param for _, param in self._inverse_params]
        return net_params

    def _eval_l2(self):
        if self.validation_func is None or self.validation_times is None:
            return float('inf')
        errors = []
        with torch.no_grad():
            for i, m in self.models.items():
                for t_val in self.validation_times:
                    t_t = torch.full((100,1), t_val, dtype=torch.float64).to(device)
                    x_t = torch.linspace(0, m['L'], 100, dtype=torch.float64).view(-1,1).to(device)
                    u_p = self.predict(i, x_t, t_t).detach().cpu().numpy()
                    u_e = self.validation_func(x_t.cpu().numpy(), t_val)
                    ne  = np.linalg.norm(u_e)
                    if ne < 1e-10: continue
                    errors.append(np.linalg.norm(u_p - u_e) / ne)
        return float(np.mean(errors)) if errors else float('inf')

    def _eval_l2_at_times(self, times):
        errors = {}
        with torch.no_grad():
            for t_val in times:
                errs = []
                for i, m in self.models.items():
                    t_t = torch.full((100,1), t_val, dtype=torch.float64).to(device)
                    x_t = torch.linspace(0, m['L'], 100, dtype=torch.float64).view(-1,1).to(device)
                    u_p = self.predict(i, x_t, t_t).detach().cpu().numpy()
                    u_e = self.validation_func(x_t.cpu().numpy(), t_val)
                    ne  = np.linalg.norm(u_e)
                    if ne < 1e-10: continue
                    errs.append(np.linalg.norm(u_p - u_e) / ne)
                errors[t_val] = float(np.mean(errs)) if errs else float('inf')
        return errors

    def _train_loop(self, epochs, strategy, use_lbfgs, optimizer, scheduler,
                    epoch_offset=0, global_best=None, log_l2_every=50):
        params         = self._collect_params()
        use_validation = self.validation_func is not None and self.validation_times is not None
        local_best_loss    = float('inf')
        local_best_weights = None
        total_planned      = epoch_offset + epochs

        for local_epoch in range(epochs + 1):
            epoch = epoch_offset + local_epoch
            self.current_epoch = epoch

            if self.use_causal:
                progress        = min(epoch / max(total_planned//2, 1), 1.0)
                self.causal_eps = self.causal_eps * (1.0 - 0.5*progress)

            if self.use_rad and local_epoch > 0 and local_epoch % self.rad_every == 0:
                self._rad_resample()

            active_strat = ("bdmm" if epoch < total_planned//2 else "gradient_ratio") \
                           if strategy == "dual" else strategy

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

            current_metric = self._eval_l2() if use_validation else lp.item()+ln.item()

            if use_validation and local_epoch % log_l2_every == 0:
                self.l2_history.append((epoch, current_metric))

            if current_metric < local_best_loss:
                local_best_loss    = current_metric
                local_best_weights = [p.data.clone() for p in params]
            if global_best is not None and current_metric < global_best['loss']:
                global_best['loss']    = current_metric
                global_best['weights'] = [p.data.clone() for p in params]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

        if local_best_weights is not None:
            for p, w in zip(params, local_best_weights): p.data.copy_(w)
        if use_lbfgs:
            self._run_lbfgs(params, epoch_offset + epochs)
        if global_best is not None:
            pm = self._eval_l2() if use_validation else \
                 sum(self.compute_losses(epoch_offset+epochs)[:2]).item()
            if pm < global_best['loss']:
                global_best['loss']    = pm
                global_best['weights'] = [p.data.clone() for p in params]
        return local_best_loss

    def train(self, epochs=3000, strategy="dual", use_lbfgs=True, log_l2_every=50):
        self._planned_epochs = epochs
        if not self.compiled: self.compile()
        params    = self._collect_params()
        optimizer = optim.Adam(params, lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs+1, eta_min=self.scheduler_min_lr)
        return self._train_loop(epochs=epochs, strategy=strategy, use_lbfgs=use_lbfgs,
                                optimizer=optimizer, scheduler=scheduler,
                                epoch_offset=0, global_best=None, log_l2_every=log_l2_every)

    def train_multistart(self, epochs=3000, strategy="dual", use_lbfgs=True,
                         n_starts=5, seed_base=0, probe_epochs=500, log_l2_every=50):
        if not self.compiled: self.compile()
        params      = self._collect_params()
        global_best = {'loss': float('inf'), 'weights': None}
        best_probe_score   = float('inf')
        best_probe_weights = None
        best_probe_lambda  = 1.0

        for s in range(n_starts):
            weight_seed = seed_base * 100 + s * 37
            self._reinit_weights_only(weight_seed)
            for name, param in self._inverse_params:
                init_val = getattr(param, '_inv_init', param.item())
                with torch.no_grad():
                    param.fill_(init_val)

            optimizer = optim.Adam(params, lr=self.lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=probe_epochs+1, eta_min=self.scheduler_min_lr)
            self._train_loop(epochs=probe_epochs, strategy=strategy, use_lbfgs=False,
                             optimizer=optimizer, scheduler=scheduler,
                             epoch_offset=0, global_best=global_best, log_l2_every=log_l2_every)
            use_val = self.validation_func is not None and self.validation_times is not None
            if use_val:
                score = self._eval_l2(); label = "L2"
            else:
                with torch.no_grad():
                    lp, ln, _ = self.compute_losses(probe_epochs)
                score = lp.item()+ln.item(); label = "train_loss"

            inv_summary = "  ".join(f"{n}={p.item():.6f}" for n, p in self._inverse_params)
            print(f"  probe {s+1}/{n_starts}  seed={weight_seed}  {label}={score:.3e}"
                  + (f"  [{inv_summary}]" if inv_summary else ""))

            if score < best_probe_score:
                best_probe_score   = score
                best_probe_weights = [p.data.clone() for p in params]
                best_probe_lambda  = self.lambda_bc

        for p, w in zip(params, best_probe_weights): p.data.copy_(w)
        self.lambda_bc = best_probe_lambda
        self.l2_history = []
        print(f"  -> continuing best probe (score={best_probe_score:.3e}) for {epochs} epochs")

        optimizer = optim.Adam(params, lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs+1, eta_min=self.scheduler_min_lr)
        self._train_loop(epochs=epochs, strategy=strategy, use_lbfgs=use_lbfgs,
                         optimizer=optimizer, scheduler=scheduler,
                         epoch_offset=0, global_best=global_best, log_l2_every=log_l2_every)

        if global_best['weights'] is not None:
            for p, w in zip(params, global_best['weights']): p.data.copy_(w)
            print(f"  -> global best L2 = {global_best['loss']:.3e}")
        return self

    def _run_lbfgs(self, params, epochs):
        pre_weights = [p.data.clone() for p in params]
        with torch.enable_grad():
            pre_loss = sum(self.compute_losses(epochs)[:2]).item()
        lbfgs = optim.LBFGS(params, max_iter=200, history_size=50,
                             line_search_fn="strong_wolfe",
                             tolerance_change=1e-12, tolerance_grad=1e-10)
        def closure():
            lbfgs.zero_grad()
            lp_l, ln_l, ld_l = self.compute_losses(epochs)
            (lp_l + self.lambda_bc*ln_l + ld_l).backward()
            return lp_l + self.lambda_bc*ln_l + ld_l
        lbfgs.step(closure)
        with torch.enable_grad():
            post_loss = sum(self.compute_losses(epochs)[:2]).item()
        if post_loss > pre_loss * 2.0:
            for p, w in zip(params, pre_weights): p.data.copy_(w)

    def report_l2(self, exact_func, eval_times=None):
        if eval_times is None: eval_times = [1.0]
        errors = []
        for i, m in self.models.items():
            for t_desired in eval_times:
                t_t = torch.full((100,1), t_desired).to(device)
                x_t = torch.linspace(0, m['L'], 100).view(-1,1).to(device)
                u_p = self.predict(i, x_t, t_t).detach().cpu().numpy()
                u_e = exact_func(x_t.cpu().numpy(), t_desired)
                ne  = np.linalg.norm(u_e)
                if ne < 1e-10:
                    print(f"t={t_desired:.4f}: skipped (zero exact)"); continue
                err = np.linalg.norm(u_p - u_e) / ne
                errors.append((t_desired, err))
                print(f"t={t_desired:.4f}: {err:.4e}")
        if errors: print(f"Mean: {np.mean([e for _,e in errors]):.4e}")
        return [e for _,e in errors]

    def report_inverse(self, true_values: dict):
        print("\n" + "=" * 50)
        print("  Inverse parameter recovery summary")
        print(f"  {'Parameter':<14} {'True':>10} {'Recovered':>12} {'Rel err':>10}")
        print("-" * 50)
        for name, param in self._inverse_params:
            rec  = param.item()
            true = true_values.get(name, float('nan'))
            rel  = abs(rec - true) / (abs(true) + 1e-16) * 100
            print(f"  {name:<14} {true:>10.6f} {rec:>12.6f} {rel:>9.3f}%")
        print("=" * 50 + "\n")

    def plot_results(self, exact_func, t_vals=(0.25, 0.5, 0.75, 1.0)):
        fig, axs = plt.subplots(1, len(t_vals), figsize=(4*len(t_vals), 4))
        if len(t_vals) == 1: axs = [axs]
        for idx, t_val in enumerate(t_vals):
            ax = axs[idx]
            for i, m in self.models.items():
                x_plot = np.linspace(0, m['L'], 100)
                x_t    = torch.tensor(x_plot).view(-1,1).to(device)
                t_t    = torch.full_like(x_t, t_val)
                u_p    = self.predict(i, x_t, t_t).detach().cpu().numpy()
                u_e    = exact_func(x_plot, t_val)
                ax.plot(x_plot, u_e, 'k--', label="Exact"            if i==0 else "")
                ax.plot(x_plot, u_p, 'r-',  label=f"PINN (Edge {i})" if i==0 else "")
            ax.set_title(f"t = {t_val}"); ax.set_xlabel("x")
            if idx == 0: ax.set_ylabel("u(x,t)")
            ax.legend()
        plt.tight_layout(); plt.show()

    def plot_inverse_history(self, true_values: dict = None):
        n = len(self._inv_param_histories)
        if n == 0:
            print("No inverse parameters registered."); return
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
        if n == 1: axes = [axes]
        for ax, (name, hist) in zip(axes, self._inv_param_histories.items()):
            ax.plot(hist, lw=1.5, color='darkorange', label=f'{name} estimate')
            if true_values and name in true_values:
                ax.axhline(true_values[name], ls='--', lw=1.5, color='k',
                           label=f'true = {true_values[name]}')
            ax.set_xlabel('Optimisation step')
            ax.set_ylabel(name)
            ax.set_title(f'{name} convergence')
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("inverse_param_history.png", dpi=150, bbox_inches="tight")
        plt.show()
