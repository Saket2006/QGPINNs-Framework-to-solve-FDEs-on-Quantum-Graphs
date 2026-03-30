
"""Complete Engine"""

#%%writefile PINN_Solver.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import gamma
import math
import copy
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import solve_banded


DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps") #Note, for MPS, we require float32, performence unknown
else:
    device = torch.device("cpu")

#This class defines the Fourier Features that the framework uses.
class FourierEmbedding(nn.Module):
    def __init__(self, in_dim=2, embed_dim=64, sigma=1.0, sampling="sobol"):
        super().__init__()
        n_freqs = embed_dim // 2
        if sampling == "sobol":
            sobol = torch.quasirandom.SobolEngine(dimension=in_dim, scramble=True)
            u = sobol.draw(n_freqs).to(DTYPE)
            u = torch.clamp(u, 1e-6, 1 - 1e-6)
            B = torch.erfinv(2 * u - 1) * (2 ** 0.5) * sigma
            B = B.T
        else:
            B = torch.randn(in_dim, n_freqs, dtype=DTYPE) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

#Builds the actual Neural Network
class PINN_Net(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, hidden_layers=3,
                 use_fourier=False, fourier_dim=64, fourier_sigma=1.0,
                 fourier_sampling="sobol"):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierEmbedding(in_dim, fourier_dim,
                                            fourier_sigma, fourier_sampling)
            first_in = fourier_dim
        else:
            self.fourier = None
            first_in = in_dim
        layers = [nn.Linear(first_in, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight,
                                        gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_fourier:
            x = self.fourier(x)
        return self.net(x)


#The matrices for L1 and L21 are formulated here

def _compute_l1_matrix_vectorized(alpha, grid_np):
    n = len(grid_np)
    if alpha <= 0:
        return torch.zeros(n, n, dtype=DTYPE).to(device)
    g1   = gamma(2 - alpha)
    h    = np.diff(grid_np)
    I, J = np.tril_indices(n, k=-1)
    tau_ij  = grid_np[I] - grid_np[J]
    tau_ij1 = grid_np[I] - grid_np[J + 1]
    w = (tau_ij ** (1 - alpha) - tau_ij1 ** (1 - alpha)) / (h[J] * g1)
    mat = np.zeros((n, n))
    np.add.at(mat, (I, J + 1),  w)
    np.add.at(mat, (I, J),     -w)
    return torch.tensor(mat, dtype=DTYPE).to(device)


def _compute_l21sigma_matrix(alpha, grid_np, sigma):
    n   = len(grid_np)
    g2a = gamma(2 - alpha)
    tau = np.diff(grid_np)
    W   = np.zeros((n, n))
    for row in range(1, n):
        t_nsig = (1.0 - sigma) * grid_np[row - 1] + sigma * grid_np[row]
        for k in range(row):
            left    = t_nsig - grid_np[k]
            right   = t_nsig - grid_np[k + 1]
            if left <= 0:
                continue
            right_c = max(right, 0.0)
            A_k = (left ** (1 - alpha) - right_c ** (1 - alpha)) / (tau[k] * g2a)
            W[row, k + 1] += A_k
            W[row, k]     -= A_k
    return torch.tensor(W, dtype=DTYPE).to(device)

#As in paramter estimation of alpha, alpha itself is not fixed, we use the following implementation, will add L21
def _compute_l1_matrix_torch(alpha, grid_t):
    n = grid_t.shape[0]
    if n <= 1:
        return torch.zeros(n, n, dtype=DTYPE, device=grid_t.device)
    h = grid_t[1:] - grid_t[:-1]
    rows, cols = torch.tril_indices(n, n, offset=-1, device=grid_t.device)
    tau_ij  = grid_t[rows] - grid_t[cols]
    tau_ij1 = grid_t[rows] - grid_t[cols + 1]
    eps = torch.tensor(1e-14, dtype=DTYPE, device=grid_t.device)
    p   = 1.0 - alpha
    g1  = torch.exp(torch.lgamma(2.0 - alpha))
    w   = (
        (torch.clamp(tau_ij, min=eps) ** p - torch.clamp(tau_ij1, min=eps) ** p)
        / (torch.clamp(h[cols], min=eps) * g1)
    )
    mat = torch.zeros((n, n), dtype=DTYPE, device=grid_t.device)
    mat.index_put_((rows, cols + 1), w, accumulate=True)
    mat.index_put_((rows, cols), -w, accumulate=True)
    return mat




class _PINNSolverBase:

    def __init__(self, graph, physics):
        self.graph = graph
        if isinstance(physics, list):
            self._physics_list = physics
            self.physics       = physics[0]
        else:
            self._physics_list = None
            self.physics       = physics

        self.compiled    = False
        self.lambda_bc   = 1.0
        self.models      = {}
        self.current_epoch = 0

        #architecture
        self.hidden_layers    = 3
        self.hidden_dim       = 64
        self.use_fourier      = False
        self.fourier_dim      = 64
        self.fourier_sigma    = 1.0
        self.fourier_sampling = "sobol"

        #mesh
        self.pts_per_unit   = 250
        self.grading_factor = 1.5
        self.anchor_pts     = 0

        #constraints
        self.constraint_mode = "soft"
        self.bc_types        = {}
        self.bc_values       = {}

        #optimiser
        self.lr               = 5e-4
        self.scheduler_min_lr = 1e-6

        #validation
        self.validation_func = None

        #inverse problem
        self.inverse_enabled         = False
        self.inverse_parameter_names = []
        self.inverse_param_bounds    = {}
        self.inverse_include_alpha   = False
        self.inverse_alpha_bounds    = (1.05, 1.95)
        self.lambda_data             = 1.0
        self.inverse_data            = {}
        self.last_data_loss          = 0.0

#Sets the architecture based on class inputs
    def set_architecture(self, hidden_layers=3, hidden_dim=64,
                         use_fourier=False, fourier_dim=64,
                         fourier_sigma=1.0, fourier_sampling="sobol"):
        self.hidden_layers    = hidden_layers
        self.hidden_dim       = hidden_dim
        self.use_fourier      = use_fourier
        self.fourier_dim      = fourier_dim
        self.fourier_sigma    = fourier_sigma
        self.fourier_sampling = fourier_sampling
        return self

#Sets the constraints based on the class inputs
    def set_constraints(self, constraint_mode="soft",
                        bc_types=None, bc_values=None):
        self.constraint_mode = constraint_mode.lower()
        self.bc_types  = bc_types  or {}
        self.bc_values = bc_values or {}
        return self

#Validates inputs
    def set_validation(self, func, **kwargs):
        self.validation_func = func
        return self

#Sets inverse problem bounds
    def set_inverse_problem(self, parameter_names=None, include_alpha=False,
                            param_bounds=None, alpha_bounds=(1.05, 1.95),
                            data_weight=1.0):
        self.inverse_enabled         = True
        self.inverse_parameter_names = list(parameter_names or [])
        self.inverse_param_bounds    = dict(param_bounds or {})
        self.inverse_include_alpha   = bool(include_alpha)
        self.inverse_alpha_bounds    = tuple(alpha_bounds)
        self.lambda_data             = float(data_weight)
        return self

#Saftey
    @staticmethod
    def _to_bounded(raw, bounds):
        if bounds is None:
            return raw
        lo, hi = float(bounds[0]), float(bounds[1])
        return lo + (hi - lo) * torch.sigmoid(raw)

    @staticmethod
    def _raw_from_value(value, bounds):
        if bounds is None:
            return torch.tensor(float(value), dtype=DTYPE, device=device)
        lo, hi = float(bounds[0]), float(bounds[1])
        v = float(np.clip((value - lo) / (hi - lo), 1e-6, 1.0 - 1e-6))
        return torch.tensor(math.log(v / (1.0 - v)), dtype=DTYPE, device=device)

    def _edge_param_value(self, edge_idx, name):
        m = self.models[edge_idx]
        if 'inv_params' not in m or name not in m['inv_params']:
            return None
        cfg = m['inv_params'][name]
        return self._to_bounded(cfg['raw'], cfg['bounds'])

    def _refresh_edge_physics_parameters(self, edge_idx):
        m = self.models[edge_idx]
        if 'inv_params' not in m:
            return
        for pname, cfg in m['inv_params'].items():
            setattr(m['physics'], pname, self._to_bounded(cfg['raw'], cfg['bounds']))

    def get_estimated_parameters(self):
        out = {}
        for i, m in self.models.items():
            if 'inv_params' not in m:
                continue
            out[i] = {}
            for name, cfg in m['inv_params'].items():
                val = float(self._to_bounded(cfg['raw'], cfg['bounds'])
                            .detach().cpu().item())
                out[i][name] = val
                setattr(m['physics'], name, val)
        return out


    def _all_params(self):
        p = [p for m in self.models.values() for p in m['net'].parameters()]
        if getattr(self, 'use_singularity_capture', False):
            p = p + [self.xi_raw]
        if self.inverse_enabled:
            for m in self.models.values():
                if 'inv_params' in m:
                    p.extend([cfg['raw'] for cfg in m['inv_params'].values()])
        return p

    def _reinit_weights(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        for m in self.models.values():
            for layer in m['net'].net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight,
                                            gain=nn.init.calculate_gain('tanh'))
                    nn.init.zeros_(layer.bias)
        self.lambda_bc = 1.0
        if getattr(self, 'use_singularity_capture', False):
            init_val = getattr(self, '_xi_init_val', 1.0)
            with torch.no_grad():
                self.xi_raw.fill_(np.log(init_val))
            if hasattr(self, '_xi_lr_base'):
                del self._xi_lr_base

#The different lambda strategies are defined
    def _update_bdmm(self, ln):
        if ln.item() == 0:
            return
        self.lambda_bc = min(self.lambda_bc + 0.5 * ln.item(), 1000.0)

    def _update_gradient_ratio(self, lp, ln, params):
        if ln.item() == 0:
            return
        gp = torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)
        gn = torch.autograd.grad(ln, params, retain_graph=True, allow_unused=True)
        np_ = torch.cat([g.view(-1) for g in gp if g is not None]).norm()
        nn_ = torch.cat([g.view(-1) for g in gn if g is not None]).norm()
        if nn_ > 1e-8:
            self.lambda_bc = float(np.clip(
                0.9 * self.lambda_bc + 0.1 * (np_ / (nn_ + 1e-8)).item(),
                0.01, 1000.0))

    def _run_lbfgs(self, params):
        pre_w = [p.data.clone() for p in params]
        with torch.enable_grad():
            lp0, ln0 = self.compute_losses()[:2]
            pre_loss = lp0.item() + ln0.item()

        lbfgs = optim.LBFGS(params, max_iter=200, history_size=50,
                             line_search_fn="strong_wolfe",
                             tolerance_change=1e-12, tolerance_grad=1e-10)

        def closure():
            lbfgs.zero_grad()
            lp_l, ln_l = self.compute_losses()[:2]
            loss_l = lp_l + self.lambda_bc * ln_l
            loss_l.backward()
            return loss_l

        lbfgs.step(closure)

        with torch.enable_grad():
            lp1, ln1 = self.compute_losses()[:2]
            post_loss = lp1.item() + ln1.item()

        if post_loss > pre_loss * 2.0:
            for p, w in zip(params, pre_w):
                p.data.copy_(w)

#Actual traning loop
    def _train_loop(self, epochs, strategy, use_lbfgs,
                    optimizer, scheduler, epoch_offset=0, global_best=None):
        params  = self._all_params()
        use_val = self.validation_func is not None
        best_l, best_w, best_lam = float('inf'), None, self.lambda_bc
        total   = epoch_offset + epochs
#using xi for para only, will experiment with expanding to ellipitc
        use_xi = getattr(self, 'use_singularity_capture', False)

        for le in range(epochs + 1):
            ep = epoch_offset + le
            self.current_epoch = ep


            if getattr(self, 'use_causal', False):
                if not hasattr(self, '_causal_eps_start'):
                    self._causal_eps_start = self.causal_eps
                es, ee = self._causal_eps_start, self.causal_eps_end
                if es > ee > 0:
                    self.causal_eps = ee + (es - ee) * math.exp(
                        -math.log(es / ee) * ep / max(total, 1))


            if strategy == "dual":
                active = "bdmm" if ep < total // 2 else "gradient_ratio"
            else:
                active = strategy

            if (use_xi and strategy == "dual" and ep == total // 2):
                xi_lr_p2 = self.lr * getattr(self, 'xi_lr_scale_phase2', 0.01)
                for g in optimizer.param_groups:
                    if any(p is self.xi_raw for p in g['params']):
                        g['lr'] = xi_lr_p2
                        break

            optimizer.zero_grad()
            lp, ln = self.compute_losses(ep)[:2]

            if ep > 0 and ep % 10 == 0:
                if active == "bdmm":
                    self._update_bdmm(ln)
                else:
                    self._update_gradient_ratio(lp, ln, params)

            loss   = lp + self.lambda_bc * ln
            metric = self._eval_l2() if use_val else lp.item() + ln.item()

            if metric < best_l:
                best_l   = metric
                best_w   = [p.data.clone() for p in params]
                best_lam = self.lambda_bc

            if global_best is not None and metric < global_best['loss']:
                global_best.update(loss=metric,
                                   params=[p.data.clone() for p in params],
                                   lam=self.lambda_bc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            if (use_xi and getattr(self, 'xi_loss_adaptive', True)
                    and hasattr(self, '_xi_lr_base')):
                ref   = getattr(self, 'xi_loss_ref',   1e-1)
                floor = getattr(self, 'xi_loss_floor', 1e-5)
                scale = float(np.clip(lp.item() / ref, floor / ref, 1.0))
                for g in optimizer.param_groups:
                    if any(p is self.xi_raw for p in g['params']):
                        g['lr'] = self._xi_lr_base * scale
                        break

            optimizer.step()
            scheduler.step()

            if le % 500 == 0:
                xi_s   = (f"  xi={torch.exp(self.xi_raw).item():.4f}"
                          if use_xi else "")
                data_s = (f" | Data={self.last_data_loss:.3e}"
                          if self.inverse_enabled and self.inverse_data else "")
                print(f"Epoch {le:6d} | loss={loss.item():.3e} |"
                      f" PDE={lp.item():.3e} | BC={ln.item():.3e} |"
                      f" λ={self.lambda_bc:.3f}{xi_s}{data_s}")

#restore best checkpoint before L-BFGS
        if best_w is not None:
            for p, w in zip(params, best_w):
                p.data.copy_(w)
            self.lambda_bc = best_lam

        if use_lbfgs:
            self._run_lbfgs(params)

        # post-LBFGS global best update
        if global_best is not None:
            m2 = self._eval_l2() if use_val else sum(self.compute_losses()[:2]).item()
            if m2 < global_best['loss']:
                global_best.update(loss=m2,
                                   params=[p.data.clone() for p in params],
                                   lam=self.lambda_bc)
        return best_l


    def train(self, epochs=4000, strategy="dual", use_lbfgs=True):
        self._planned_epochs = epochs
        if not self.compiled:
            self.compile()
        params = self._all_params()
        opt    = self._make_optimizer(params)
        sch    = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs + 1, eta_min=self.scheduler_min_lr)
        return self._train_loop(epochs, strategy, use_lbfgs, opt, sch)

#Multiple seed training to combat variations
    def train_multistart(self, epochs=4000, strategy="dual", use_lbfgs=True,
                         n_starts=3, seed_base=0, probe_epochs=500):
        if not self.compiled:
            self.compile()

        params      = self._all_params()
        global_best = dict(loss=float('inf'), params=None, lam=1.0)
        bp_score, bp_params, bp_lam = float('inf'), None, 1.0

        for s in range(n_starts):
            seed = seed_base * 100 + s * 37
            self._reinit_weights(seed=seed)
            opt = self._make_optimizer(params)
            sch = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=probe_epochs + 1, eta_min=self.scheduler_min_lr)
            self._train_loop(probe_epochs, strategy, False, opt, sch,
                             epoch_offset=0, global_best=global_best)

            score = self._eval_l2() if self.validation_func else (
                sum(self.compute_losses()[:2]).item())
            lbl = "L2" if self.validation_func else "loss"

            xi_s = (f"  xi={torch.exp(self.xi_raw).item():.4f}"
                    if getattr(self, 'use_singularity_capture', False) else "")
            print(f"  probe {s+1}/{n_starts}  seed={seed}  {lbl}={score:.3e}{xi_s}")

            if score < bp_score:
                bp_score  = score
                bp_params = [p.data.clone() for p in params]
                bp_lam    = self.lambda_bc

#restore best probe
        if bp_params is not None:
            for p, w in zip(params, bp_params):
                p.data.copy_(w)
        self.lambda_bc = bp_lam
        print(f"  -> continuing best probe (score={bp_score:.3e}) for {epochs} epochs")

        opt = self._make_optimizer(params)
        sch = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs + 1, eta_min=self.scheduler_min_lr)
        self._train_loop(epochs, strategy, use_lbfgs, opt, sch,
                         epoch_offset=probe_epochs, global_best=global_best)

        if global_best['params'] is not None:
            for p, w in zip(params, global_best['params']):
                p.data.copy_(w)
            self.lambda_bc = global_best['lam']
            print(f"  -> global best L2 = {global_best['loss']:.3e}")
        return self

    def train_inverse(self, epochs=4000, strategy="dual", use_lbfgs=True):
        if not self.inverse_enabled:
            raise RuntimeError("Call set_inverse_problem(...) before train_inverse(...).")
        if not self.inverse_data:
            raise RuntimeError("Call generate_noisy_edge_data(...) before train_inverse(...).")
        return self.train(epochs=epochs, strategy=strategy, use_lbfgs=use_lbfgs)

#Error calculation

    def report_l2(self, exact_func, n_pts=200):
        print("\n--- L2 Error Report ---")
        errors = []
        with torch.no_grad():
            for i, m in self.models.items():
                vf    = exact_func[i] if isinstance(exact_func, list) else exact_func
                x_np  = np.linspace(0, m['L'], n_pts)
                x_t   = torch.tensor(x_np, dtype=DTYPE).view(-1, 1).to(device)
                u_p   = self._predict_np(i, x_t)
                u_e   = self._exact_np(vf, x_np, m['L'])
                norm_e = np.linalg.norm(u_e)
                if norm_e < 1e-10:
                    print(f"  Edge {i} {m['nodes']}: exact norm ~0, skipped")
                    continue
                rel = np.linalg.norm(u_p - u_e) / norm_e
                errors.append(rel)
                print(f"  Edge {i} {m['nodes']}: Rel L2 = {rel:.4e}")
        if errors:
            print(f"  Mean: {np.mean(errors):.4e}")
        return errors

#plotting
    def plot_results(self, exact_func=None, n_pts=200):
        n = len(self.models)
        fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n))
        if n == 1:
            axes = [axes]
        for i, m in self.models.items():
            x_np = np.linspace(0, m['L'], n_pts)
            x_t  = torch.tensor(x_np, dtype=DTYPE).view(-1, 1).to(device)
            u_p  = self._predict_np(i, x_t)
            axes[i].plot(x_np, u_p, 'r-', label='PINN')
            if exact_func is not None:
                ef  = exact_func[i] if isinstance(exact_func, list) else exact_func
                u_e = self._exact_np(ef, x_np, m['L'])
                axes[i].plot(x_np, u_e, 'k--', alpha=0.7, label='Exact')
            if getattr(self, 'use_anchors', False) and self.anchor_X and i in self.anchor_X:
                axes[i].scatter(self.anchor_X[i].cpu().numpy().flatten(),
                                self.anchor_U[i].cpu().numpy().flatten(),
                                s=8, c='blue', alpha=0.4, label='FD anchors')
            axes[i].set_title(f"Edge {i}  nodes={m['nodes']}")
            axes[i].set_xlabel("x"); axes[i].set_ylabel("u(x)")
            axes[i].legend(); axes[i].grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    def post_process(self, u_exact_func=None):  # backward-compat alias
        self.plot_results(exact_func=u_exact_func)

    def plot_graph_topology(self):
        G = nx.Graph()
        for u, v, L in self.graph.edges:
            G.add_edge(u, v, weight=L)
        plt.figure(figsize=(5, 3))
        nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
        plt.title("Metric Graph Topology"); plt.show()



    def _make_optimizer(self, params):
        """Override in ParabolicPINNSolver for xi separate lr."""
        return optim.Adam(params, lr=self.lr)

    def _predict_np(self, idx, x_t):
        """Call predict() and return flat numpy — handles (x,) and (x,t) signatures."""
        raise NotImplementedError

    def _exact_np(self, vf, x_np, L):
        """Call validation function and return flat numpy."""
        raise NotImplementedError

    # ── abstract interface ────────────────────────────────────────────────────

    def compile(self):
        raise NotImplementedError

    def predict(self, idx, *args):
        raise NotImplementedError

    def compute_losses(self, epoch=0):
        """Must return at least (lp, ln); extra terms ignored by base."""
        raise NotImplementedError

    def _eval_l2(self):
        raise NotImplementedError


# ══ Parabolic solver ══════════════════════════════════════════════════════════

class ParabolicPINNSolver(_PINNSolverBase):

    def __init__(self, graph, physics):
        super().__init__(graph, physics)


        self.mesh_type      = "power_law"
        self.pinn_pts       = 100
        self.n_t            = 100
        self.t_max          = 1.0


        self.frac_scheme = "L1"
        self.l21_sigma   = 0.5
        self.t_nsig_grid = None


        self.use_causal      = False
        self.causal_eps      = 1.0
        self.causal_eps_end  = 1e-4
        self.causal_bc_floor = 0.1


        self.use_singularity_capture = False
        self.xi_raw           = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.xi_loss_adaptive = True
        self.xi_loss_ref      = 1e-1
        self.xi_loss_floor    = 1e-5


        self.val_pts_per_unit = 1000


        self.lr               = 5e-4
        self.scheduler_min_lr = 1e-5
        self.lambda_bc        = 1.0



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
            alpha = getattr(self.physics, 'alpha', 0.5)
            self.l21_sigma = (1.0 - alpha / 2.0) if sigma is None else float(sigma)
        return self

    def set_causal_training(self, enabled=True, eps=1.0, eps_end=1e-4, bc_floor=0.1):
        self.use_causal      = enabled
        self.causal_eps      = eps
        self.causal_eps_end  = eps_end
        self.causal_bc_floor = bc_floor
        return self

    def set_singularity_capture(self, enabled=True, xi_init=None,
                                xi_loss_adaptive=True,
                                xi_loss_ref=1e-1, xi_loss_floor=1e-5):
        self.use_singularity_capture = enabled
        self.xi_loss_adaptive        = xi_loss_adaptive
        self.xi_loss_ref             = float(xi_loss_ref)
        self.xi_loss_floor           = float(xi_loss_floor)
        if enabled:
            init = float(xi_init) if xi_init is not None else 1.0
            self._xi_init_val = init
            self.xi_raw = nn.Parameter(
                torch.tensor(np.log(init), dtype=DTYPE))
        return self

    def set_lr(self, lr=5e-4, min_lr=1e-5, xi_lr_scale=0.1, xi_lr_scale_phase2=0.01):
        self.lr                 = lr
        self.scheduler_min_lr   = min_lr
        self.xi_lr_scale        = xi_lr_scale
        self.xi_lr_scale_phase2 = xi_lr_scale_phase2
        return self

    def set_validation(self, func, times=None, pts_per_unit=1000):
        self.validation_func  = func
        self.val_pts_per_unit = pts_per_unit
        return self

    def generate_noisy_edge_data(self, exact_func, n_points_per_edge=40,
                                 noise_std=0.01, seed=0):
        if not self.compiled:
            self.compile()
        rng = np.random.default_rng(seed)
        self.inverse_data = {}
        for i, m in self.models.items():
            n    = max(int(n_points_per_edge), 4)
            x_np = rng.uniform(0.0, m['L'], size=n)
            t_np = rng.uniform(0.0, self.t_max, size=n)
            if isinstance(exact_func, list):
                clean = exact_func[i](x_np, t_np).reshape(-1)
            else:
                try:
                    clean = exact_func(x_np, t_np, i).reshape(-1)
                except TypeError:
                    clean = exact_func(x_np, t_np).reshape(-1)
            noisy = clean + rng.normal(0.0, float(noise_std), size=n)
            self.inverse_data[i] = dict(
                x=torch.tensor(x_np, dtype=DTYPE, device=device).view(-1, 1),
                t=torch.tensor(t_np, dtype=DTYPE, device=device).view(-1, 1),
                u_obs=torch.tensor(noisy, dtype=DTYPE, device=device).view(-1, 1),
                noise_std=float(noise_std),
            )
        return self.inverse_data



    def _generate_mesh(self, max_val, n_pts, is_spatial=False):
        if is_spatial:
            return np.linspace(0, max_val, n_pts)
        if self.mesh_type == "power_law":
            idx = np.arange(n_pts)
            return max_val * (idx / (n_pts - 1)) ** self.grading_factor
        return np.linspace(0, max_val, n_pts)

    def compile(self):
        if self.use_singularity_capture:
            self.xi_raw = nn.Parameter(self.xi_raw.data.to(device))
        else:
            self.xi_raw = self.xi_raw.to(device)

        t_np        = self._generate_mesh(self.t_max, self.n_t)
        self.t_grid = torch.tensor(t_np, dtype=DTYPE).view(-1, 1).to(device)

        alpha = getattr(self.physics, 'alpha', None)
        if alpha is not None:
            if self.frac_scheme == "L21sigma":
                if not hasattr(self, 'l21_sigma') or self.l21_sigma is None:
                    self.l21_sigma = 1.0 - alpha / 2.0
                self.Dt_alpha = _compute_l21sigma_matrix(alpha, t_np, self.l21_sigma)
                t_nsig_np     = np.empty(self.n_t)
                t_nsig_np[0]  = t_np[0]
                t_nsig_np[1:] = ((1.0 - self.l21_sigma) * t_np[:-1]
                                 + self.l21_sigma * t_np[1:])
                self.t_nsig_grid = torch.tensor(t_nsig_np, dtype=DTYPE).view(-1, 1).to(device)
            else:
                self.Dt_alpha    = _compute_l1_matrix_vectorized(alpha, t_np)
                self.t_nsig_grid = self.t_grid

            tau_np = np.diff(t_np)
            print(f"  [mesh] r={self.grading_factor:.2f}"
                  f"  scheme={self.frac_scheme}"
                  f"  tau_min={tau_np.min():.3e}"
                  f"  tau_max={tau_np.max():.3e}"
                  f"  ratio={tau_np.max()/tau_np.min():.1f}"
                  f"  Z(t)={self.use_singularity_capture}")

        net_in_dim = 3 if self.use_singularity_capture else 2

        for i, edge in enumerate(self.graph.edges):
            u_node, v_node, L = edge[0], edge[1], edge[2]
            n_x    = max(int(self.pinn_pts * L), 2)
            x_np   = self._generate_mesh(L, n_x, is_spatial=True)
            x_grid = torch.tensor(x_np, dtype=DTYPE).view(-1, 1).to(device)
            net    = PINN_Net(
                in_dim=net_in_dim, hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers, use_fourier=self.use_fourier,
                fourier_dim=self.fourier_dim, fourier_sigma=self.fourier_sigma,
                fourier_sampling=self.fourier_sampling,
            ).to(device).to(DTYPE)

            T_mesh      = self.t_grid.repeat_interleave(n_x, dim=0)
            T_mesh_nsig = self.t_nsig_grid.repeat_interleave(n_x, dim=0)
            phys_i      = self._physics_list[i] if self._physics_list else self.physics
            f_target    = phys_i.get_f_target(
                x_grid.repeat(self.n_t, 1), T_mesh_nsig, L
            ).view(-1, 1).to(device)

            self.models[i] = dict(
                net=net, L=L, n_x=n_x, nodes=(u_node, v_node),
                x_grid=x_grid,
                X_mesh=x_grid.repeat(self.n_t, 1),
                T_mesh=T_mesh,
                T_mesh_nsig=T_mesh_nsig,
                f_target=f_target,
                physics=phys_i,
            )

            if self.inverse_enabled:
                if self.inverse_include_alpha and self.frac_scheme != "L1":
                    raise NotImplementedError(
                        "Inverse alpha estimation requires frac_scheme='L1'.")
                inv_params = {}
                for pname in self.inverse_parameter_names:
                    if not hasattr(phys_i, pname):
                        raise AttributeError(
                            f"Physics on edge {i} has no attribute '{pname}'.")
                    bnd = self.inverse_param_bounds.get(pname, None)
                    inv_params[pname] = {
                        "raw": nn.Parameter(
                            self._raw_from_value(float(getattr(phys_i, pname)), bnd).to(device)),
                        "bounds": bnd,
                    }
                if self.inverse_include_alpha:
                    ab = self.inverse_alpha_bounds
                    inv_params['alpha'] = {
                        "raw": nn.Parameter(
                            self._raw_from_value(float(phys_i.alpha), ab).to(device)),
                        "bounds": ab,
                    }
                if inv_params:
                    self.models[i]['inv_params'] = inv_params

        self.compiled = True
        return self


    def predict(self, idx, x, t):
        m = self.models[idx]
        if self.use_singularity_capture:
            xi  = torch.exp(self.xi_raw)
            z   = t.clamp(min=1e-30).pow(xi)
            inp = torch.cat([x, t, z], dim=-1)
        else:
            inp = torch.cat([x, t], dim=-1)
        u_raw = m['net'](inp)

        if self.constraint_mode == "hard":
            _deg = {}
            for edge in self.graph.edges:
                _deg[edge[0]] = _deg.get(edge[0], 0) + 1
                _deg[edge[1]] = _deg.get(edge[1], 0) + 1
            u_node, v_node = m['nodes']
            tl = (self.bc_types.get(u_node, "dirichlet")
                  if _deg.get(u_node, 1) == 1 else "junction")
            tr = (self.bc_types.get(v_node, "dirichlet")
                  if _deg.get(v_node, 1) == 1 else "junction")
            if tl == "dirichlet" and tr == "dirichlet":
                vl, vr = self._get_bc_value(u_node, t), self._get_bc_value(v_node, t)
                g_x    = vl + (x / m['L']) * (vr - vl)
                dist_x = x * (m['L'] - x)
                phys_i = m['physics']
                t0     = torch.zeros_like(t)
                compat = (
                    torch.max(torch.abs(self._get_bc_value(u_node, t0)
                                        - phys_i.get_ic(torch.zeros_like(x)))).item() < 1e-6
                    and
                    torch.max(torch.abs(self._get_bc_value(v_node, t0)
                                        - phys_i.get_ic(torch.ones_like(x) * m['L']))).item() < 1e-6
                )
                if compat:
                    return g_x + dist_x * u_raw
                w_ic = torch.exp(-20.0 * t)
                return w_ic * phys_i.get_ic(x) + (1.0 - w_ic) * (g_x + dist_x * u_raw)
        return u_raw

    def _get_bc_value(self, node_idx, t):
        bc = self.bc_values.get(node_idx, 0.0)
        return bc(t) if callable(bc) else bc


    def compute_losses(self, epoch=0):
        lp, ln    = 0, 0
        _causal_w = None

        for i, m in self.models.items():
            if self.inverse_enabled and 'inv_params' in m:
                self._refresh_edge_physics_parameters(i)
            n_x_m = m['n_x']

            u_grid     = self.predict(i, m['X_mesh'], m['T_mesh'])
            u_reshaped = u_grid.view(self.n_t, n_x_m)
            if self.inverse_enabled and 'inv_params' in m and 'alpha' in m['inv_params']:
                alpha_i    = self._edge_param_value(i, 'alpha')
                Dt_i       = _compute_l1_matrix_torch(alpha_i, self.t_grid.view(-1))
                dt_alpha_u = torch.mm(Dt_i, u_reshaped).view(-1, 1)
            else:
                dt_alpha_u = torch.mm(self.Dt_alpha, u_reshaped).view(-1, 1)

            X      = m['X_mesh'].clone().detach().requires_grad_(True)
            T_eval = m['T_mesh_nsig'].clone().detach().requires_grad_(True)
            u_sp   = self.predict(i, X, T_eval)
            u_x    = torch.autograd.grad(u_sp, X, torch.ones_like(u_sp),
                                         create_graph=True)[0]
            u_xx   = torch.autograd.grad(u_x, X, torch.ones_like(u_x),
                                         create_graph=True)[0]

            f_target = m['f_target']
            res = m['physics'].F(X, T_eval, u_sp, u_x, u_xx, dt_alpha_u, f_target)

            if self.use_causal:
                res_t  = res.view(self.n_t, n_x_m)
                loss_t = res_t[1:].pow(2).mean(dim=1)
                s      = torch.linspace(0, 1, self.n_t - 1, dtype=DTYPE, device=device)
                weights = torch.exp(-self.causal_eps * s).detach()
                lp     += (weights * loss_t).mean()
                _causal_w = weights.clamp(min=self.causal_bc_floor)
            else:
                lp += res.view(self.n_t, n_x_m)[1:].pow(2).mean()

        _bc_w = _causal_w

        _deg = {}
        for edge in self.graph.edges:
            _deg[edge[0]] = _deg.get(edge[0], 0) + 1
            _deg[edge[1]] = _deg.get(edge[1], 0) + 1

        #IC
        for i, m in self.models.items():
            if self.constraint_mode == "soft":
                X_ic = m['x_grid']
                u_ic = self.predict(i, X_ic, torch.zeros_like(X_ic))
                ln  += torch.mean((u_ic - m['physics'].get_ic(X_ic)) ** 2)

        #leaf BCs
        for i, m in self.models.items():
            for node_idx, x_val in [(m['nodes'][0], 0.0), (m['nodes'][1], m['L'])]:
                if _deg.get(node_idx, 1) > 1:
                    continue
                bc_type = self.bc_types.get(node_idx, "dirichlet")
                if self.constraint_mode == "hard" and bc_type == "dirichlet":
                    continue
                T_bc = self.t_grid.clone().detach().requires_grad_(True)
                X_bc = torch.full_like(T_bc, x_val).requires_grad_(True)
                u_bc = self.predict(i, X_bc, T_bc)
                if bc_type == "dirichlet":
                    res_bc = (u_bc - self._get_bc_value(node_idx, T_bc)).pow(2).view(-1)
                    ln += ((res_bc[1:] * _bc_w).mean() + res_bc[0]
                           if _bc_w is not None else res_bc.mean())
                else:
                    u_x_bc  = torch.autograd.grad(u_bc, X_bc,
                                  torch.ones_like(u_bc), create_graph=True)[0]
                    res_nbc = (u_x_bc - self._get_bc_value(node_idx, T_bc)).pow(2).view(-1)
                    ln += ((res_nbc[1:] * _bc_w).mean() + res_nbc[0]
                           if _bc_w is not None else res_nbc.mean())

        #junction: continuity + Kirchhoff
        for node_idx, d in _deg.items():
            if d <= 1:
                continue
            inc = []
            for i, m in self.models.items():
                if node_idx == m['nodes'][0]: inc.append((i, 0.0,    1.0))
                if node_idx == m['nodes'][1]: inc.append((i, m['L'], -1.0))
            if len(inc) < 2:
                continue
            T_jc  = self.t_grid.clone().detach()
            u_ref = None
            flux  = torch.zeros_like(T_jc)
            for idx_e, xv, s in inc:
                X_j = torch.full_like(T_jc, xv).requires_grad_(True)
                T_j = T_jc.clone().requires_grad_(True)
                u_j = self.predict(idx_e, X_j, T_j)
                if u_ref is None:
                    u_ref = u_j.detach()
                else:
                    ct = (u_j - u_ref).pow(2).view(-1)
                    ln += ((ct[1:] * _bc_w).mean() + ct[0]
                           if _bc_w is not None else ct.mean())
                u_x_j = torch.autograd.grad(u_j, X_j,
                            torch.ones_like(u_j), create_graph=True)[0]
                flux = flux + u_x_j * s
            ft = flux.pow(2).view(-1)
            ln += ((ft[1:] * _bc_w).mean() + ft[0]
                   if _bc_w is not None else ft.mean())

        if ln == 0:
            ln = torch.tensor(0.0, requires_grad=True, dtype=DTYPE).to(device)

        if self.inverse_enabled and self.inverse_data:
            ld = torch.tensor(0.0, dtype=DTYPE, device=device)
            for i, obs in self.inverse_data.items():
                up = self.predict(i, obs['x'], obs['t'])
                ld = ld + torch.mean((up - obs['u_obs']) ** 2)
            self.last_data_loss = float(ld.detach().cpu().item())
            lp = lp + self.lambda_data * ld
        else:
            self.last_data_loss = 0.0
        return lp, ln


    def _eval_l2(self):
        if self.validation_func is None:
            return float('inf')
        errors = []
        with torch.no_grad():
            for i, m in self.models.items():
                vf  = (self.validation_func[i]
                       if isinstance(self.validation_func, list)
                       else self.validation_func)
                n   = max(int(self.val_pts_per_unit * m['L']), 100)
                x_np = np.random.uniform(0, m['L'], n)
                t_np = np.random.uniform(0, self.t_max, n)
                x_t  = torch.tensor(x_np, dtype=DTYPE).view(-1, 1).to(device)
                t_t  = torch.tensor(t_np, dtype=DTYPE).view(-1, 1).to(device)
                up   = self.predict(i, x_t, t_t).detach().cpu().numpy().flatten()
                ue   = vf(x_np.reshape(-1, 1), t_np).flatten()
                nrm  = np.linalg.norm(ue)
                if nrm > 1e-10:
                    errors.append(np.linalg.norm(up - ue) / nrm)
        return float(np.mean(errors)) if errors else float('inf')

    def _predict_np(self, idx, x_t):

        t_mid = torch.full_like(x_t, self.t_max / 2.0)
        return self.predict(idx, x_t, t_mid).detach().cpu().numpy().flatten()

    def _exact_np(self, vf, x_np, L):
        return np.asarray(vf(x_np, L)).flatten()



    def _make_optimizer(self, params):
        if not self.use_singularity_capture:
            return optim.Adam(params, lr=self.lr)
        net_params  = [p for p in params if p is not self.xi_raw]
        xi_lr       = self.lr * getattr(self, 'xi_lr_scale', 0.1)
        self._xi_lr_base = xi_lr
        return optim.Adam([
            {'params': net_params,    'lr': self.lr},
            {'params': [self.xi_raw], 'lr': xi_lr},
        ])


    def report_l2(self, exact_func):
        errors = []
        with torch.no_grad():
            for i, m in self.models.items():
                vf   = exact_func[i] if isinstance(exact_func, list) else exact_func
                n    = max(int(getattr(self, 'val_pts_per_unit', 1000) * m['L']), 100)
                x_np = np.random.uniform(0, m['L'], n)
                t_np = np.random.uniform(0, self.t_max, n)
                x_t  = torch.tensor(x_np, dtype=DTYPE).view(-1, 1).to(device)
                t_t  = torch.tensor(t_np, dtype=DTYPE).view(-1, 1).to(device)
                up   = self.predict(i, x_t, t_t).detach().cpu().numpy().flatten()
                ue   = vf(x_np.reshape(-1, 1), t_np).flatten()
                nrm  = np.linalg.norm(ue)
                if nrm < 1e-10:
                    continue
                err  = np.linalg.norm(up - ue) / nrm
                errors.append(err)
                print(f"  Edge {i}: {err:.4e}  ({n} random pts)")
        if errors:
            print(f"  Mean: {np.mean(errors):.4e}")
        return errors




class EllipticPINNSolver(_PINNSolverBase):

    def __init__(self, graph, physics):
        super().__init__(graph, physics)

        self.anchor_X             = None
        self.anchor_U             = None
        self.use_anchors          = False
        self.lambda_data_schedule = None
        self.use_ntk_balance      = False
        self.ntk_every            = 200

        self.lr               = 5e-4
        self.scheduler_min_lr = 1e-6

    # ── elliptic-only setters ─────────────────────────────────────────────────

    def set_mesh(self, pts_per_unit=250, anchor_pts=0, grading_factor=1.5):
        self.pts_per_unit   = pts_per_unit
        self.anchor_pts     = anchor_pts
        self.grading_factor = grading_factor
        return self

    def set_lambda_data_schedule(self, schedule):
        self.lambda_data_schedule = schedule
        return self

    def set_ntk_balancing(self, enabled=True):
        self.use_ntk_balance = enabled
        return self

    def generate_noisy_edge_data(self, exact_func, n_points_per_edge=40,
                                 noise_std=0.01, seed=0):
        if not self.compiled:
            self.compile()
        rng = np.random.default_rng(seed)
        self.inverse_data = {}
        for i, m in self.models.items():
            n    = max(int(n_points_per_edge), 4)
            x_np = rng.uniform(0.0, m['L'], size=n)
            if isinstance(exact_func, list):
                clean = exact_func[i](x_np).reshape(-1)
            else:
                try:
                    clean = exact_func(x_np, i).reshape(-1)
                except TypeError:
                    clean = exact_func(x_np).reshape(-1)
            noisy = clean + rng.normal(0.0, float(noise_std), size=n)
            self.inverse_data[i] = dict(
                x=torch.tensor(x_np, dtype=DTYPE, device=device).view(-1, 1),
                u_obs=torch.tensor(noisy, dtype=DTYPE, device=device).view(-1, 1),
                noise_std=float(noise_std),
            )
        return self.inverse_data

    # ── fractional matrices ───────────────────────────────────────────────────

    @staticmethod
    def _l1(alpha, grid_np):
        return _compute_l1_matrix_vectorized(alpha, grid_np)

    @staticmethod
    def _l1_torch(alpha, grid_t):
        return _compute_l1_matrix_torch(alpha, grid_t)

    # kept for backward compat
    @staticmethod
    def compute_l1_matrix(alpha, x_np):
        return _compute_l1_matrix_vectorized(alpha, x_np)

    # ── mesh ──────────────────────────────────────────────────────────────────

    def _generate_mesh(self, L, n_pts):
        """Symmetric graded mesh on [0,L]: dense at both endpoints."""
        xi = np.linspace(-1, 1, n_pts)
        return (np.sign(xi) * np.abs(xi) ** self.grading_factor + 1) * 0.5 * L

    # ── compile ───────────────────────────────────────────────────────────────

    def compile(self):
        alpha0 = self.physics.alpha
        assert 1.0 < alpha0 < 2.0, \
            f"Elliptic engine requires 1 < alpha < 2, got alpha={alpha0}"

        self.models = {}
        for i, edge in enumerate(self.graph.edges):
            u_node, v_node, L = edge[0], edge[1], edge[2]
            phys_i = self._physics_list[i] if self._physics_list else self.physics
            n_pts  = max(int(self.pts_per_unit * L), 2)
            x_np   = self._generate_mesh(L, n_pts)
            x_grid = torch.tensor(x_np, dtype=DTYPE).view(-1, 1).to(device)

            a  = float(phys_i.alpha)
            Da = self._l1(a - 1.0, x_np)
            Db = self._l1(float(phys_i.beta), x_np)

            net = PINN_Net(
                in_dim=1, hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers, use_fourier=self.use_fourier,
                fourier_dim=self.fourier_dim, fourier_sigma=self.fourier_sigma,
                fourier_sampling=self.fourier_sampling,
            ).to(device)

            f_target = phys_i.get_f_target(x_np, L,
                                            Da.cpu().numpy(), Db.cpu().numpy())

            self.models[i] = {
                'net'     : net,
                'L'       : L,
                'n'       : n_pts,
                'nodes'   : (u_node, v_node),
                'x_grid'  : x_grid,
                'x_np'    : x_np,
                'Da'      : Da,
                'Db'      : Db,
                'f_target': torch.tensor(f_target, dtype=DTYPE).view(-1, 1).to(device),
                'physics' : phys_i,
            }

            if self.inverse_enabled:
                inv_params = {}
                for pname in self.inverse_parameter_names:
                    if not hasattr(phys_i, pname):
                        raise AttributeError(
                            f"Physics on edge {i} has no attribute '{pname}'.")
                    bnd = self.inverse_param_bounds.get(pname, None)
                    inv_params[pname] = {
                        "raw": nn.Parameter(
                            self._raw_from_value(float(getattr(phys_i, pname)), bnd).to(device)),
                        "bounds": bnd,
                    }
                if self.inverse_include_alpha:
                    ab = self.inverse_alpha_bounds
                    inv_params['alpha'] = {
                        "raw": nn.Parameter(
                            self._raw_from_value(float(phys_i.alpha), ab).to(device)),
                        "bounds": ab,
                    }
                if inv_params:
                    self.models[i]['inv_params'] = inv_params

        if self.anchor_pts > 0:
            self._run_fd_inverse_solver()
            if self.lambda_data_schedule is None:
                total = getattr(self, '_planned_epochs', 4000)
                p1, p2 = total // 4, total // 2
                self.lambda_data_schedule = (
                    lambda ep, p1=p1, p2=p2:
                    0.1 if ep < p1 else 0.01 if ep < p2 else 0.0)

        self.compiled = True
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, idx, x):
        m     = self.models[idx]
        u_raw = m['net'](x)
        if self.constraint_mode == "hard":
            u_node, v_node = m['nodes']
            if (self.bc_types.get(u_node, "dirichlet") == "dirichlet" and
                    self.bc_types.get(v_node, "dirichlet") == "dirichlet"):
                vl = self._get_bc_tensor(u_node)
                vr = self._get_bc_tensor(v_node)
                return vl + (x / m['L']) * (vr - vl) + x * (m['L'] - x) * u_raw
        return u_raw

    def _get_bc_tensor(self, node_idx):
        val = self.bc_values.get(node_idx, 0.0)
        return (val() if callable(val)
                else torch.tensor(float(val), dtype=DTYPE, device=device))

    # ── losses ────────────────────────────────────────────────────────────────

    def compute_losses(self, epoch=0):
        lp = torch.tensor(0.0, dtype=DTYPE, device=device)
        ln = torch.tensor(0.0, dtype=DTYPE, device=device)

        # ── PDE residual ──
        for i, m in self.models.items():
            if self.inverse_enabled and 'inv_params' in m:
                self._refresh_edge_physics_parameters(i)

            phys = m['physics']
            xt   = m['x_grid'].clone().detach().requires_grad_(True)
            u    = self.predict(i, xt)
            du   = torch.autograd.grad(u, xt, torch.ones_like(u),
                                       create_graph=True, allow_unused=True)[0]
            if du is None:
                du = torch.zeros_like(u)

            grid_1d     = m['x_grid'].view(-1)
            need_rebuild = (
                self.inverse_enabled and 'inv_params' in m
                and ('alpha' in m['inv_params'] or 'beta' in m['inv_params']))
            if need_rebuild:
                a_ord = (self._edge_param_value(i, 'alpha') if 'alpha' in m['inv_params']
                         else torch.tensor(float(phys.alpha), dtype=DTYPE, device=device))
                b_ord = (self._edge_param_value(i, 'beta') if 'beta' in m['inv_params']
                         else torch.tensor(float(phys.beta), dtype=DTYPE, device=device))
                Da_t      = self._l1_torch(a_ord - 1.0, grid_1d)
                Db_t      = self._l1_torch(b_ord, grid_1d)
                d_beta_u  = torch.mm(Db_t, u)
                d_alpha_u = torch.mm(Da_t, du)
            else:
                d_beta_u  = torch.mm(m['Db'], u)
                d_alpha_u = torch.mm(m['Da'], du)

            # residual: D^alpha u' - F(x, u, u', D^beta u, f) = 0
            res = d_alpha_u - phys.F(xt, u, du, d_beta_u, m['f_target'])
            lp  = lp + torch.mean(res ** 2)

        # ── BC / junction ── (evaluated over the full collocation grid, not one point)
        # Build degree map
        _deg = {}
        for edge in self.graph.edges:
            _deg[edge[0]] = _deg.get(edge[0], 0) + 1
            _deg[edge[1]] = _deg.get(edge[1], 0) + 1

        for node in self.graph.nodes:
            inc = []
            for i, m in self.models.items():
                if node == m['nodes'][0]: inc.append((i, 0.0,    1.0))
                if node == m['nodes'][1]: inc.append((i, m['L'], -1.0))
            if not inc:
                continue

            if len(inc) == 1:
                # leaf / boundary node — use full x_grid of that edge
                idx, xv, _ = inc[0]
                xi    = torch.tensor([[xv]], dtype=DTYPE, device=device,
                                     requires_grad=True)
                u_pred = self.predict(idx, xi)
                bc_type = self.bc_types.get(node, "dirichlet")
                if self.constraint_mode == "hard" and bc_type == "dirichlet":
                    pass
                elif bc_type == "dirichlet":
                    ln = ln + (u_pred - self._get_bc_tensor(node)).pow(2).mean()
                else:
                    du_p = torch.autograd.grad(u_pred, xi, torch.ones_like(u_pred),
                                               create_graph=True, allow_unused=True)[0]
                    ln = ln + ((du_p if du_p is not None
                                else torch.zeros_like(xi))
                               - self._get_bc_tensor(node)).pow(2).mean()
            else:
                # junction: continuity evaluated at boundary point of each edge
                preds = []
                for idx_e, xv, _ in inc:
                    xi = torch.tensor([[xv]], dtype=DTYPE, device=device)
                    preds.append(self.predict(idx_e, xi))
                for j in range(1, len(preds)):
                    ln = ln + (preds[0] - preds[j]).pow(2).mean()

                # Kirchhoff flux balance
                flux = torch.tensor(0.0, dtype=DTYPE, device=device)
                for idx_e, xv, s in inc:
                    xi    = torch.tensor([[xv]], dtype=DTYPE, device=device,
                                         requires_grad=True)
                    u_val = self.predict(idx_e, xi)
                    grad  = torch.autograd.grad(u_val, xi, torch.ones_like(u_val),
                                                create_graph=True, allow_unused=True)[0]
                    if grad is not None:
                        flux = flux + grad * s
                ln = ln + flux.pow(2).mean()

        # ── FD anchor data ──
        l_data = torch.tensor(0.0, dtype=DTYPE, device=device)
        if self.use_anchors and self.anchor_X is not None:
            lam = (self.lambda_data_schedule(epoch)
                   if self.lambda_data_schedule is not None else 0.0)
            if lam > 0.0:
                for i in self.anchor_X:
                    u_pred = self.predict(i, self.anchor_X[i])
                    l_data = l_data + lam * torch.mean(
                        (u_pred - self.anchor_U[i]) ** 2)

        # ── inverse data ──
        if self.inverse_enabled and self.inverse_data:
            ld = torch.tensor(0.0, dtype=DTYPE, device=device)
            for i, obs in self.inverse_data.items():
                up = self.predict(i, obs['x'])
                ld = ld + torch.mean((up - obs['u_obs']) ** 2)
            self.last_data_loss = float(ld.detach().cpu().item())
            lp = lp + self.lambda_data * ld
        else:
            self.last_data_loss = 0.0

        return lp, ln, l_data

    # ── eval L2 ───────────────────────────────────────────────────────────────

    def _eval_l2(self):
        if self.validation_func is None:
            return float('inf')
        errors = []
        with torch.no_grad():
            for i, m in self.models.items():
                vf   = (self.validation_func[i]
                        if isinstance(self.validation_func, list)
                        else self.validation_func)
                x_t  = torch.linspace(0, m['L'], 200, dtype=DTYPE).view(-1, 1).to(device)
                u_p  = self.predict(i, x_t).cpu().numpy().flatten()
                u_e  = vf(x_t.cpu().numpy().flatten(), m['L'])
                norm_e = np.linalg.norm(u_e)
                if norm_e < 1e-10:
                    continue
                errors.append(np.linalg.norm(u_p - u_e) / norm_e)
        return float(np.mean(errors)) if errors else float('inf')

    def _predict_np(self, idx, x_t):
        return self.predict(idx, x_t).detach().cpu().numpy().flatten()

    def _exact_np(self, vf, x_np, L):
        return np.asarray(vf(x_np, L)).flatten()

    # ── FD anchor solver ──────────────────────────────────────────────────────

    def _run_fd_inverse_solver(self):
        alpha = self.physics.alpha
        self.anchor_X = {}
        self.anchor_U = {}
        for i, edge in enumerate(self.graph.edges):
            u_node, v_node, L = edge[0], edge[1], edge[2]
            n_x  = self.anchor_pts
            x_np = np.linspace(0, L, n_x)
            dx   = x_np[1] - x_np[0]
            Da_fd = self._l1(alpha - 1.0, x_np).cpu().numpy()
            Db_fd = self._l1(self.physics.beta, x_np).cpu().numpy()
            f_np  = self.physics.get_f_target(x_np, L, Da_fd, Db_fd)
            bc_left  = float(self.bc_values.get(u_node, 0.0))
            bc_right = float(self.bc_values.get(v_node, 0.0))
            u = np.linspace(bc_left, bc_right, n_x)
            for _ in range(50):
                u_old = u.copy()
                du = np.zeros(n_x)
                du[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
                du[0]    = (u[1]  - u[0])   / dx
                du[-1]   = (u[-1] - u[-2])  / dx
                n_int = n_x - 2
                res_int = np.array([
                    (Da_fd @ du)[j+1]
                    - self.physics.F(
                        x_np[j+1], u[j+1], du[j+1],
                        (Db_fd @ u)[j+1], f_np[j+1])
                    for j in range(n_int)
                ])
                eps = 1e-6
                ab  = np.zeros((3, n_int))
                ab[1] = np.array([
                    ((Da_fd @ np.gradient(
                        np.array([u[k] + (eps if k == j+1 else 0)
                                  for k in range(n_x)]), dx))[j+1]
                     - self.physics.F(
                         x_np[j+1],
                         u[j+1] + eps, du[j+1],
                         (Db_fd @ (u + eps * (np.arange(n_x) == j+1)))[j+1],
                         f_np[j+1])
                     - res_int[j]) / eps
                    for j in range(n_int)
                ])
                for k in range(n_int):
                    j = k + 1
                    if k > 0:
                        u_m = u.copy(); u_m[j - 1] += eps
                        du_m = np.zeros(n_x)
                        du_m[1:-1] = (u_m[2:] - u_m[:-2]) / (2 * dx)
                        du_m[0]  = (u_m[1] - u_m[0]) / dx
                        du_m[-1] = (u_m[-1] - u_m[-2]) / dx
                        r_m = ((Da_fd @ du_m)[j]
                               - self.physics.F(x_np[j], u_m[j], du_m[j],
                                                (Db_fd @ u_m)[j], f_np[j]))
                        ab[2, k - 1] = (r_m - res_int[k]) / eps
                    if k < n_int - 1:
                        u_q = u.copy(); u_q[j + 1] += eps
                        du_q = np.zeros(n_x)
                        du_q[1:-1] = (u_q[2:] - u_q[:-2]) / (2 * dx)
                        du_q[0]  = (u_q[1] - u_q[0]) / dx
                        du_q[-1] = (u_q[-1] - u_q[-2]) / dx
                        r_q = ((Da_fd @ du_q)[j]
                               - self.physics.F(x_np[j], u_q[j], du_q[j],
                                                (Db_fd @ u_q)[j], f_np[j]))
                        ab[0, k + 1] = (r_q - res_int[k]) / eps
                delta = solve_banded((1, 1), ab, -res_int)
                u[1:-1] += delta
                u[0]  = bc_left
                u[-1] = bc_right
                if np.max(np.abs(u - u_old)) < 1e-10:
                    break
            if not np.all(np.isfinite(u)):
                u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
            self.anchor_X[i] = torch.tensor(x_np, dtype=DTYPE).view(-1, 1).to(device)
            self.anchor_U[i] = torch.tensor(u,    dtype=DTYPE).view(-1, 1).to(device)
        self.use_anchors = True




DEFAULT_ELLIPTIC_ARCH = dict(
    hidden_layers=4, hidden_dim=96,
    use_fourier=True, fourier_dim=64,
    fourier_sigma=8.0, fourier_sampling="sobol",
)

DEFAULT_PARABOLIC_ARCH = dict(
    hidden_layers=4, hidden_dim=128,
    use_fourier=True, fourier_dim=64,
    fourier_sigma=1.0, fourier_sampling="gaussian",
)


def run_elliptic_forward(
    graph, physics, exact_funcs, *,
    epochs=8000, n_starts=3, probe_epochs=600,
    seed_base=0, pts_per_unit=250, grading_factor=1.5, arch=None,
):

    arch = arch or DEFAULT_ELLIPTIC_ARCH
    solver = EllipticPINNSolver(graph, physics)
    solver.set_architecture(**arch)
    solver.set_mesh(pts_per_unit=pts_per_unit, anchor_pts=0,
                    grading_factor=grading_factor)
    solver.set_constraints("soft")
    solver.set_validation(exact_funcs)
    solver.compile()
    solver.train_multistart(
        epochs=epochs, strategy="dual", use_lbfgs=True,
        n_starts=n_starts, seed_base=seed_base, probe_epochs=probe_epochs,
    )
    solver.report_l2(exact_funcs)
    solver.plot_results(exact_func=exact_funcs)
    return solver


def run_elliptic_inverse(
    graph, physics_list, exact_funcs_per_edge, *,
    parameter_names=("beta", "reaction"),
    include_alpha=False, param_bounds=None,
    alpha_bounds=(1.05, 1.95), data_weight=20.0,
    n_points_per_edge=80, noise_std=0.01, seed=123,
    epochs=6000, pts_per_unit=220, grading_factor=1.5, arch=None,
):
    arch = arch or DEFAULT_ELLIPTIC_ARCH
    torch.manual_seed(seed); np.random.seed(seed)
    solver = EllipticPINNSolver(graph, physics_list)
    solver.set_architecture(**arch)
    solver.set_mesh(pts_per_unit=pts_per_unit, anchor_pts=0,
                    grading_factor=grading_factor)
    solver.set_constraints("soft")
    solver.set_inverse_problem(
        parameter_names=list(parameter_names),
        include_alpha=include_alpha,
        param_bounds=dict(param_bounds) if param_bounds else {},
        alpha_bounds=alpha_bounds, data_weight=data_weight,
    )
    val_list = [lambda x, L, i=i: exact_funcs_per_edge[i](x)
                for i in range(len(physics_list))]
    solver.set_validation(val_list)
    solver.compile()
    solver.generate_noisy_edge_data(
        exact_func=[lambda x, i=i: exact_funcs_per_edge[i](x)
                    for i in range(len(physics_list))],
        n_points_per_edge=n_points_per_edge, noise_std=noise_std, seed=seed,
    )
    solver.train_inverse(epochs=epochs, strategy="dual", use_lbfgs=True)
    est = solver.get_estimated_parameters()
    solver.report_l2(val_list)
    return solver, est


def run_parabolic_forward_sweep(
    graph, physics_list, exact_u_per_edge, *,
    r_values=(1.0, 2.0, 4.0), scheme="L21sigma",
    epochs=20000, n_starts=3, probe_epochs=1000,
    seed_base=0, pinn_pts=100, n_t=100, t_max=1.0, arch=None,
):
    arch    = arch or DEFAULT_PARABOLIC_ARCH
    n_edges = len(physics_list)
    results = {}
    for r in r_values:
        torch.manual_seed(seed_base); np.random.seed(seed_base)
        solver = ParabolicPINNSolver(graph, physics_list)
        solver.set_frac_scheme(scheme)
        solver.set_architecture(**arch)
        solver.set_mesh(mesh_type="power_law", pinn_pts=pinn_pts,
                        n_t=n_t, grading_factor=r, t_max=t_max)
        solver.set_constraints("soft")
        solver.set_causal_training(enabled=True, eps=1.0, eps_end=1e-4, bc_floor=0.1)
        alpha0 = float(getattr(physics_list[0], "alpha", 0.5))
        solver.set_singularity_capture(enabled=True, xi_init=alpha0, xi_loss_adaptive=True)
        solver.set_lr(lr=5e-4, min_lr=1e-5, xi_lr_scale=0.1, xi_lr_scale_phase2=0.01)
        solver.set_validation(
            [lambda x, t, i=i: exact_u_per_edge(i, x, t) for i in range(n_edges)],
            pts_per_unit=1000)
        solver.compile()
        solver.train_multistart(epochs=epochs, strategy="dual", use_lbfgs=True,
                                n_starts=n_starts, seed_base=seed_base,
                                probe_epochs=probe_epochs)
        results[r] = solver.report_l2(
            [lambda x, t, i=i: exact_u_per_edge(i, x, t) for i in range(n_edges)])
    return results


def parabolic_sweep_error_plot(results, r_values, alpha, scheme, n_edges,
                               save_path="parabolic_star_sweep.png"):
    r_vals = [r for r in r_values if results.get(r)]
    if not r_vals:
        return
    means  = [np.mean(results[r]) for r in r_vals]
    r_opt  = 2.0 / alpha if alpha > 0 else None
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"Parabolic star graph — per-edge mean rel L2\n"
                 f"alpha={alpha}, scheme={scheme}, {n_edges} edges", fontsize=12)
    best_r = r_vals[int(np.argmin(means))]
    ax.semilogy(r_vals, means, "s-", color="steelblue", linewidth=2, markersize=8)
    ax.scatter([best_r], [min(means)], color="red", zorder=5, s=120,
               label=f"best r={best_r:.1f} ({min(means):.2e})")
    if r_opt is not None:
        ax.axvline(r_opt, color="red", linestyle="--", linewidth=1.2,
                   label=f"opt r={r_opt:.1f}")
    ax.set_xlabel("Grading factor r"); ax.set_ylabel("Mean rel L2")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(list(r_values)); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.show()


def run_parabolic_inverse(
    graph, physics_list, exact_u_per_edge, *,
    scheme="L1", r_grading=2.0, epochs=12000,
    n_points_per_edge=60, noise_std=0.01, data_weight=20.0,
    seed=123, parameter_names=("nu",), include_alpha=True,
    param_bounds=None, alpha_bounds=(0.2, 0.95),
    pinn_pts=100, n_t=100, t_max=1.0, arch=None,
):
    arch    = arch or DEFAULT_PARABOLIC_ARCH
    n_edges = len(physics_list)
    torch.manual_seed(seed); np.random.seed(seed)
    solver = ParabolicPINNSolver(graph, physics_list)
    solver.set_frac_scheme(scheme)
    solver.set_architecture(**arch)
    solver.set_mesh(mesh_type="power_law", pinn_pts=pinn_pts,
                    n_t=n_t, grading_factor=r_grading, t_max=t_max)
    solver.set_constraints("soft")
    solver.set_causal_training(enabled=True, eps=1.0, eps_end=1e-4, bc_floor=0.1)
    alpha0 = float(getattr(physics_list[0], "alpha", 0.5))
    solver.set_singularity_capture(enabled=True, xi_init=alpha0, xi_loss_adaptive=True)
    solver.set_lr(lr=5e-4, min_lr=1e-5, xi_lr_scale=0.1, xi_lr_scale_phase2=0.01)
    pb = dict(param_bounds) if param_bounds else {}
    if "nu" in parameter_names and "nu" not in pb:
        pb["nu"] = (0.1, 2.0)
    solver.set_inverse_problem(parameter_names=list(parameter_names),
                               include_alpha=include_alpha,
                               param_bounds=pb, alpha_bounds=alpha_bounds,
                               data_weight=data_weight)
    exact_fns = [lambda x, t, i=i: exact_u_per_edge(i, x, t) for i in range(n_edges)]
    solver.compile()
    solver.generate_noisy_edge_data(exact_func=exact_fns,
                                    n_points_per_edge=n_points_per_edge,
                                    noise_std=noise_std, seed=seed)
    solver.train_inverse(epochs=epochs, strategy="dual", use_lbfgs=True)
    est = solver.get_estimated_parameters()
    solver.report_l2(exact_fns)
    return solver, est
