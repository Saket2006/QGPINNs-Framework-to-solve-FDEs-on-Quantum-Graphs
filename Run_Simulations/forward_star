import sys
sys.path.insert(0, '/kaggle/working')

from Parabolic_Engine import ParabolicPINNSolver
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import gamma


ALPHA        = 0.5
SCHEME       = "L21sigma"
R_VALUES     = [1.0, 1.2, 1.5, 1.8, 2.0]
EVAL_TIMES   = [0.25, 0.5, 0.75, 1.0]
EPOCHS       = 20000
N_STARTS     = 3
PROBE_EPOCHS = 1000
LOG_EVERY    = 100

ARCH = dict(hidden_layers=4, hidden_dim=60, use_fourier=True,
            fourier_dim=64, fourier_sigma=1.0, fourier_sampling="gaussian")
N_T      = 150
PINN_PTS = 100

JUNCTION_NODE = 0
LEAF_NODES    = [1, 2, 3]
N_EDGES       = 3
EDGE_LENGTH   = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StarGraph:
    def __init__(self):
        self.edges = [
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
        ]
        self.nodes = [0, 1, 2, 3]

class StarPhysics:
    def __init__(self, alpha):
        self.alpha = alpha

    def F(self, x, t, u, u_x, u_xx, dt_alpha_u, f_target):
        return dt_alpha_u - 0.1 * u_xx + (u * u_x) - f_target

    def get_f_target(self, x, t, L):
        a      = self.alpha
        G1     = gamma(a + 1)
        t_a    = torch.pow(t.clamp(min=1e-10), a)
        t_2a   = torch.pow(t.clamp(min=1e-10), 2 * a)
        cos_hx = torch.cos(np.pi * x / 2)
        sin_px = torch.sin(np.pi * x)
        return (G1 * cos_hx
                + 0.1 * (np.pi / 2)**2 * t_a * cos_hx
                - (np.pi / 4) * t_2a * sin_px)

    def get_ic(self, x):
        return torch.zeros_like(x)

def exact_u(x, t, alpha=ALPHA):
    return t ** alpha * np.cos(np.pi * np.asarray(x) / 2)


class StarPINNSolver(ParabolicPINNSolver):

    def __init__(self, graph, physics,
                 lambda_continuity=1.0, lambda_kirchhoff=1.0):
        super().__init__(graph, physics)
        self.lambda_continuity = lambda_continuity
        self.lambda_kirchhoff  = lambda_kirchhoff

    def compute_losses(self, epoch=0):
        lp, ln, l_data = super().compute_losses(epoch)

        T_junc = self.t_grid.clone().detach().requires_grad_(True)
        if self.use_time_windowing and self.current_t_max is not None:
            mask = (T_junc <= self.current_t_max).flatten()
            if mask.sum() == 0:
                return lp, ln, l_data
            T_junc = T_junc[mask].requires_grad_(True)

        X_junc = torch.zeros_like(T_junc)

        u_at_junc  = []
        ux_at_junc = []
        for edge_idx in range(N_EDGES):
            X_j = X_junc.clone().requires_grad_(True)
            u_j = self.predict(edge_idx, X_j, T_junc)
            u_x_j = torch.autograd.grad(
                u_j, X_j, torch.ones_like(u_j), create_graph=True
            )[0]
            u_at_junc.append(u_j)
            ux_at_junc.append(u_x_j)

        l_cont = torch.tensor(0.0, dtype=torch.float64).to(device)
        ref = u_at_junc[0]
        for k in range(1, N_EDGES):
            l_cont = l_cont + torch.mean((u_at_junc[k] - ref) ** 2)
        l_cont = l_cont / (N_EDGES - 1)

        kirchhoff_sum = sum(ux_at_junc)
        l_kirch = torch.mean(kirchhoff_sum ** 2)

        ln = ln + self.lambda_continuity * l_cont + self.lambda_kirchhoff * l_kirch

        return lp, ln, l_data

def make_solver(r, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)

    bc_types = {
        JUNCTION_NODE: "none", 
        1: "dirichlet",
        2: "dirichlet",
        3: "dirichlet",
    }
    bc_values = {1: 0.0, 2: 0.0, 3: 0.0}

    solver = StarPINNSolver(
        graph=StarGraph(),
        physics=StarPhysics(ALPHA),
        lambda_continuity=10.0,
        lambda_kirchhoff=10.0,
    )
    solver.set_frac_scheme(SCHEME)
    solver.set_architecture(**ARCH)
    solver.set_mesh(mesh_type="power_law", pinn_pts=PINN_PTS,
                    n_t=N_T, grading_factor=r, t_max=1.0)
    solver.set_constraints("soft", bc_types=bc_types, bc_values=bc_values)
    solver.set_ntk_balancing(enabled=True)
    solver.ntk_every = 200
    solver.set_rad_resampling(enabled=False)
    solver.set_validation(lambda x, t: exact_u(x, t, ALPHA), times=EVAL_TIMES)
    solver.compile()
    return solver



def run_sweep():


    results = {}

    for r in R_VALUES:
        print(f"\n{'='*60}")
        print(f"  T(3,2) star  |  r={r}  scheme={SCHEME}  α={ALPHA}")
        print(f"{'='*60}")

        solver = make_solver(r)
        solver.train_multistart(
            epochs=EPOCHS, strategy="dual", use_lbfgs=True,
            n_starts=N_STARTS, seed_base=0, probe_epochs=PROBE_EPOCHS,
            log_l2_every=LOG_EVERY,
        )


        l2_curve  = solver.l2_history
        final_err = solver._eval_l2_at_times(EVAL_TIMES)
        best_mean = min(v for _, v in l2_curve) if l2_curve else float('inf')

        results[r] = {
            'l2_curve':  l2_curve,
            'final_err': final_err,
            'best_mean': best_mean,
            'solver':    solver,
        }

        print(f"\n  Final per-time L2 errors (averaged over 3 edges):")
        for t_val, err in final_err.items():
            print(f"    t={t_val}: {err:.4e}")

    plot_all(results)
    print_table(results)
    return results

def plot_all(results):
    colors   = plt.cm.tab10(np.linspace(0, 0.9, len(R_VALUES)))
    r_color  = {r: colors[i] for i, r in enumerate(R_VALUES)}
    t_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(EVAL_TIMES)))
    e_styles = ['-', '--', ':']

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        f"T(3,2) star metric graph  |  u=t^α cos(πx/2)  "
        f"scheme={SCHEME}  α={ALPHA}  edges={N_EDGES}×L={EDGE_LENGTH}",
        fontsize=12
    )

    ax1 = fig.add_subplot(3, 4, 1)
    for r in R_VALUES:
        curve = results[r]['l2_curve']
        if not curve: continue
        iters, errs = zip(*curve)
        ax1.semilogy(iters, errs, lw=1.8, color=r_color[r], label=f"r={r}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Mean rel L2")
    ax1.set_title("L2 vs iterations"); ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = fig.add_subplot(3, 4, 2)
    means = [results[r]['best_mean'] for r in R_VALUES]
    bars  = ax2.bar([str(r) for r in R_VALUES], means,
                    color=[r_color[r] for r in R_VALUES], edgecolor='k', linewidth=0.8)
    best_idx = int(np.argmin(means))
    bars[best_idx].set_edgecolor('red'); bars[best_idx].set_linewidth(2.5)
    ax2.set_yscale('log')
    ax2.set_xlabel("r"); ax2.set_ylabel("Best mean rel L2")
    ax2.set_title("Best L2 per r (red = winner)"); ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, val*1.05,
                 f"{val:.2e}", ha='center', va='bottom', fontsize=8)

    ax3 = fig.add_subplot(3, 4, 3)
    for j, t_val in enumerate(EVAL_TIMES):
        errs_t = [results[r]['final_err'].get(t_val, np.inf) for r in R_VALUES]
        ax3.semilogy(R_VALUES, errs_t, "o-", color=t_colors[j],
                     lw=1.8, markersize=6, label=f"t={t_val}")
    ax3.set_xlabel("r"); ax3.set_ylabel("Final rel L2")
    ax3.set_title("Final L2 per slice vs r")
    ax3.legend(fontsize=9); ax3.grid(True, which="both", alpha=0.3)
    ax3.set_xticks(R_VALUES)

    ax4 = fig.add_subplot(3, 4, 4)
    ax4.axis('off')
    jx, jy = 0.5, 0.5
    leaf_pos = [(0.5, 0.9), (0.15, 0.2), (0.85, 0.2)]
    leaf_labels = ['Node 1\n(leaf)', 'Node 2\n(leaf)', 'Node 3\n(leaf)']
    for (lx, ly), lbl in zip(leaf_pos, leaf_labels):
        ax4.annotate("", xy=(lx, ly), xytext=(jx, jy),
                     arrowprops=dict(arrowstyle="-", lw=2, color='steelblue'))
        ax4.text(lx, ly, lbl, ha='center', va='center', fontsize=9,
                 bbox=dict(boxstyle='round', fc='lightblue', ec='steelblue'))
    ax4.text(jx, jy, 'Node 0\n(junction)', ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', fc='lightyellow', ec='darkorange', lw=2))
    for k, (lx, ly) in enumerate(leaf_pos):
        mx, my = (jx+lx)/2, (jy+ly)/2
        ax4.text(mx + 0.04, my, f'e{k}\nL=1', fontsize=8, color='steelblue', ha='center')
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.set_title("T(3,2) star topology\nKirchhoff + continuity at junction")

    best_r  = R_VALUES[int(np.argmin(means))]
    solver  = results[best_r]['solver']
    x_plot  = np.linspace(0, EDGE_LENGTH, 200)
    x_t     = torch.tensor(x_plot, dtype=torch.float64).view(-1, 1).to(device)

    for idx, t_val in enumerate(EVAL_TIMES):
        ax = fig.add_subplot(3, 4, 5 + idx)
        t_t = torch.full_like(x_t, t_val)
        u_e = exact_u(x_plot, t_val)
        ax.plot(x_plot, u_e, 'k--', lw=2.0, label='Exact', zorder=5)
        for edge_idx in range(N_EDGES):
            u_p = solver.predict(edge_idx, x_t, t_t).detach().cpu().numpy().flatten()
            ax.plot(x_plot, u_p, lw=1.4, ls=e_styles[edge_idx],
                    color=f'C{edge_idx}', label=f'Edge {edge_idx}')
        ax.set_title(f"t={t_val}  (best r={best_r})")
        ax.set_xlabel("x  (0=junction, 1=leaf)")
        if idx == 0: ax.set_ylabel("u(x,t)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax9 = fig.add_subplot(3, 4, 9)
    t_t = torch.full_like(x_t, 1.0)
    u_e1 = exact_u(x_plot, 1.0)
    for r in R_VALUES:
        s   = results[r]['solver']
        u_p = s.predict(0, x_t, t_t).detach().cpu().numpy().flatten()
        ax9.semilogy(x_plot, np.abs(u_p - u_e1) + 1e-16,
                     lw=1.4, color=r_color[r], label=f"r={r}")
    ax9.set_xlabel("x"); ax9.set_ylabel("|error|")
    ax9.set_title("Pointwise error t=1, edge 0"); ax9.legend(fontsize=9)
    ax9.grid(True, which="both", alpha=0.3)

    ax10 = fig.add_subplot(3, 4, 10)
    t_range = np.linspace(0.01, 1.0, 100)
    t_torch = torch.tensor(t_range, dtype=torch.float64).view(-1, 1).to(device)
    X0      = torch.zeros_like(t_torch)
    u_exact_junc = exact_u(np.zeros(100), t_range)
    ax10.plot(t_range, u_exact_junc, 'k--', lw=2, label='Exact u(0,t)=t^α')
    for edge_idx in range(N_EDGES):
        u_j = solver.predict(edge_idx, X0, t_torch).detach().cpu().numpy().flatten()
        ax10.plot(t_range, u_j, lw=1.4, ls=e_styles[edge_idx],
                  color=f'C{edge_idx}', label=f'Edge {edge_idx}')
    ax10.set_xlabel("t"); ax10.set_ylabel("u(0,t)")
    ax10.set_title(f"Junction value continuity (best r={best_r})")
    ax10.legend(fontsize=9); ax10.grid(alpha=0.3)

    ax11 = fig.add_subplot(3, 4, 11)
    kirch_vals = []
    with torch.enable_grad():
        for t_val in t_range:
            T_k = torch.full((1, 1), t_val, dtype=torch.float64).to(device)
            s_total = 0.0
            for edge_idx in range(N_EDGES):
                X_k = torch.zeros(1, 1, dtype=torch.float64).to(device).requires_grad_(True)
                u_k = solver.predict(edge_idx, X_k, T_k)
                u_x = torch.autograd.grad(u_k, X_k, torch.ones_like(u_k))[0]
                s_total += u_x.item()
            kirch_vals.append(abs(s_total))
    ax11.semilogy(t_range, np.array(kirch_vals) + 1e-16,
                  color='purple', lw=1.5, label='|Σ ∂u/∂x|_{x=0}|')
    ax11.axhline(1e-3, ls='--', color='gray', lw=1, label='1e-3 reference')
    ax11.set_xlabel("t"); ax11.set_ylabel("|Kirchhoff residual|")
    ax11.set_title(f"Kirchhoff condition (best r={best_r})")
    ax11.legend(fontsize=9); ax11.grid(True, which="both", alpha=0.3)

    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis('off')
    col_labels = ['r'] + [f't={t}' for t in EVAL_TIMES] + ['best_mean']
    table_data = []
    for r in R_VALUES:
        row = [str(r)]
        for t_val in EVAL_TIMES:
            row.append(f"{results[r]['final_err'].get(t_val, np.inf):.2e}")
        row.append(f"{results[r]['best_mean']:.2e}")
        table_data.append(row)
    tbl = ax12.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.2, 1.6)
    ax12.set_title("L2 error summary", pad=20)

    plt.tight_layout()
    plt.savefig("T32_star_sweep.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → T32_star_sweep.png")

def print_table(results):
    header = f"\n{'r':>5}  " + "  ".join(f"t={t}" for t in EVAL_TIMES) + "  mean_best"
    print(header)
    print("-" * len(header))
    for r in R_VALUES:
        errs = [results[r]['final_err'].get(t, float('inf')) for t in EVAL_TIMES]
        row  = f"{r:>5.1f}  " + "  ".join(f"{e:.3e}" for e in errs)
        row += f"  {results[r]['best_mean']:.3e}"
        print(row)


if __name__ == "__main__":
    run_sweep()
