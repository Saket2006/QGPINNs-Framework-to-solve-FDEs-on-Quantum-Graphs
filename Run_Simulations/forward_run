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
EPOCHS       = 15000
N_STARTS     = 3
PROBE_EPOCHS = 1000
LOG_EVERY    = 100

ARCH = dict(hidden_layers=4, hidden_dim=60, use_fourier=True,
            fourier_dim=64, fourier_sigma=1.0, fourier_sampling="gaussian")
N_T      = 250
PINN_PTS = 200
BC_TYPES  = {0: "dirichlet", 1: "dirichlet"}
BC_VALUES = {0: 0.0, 1: 0.0}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Graph:
    def __init__(self):
        self.edges = [(0, 1, 1.0)]
        self.nodes = [0, 1]


class Physics:
    def __init__(self, alpha):
        self.alpha = alpha

    def F(self, x, t, u, u_x, u_xx, dt_alpha_u, f_target):
        return dt_alpha_u - 0.1 * u_xx + (u * u_x) - f_target

    def get_f_target(self, x, t, L):
        a      = self.alpha
        G1     = gamma(a + 1)
        t_a    = torch.pow(t.clamp(min=1e-10), a)
        t_2a   = torch.pow(t.clamp(min=1e-10), 2 * a)
        sinpx  = torch.sin(np.pi * x)
        sin2px = torch.sin(2 * np.pi * x)
        return G1 * sinpx + 0.1 * np.pi**2 * t_a * sinpx + 0.5 * np.pi * t_2a * sin2px

    def get_ic(self, x):
        return torch.zeros_like(x)


def exact_u(x, t, alpha=ALPHA):
    return t ** alpha * np.sin(np.pi * np.asarray(x))


def make_solver(r, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    solver = ParabolicPINNSolver(graph=Graph(), physics=Physics(ALPHA))
    solver.set_frac_scheme(SCHEME)
    solver.set_architecture(**ARCH)
    solver.set_mesh(mesh_type="power_law", pinn_pts=PINN_PTS,
                    n_t=N_T, grading_factor=r, t_max=1.0)
    solver.set_constraints("soft", bc_types=BC_TYPES, bc_values=BC_VALUES)
    solver.set_ntk_balancing(enabled=True)
    solver.ntk_every = 200
    solver.set_rad_resampling(enabled=False)
    solver.set_validation(lambda x, t: exact_u(x, t, ALPHA), times=EVAL_TIMES)
    solver.compile()
    return solver


def run_sweep():
    results = {}

    for r in R_VALUES:
        print(f"\n{'='*55}")
        print(f"  r = {r}   scheme={SCHEME}   alpha={ALPHA}")
        print(f"{'='*55}")

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

        print(f"\n  Final per-time L2 errors:")
        for t_val, err in final_err.items():
            print(f"    t={t_val}: {err:.4e}")
        print(f"  Best mean L2 over training: {best_mean:.4e}")

    plot_all(results)
    print_table(results)
    return results


def plot_all(results):
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(R_VALUES)))
    r_color = {r: colors[i] for i, r in enumerate(R_VALUES)}

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Forward problem sweep  |  scheme={SCHEME}  α={ALPHA}  "
                 f"epochs={EPOCHS}  multistart {N_STARTS}×{PROBE_EPOCHS}+{EPOCHS}",
                 fontsize=12)

    ax1 = fig.add_subplot(2, 3, 1)
    for r in R_VALUES:
        curve = results[r]['l2_curve']
        if not curve: continue
        iters, errs = zip(*curve)
        ax1.semilogy(iters, errs, lw=1.8, color=r_color[r], label=f"r={r}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Mean rel L2")
    ax1.set_title("L2 error vs iterations (all r)"); ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    means = [results[r]['best_mean'] for r in R_VALUES]
    bars  = ax2.bar([str(r) for r in R_VALUES], means, color=[r_color[r] for r in R_VALUES],
                    edgecolor='k', linewidth=0.8)
    best_idx = int(np.argmin(means))
    bars[best_idx].set_edgecolor('red'); bars[best_idx].set_linewidth(2.5)
    ax2.set_yscale('log')
    ax2.set_xlabel("r"); ax2.set_ylabel("Best mean rel L2")
    ax2.set_title("Best L2 per r (red = winner)"); ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, val*1.05,
                 f"{val:.2e}", ha='center', va='bottom', fontsize=8)

    ax3 = fig.add_subplot(2, 3, 3)
    t_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(EVAL_TIMES)))
    for j, t_val in enumerate(EVAL_TIMES):
        errs_t = [results[r]['final_err'].get(t_val, float('inf')) for r in R_VALUES]
        ax3.semilogy(R_VALUES, errs_t, "o-", color=t_colors[j],
                     lw=1.8, markersize=6, label=f"t={t_val}")
    ax3.set_xlabel("r"); ax3.set_ylabel("Final rel L2")
    ax3.set_title("Final L2 per time-slice vs r")
    ax3.legend(fontsize=9); ax3.grid(True, which="both", alpha=0.3)
    ax3.set_xticks(R_VALUES)

    for idx, t_val in enumerate(EVAL_TIMES):
        ax = fig.add_subplot(2, 4, 5 + idx)
        best_r = R_VALUES[int(np.argmin([results[r]['final_err'].get(t_val, np.inf) for r in R_VALUES]))]
        solver  = results[best_r]['solver']
        m       = solver.models[0]
        x_plot  = np.linspace(0, m['L'], 200)
        x_t     = torch.tensor(x_plot, dtype=torch.float64).view(-1, 1).to(device)
        t_t     = torch.full_like(x_t, t_val)
        u_e     = exact_u(x_plot, t_val)
        ax.plot(x_plot, u_e, 'k--', lw=1.8, label='Exact')
        for r in R_VALUES:
            s   = results[r]['solver']
            u_p = s.predict(0, x_t, t_t).detach().cpu().numpy().flatten()
            ax.plot(x_plot, u_p, lw=1.2, color=r_color[r],
                    alpha=0.85, label=f"r={r}")
        ax.set_title(f"t = {t_val}  (best r={best_r})")
        ax.set_xlabel("x")
        if idx == 0: ax.set_ylabel("u(x,t)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("forward_sweep.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → forward_sweep.png")


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
