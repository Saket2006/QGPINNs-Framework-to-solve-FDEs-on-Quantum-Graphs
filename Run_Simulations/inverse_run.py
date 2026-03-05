import sys
sys.path.insert(0, '/kaggle/working')

from Parabolic_Engine import ParabolicPINNSolver
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import gamma

ALPHA        = 0.5
SCHEME       = "L21sigma"
TRUE_KAPPA   = 0.1
KAPPA_INIT   = 0.5
N_OBS        = 200
NOISE_LEVEL  = 0.01
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


class InverseKappaPhysics:
    def __init__(self, alpha, kappa_param):
        self.alpha       = alpha
        self.kappa_param = kappa_param

    def F(self, x, t, u, u_x, u_xx, dt_alpha_u, f_target):
        return dt_alpha_u - self.kappa_param * u_xx + (u * u_x) - f_target

    def get_f_target(self, x, t, L):
        a      = self.alpha
        G1     = gamma(a + 1)
        t_a    = torch.pow(t.clamp(min=1e-10), a)
        t_2a   = torch.pow(t.clamp(min=1e-10), 2 * a)
        sinpx  = torch.sin(np.pi * x)
        sin2px = torch.sin(2 * np.pi * x)
        return G1 * sinpx + TRUE_KAPPA * np.pi**2 * t_a * sinpx + 0.5 * np.pi * t_2a * sin2px

    def get_ic(self, x):
        return torch.zeros_like(x)


def exact_u(x, t, alpha=ALPHA):
    return t ** alpha * np.sin(np.pi * np.asarray(x))


def make_observations(n_obs, noise_level, alpha, seed=42):
    rng   = np.random.default_rng(seed)
    x_obs = rng.uniform(0.0, 1.0, n_obs)
    t_obs = rng.uniform(0.05, 1.0, n_obs)
    u_obs = exact_u(x_obs, t_obs, alpha)
    if noise_level > 0:
        u_obs = u_obs + noise_level * np.abs(u_obs) * rng.standard_normal(n_obs)
    to = lambda a: torch.tensor(a, dtype=torch.float64).view(-1, 1).to(device)
    return to(x_obs), to(t_obs), to(u_obs)


def make_solver(r, x_obs, t_obs, u_obs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    kappa = nn.Parameter(torch.tensor(KAPPA_INIT, dtype=torch.float64).to(device))
    kappa._inv_init = KAPPA_INIT
    solver = ParabolicPINNSolver(
        graph=Graph(),
        physics=InverseKappaPhysics(ALPHA, kappa),
    )
    solver.set_frac_scheme(SCHEME)
    solver.set_architecture(**ARCH)
    solver.set_mesh(mesh_type="power_law", pinn_pts=PINN_PTS,
                    n_t=N_T, grading_factor=r, t_max=1.0)
    solver.set_constraints("soft", bc_types=BC_TYPES, bc_values=BC_VALUES)
    solver.set_ntk_balancing(enabled=True)
    solver.ntk_every = 200
    solver.set_rad_resampling(enabled=False)
    solver.set_validation(lambda x, t: exact_u(x, t, ALPHA), times=EVAL_TIMES)
    solver.set_inverse_param("kappa", kappa)
    solver.set_observations(x_obs, t_obs, u_obs, lambda_data=10.0)
    solver.compile()
    return solver, kappa


def run_sweep():
    x_obs, t_obs, u_obs = make_observations(N_OBS, NOISE_LEVEL, ALPHA)
    print(f"Generated {N_OBS} observations from exact solution  (noise={NOISE_LEVEL*100:.1f}%)")

    results = {}

    for r in R_VALUES:
        print(f"\n{'='*55}")
        print(f"  INVERSE  r={r}  scheme={SCHEME}  alpha={ALPHA}")
        print(f"  true κ={TRUE_KAPPA}  init κ={KAPPA_INIT}")
        print(f"{'='*55}")

        solver, kappa = make_solver(r, x_obs, t_obs, u_obs)
        solver.train_multistart(
            epochs=EPOCHS, strategy="dual", use_lbfgs=True,
            n_starts=N_STARTS, seed_base=0, probe_epochs=PROBE_EPOCHS,
            log_l2_every=LOG_EVERY,
        )

        rec       = kappa.item()
        rel_kappa = abs(rec - TRUE_KAPPA) / TRUE_KAPPA * 100
        l2_curve  = solver.l2_history
        final_err = solver._eval_l2_at_times(EVAL_TIMES)
        best_mean = min(v for _, v in l2_curve) if l2_curve else float('inf')
        kappa_hist = solver._inv_param_histories["kappa"]

        results[r] = {
            'l2_curve':    l2_curve,
            'final_err':   final_err,
            'best_mean':   best_mean,
            'kappa_rec':   rec,
            'rel_kappa':   rel_kappa,
            'kappa_hist':  kappa_hist,
            'solver':      solver,
            'kappa_param': kappa,
        }

        solver.report_inverse(true_values={"kappa": TRUE_KAPPA})
        print(f"  Final per-time L2 errors:")
        for t_val, err in final_err.items():
            print(f"    t={t_val}: {err:.4e}")

    plot_all(results, x_obs, t_obs, u_obs)
    print_table(results)
    return results


def plot_all(results, x_obs, t_obs, u_obs):
    colors   = plt.cm.tab10(np.linspace(0, 0.9, len(R_VALUES)))
    r_color  = {r: colors[i] for i, r in enumerate(R_VALUES)}
    t_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(EVAL_TIMES)))

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        f"Inverse: recover κ  |  true={TRUE_KAPPA}  init={KAPPA_INIT}  "
        f"N_obs={N_OBS}  noise={NOISE_LEVEL*100:.1f}%  scheme={SCHEME}  α={ALPHA}",
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
    ax2.set_xlabel("r"); ax2.set_ylabel("Best mean L2")
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
    for r in R_VALUES:
        hist = results[r]['kappa_hist']
        ax4.plot(hist, lw=1.5, color=r_color[r], label=f"r={r}")
    ax4.axhline(TRUE_KAPPA, ls='--', lw=1.8, color='k', label=f'true={TRUE_KAPPA}')
    ax4.axhline(KAPPA_INIT, ls=':',  lw=1.2, color='gray', label=f'init={KAPPA_INIT}')
    ax4.set_xlabel("Step"); ax4.set_ylabel("κ")
    ax4.set_title("κ convergence (all r)"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(3, 4, 5)
    rec_vals = [results[r]['kappa_rec'] for r in R_VALUES]
    rel_errs = [results[r]['rel_kappa'] for r in R_VALUES]
    bars2 = ax5.bar([str(r) for r in R_VALUES], rel_errs,
                    color=[r_color[r] for r in R_VALUES], edgecolor='k', linewidth=0.8)
    best_kappa_idx = int(np.argmin(rel_errs))
    bars2[best_kappa_idx].set_edgecolor('red'); bars2[best_kappa_idx].set_linewidth(2.5)
    ax5.set_xlabel("r"); ax5.set_ylabel("Rel error κ (%)")
    ax5.set_title("κ recovery error per r"); ax5.grid(axis='y', alpha=0.3)
    for bar, val, rec in zip(bars2, rel_errs, rec_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f"{rec:.4f}", ha='center', va='bottom', fontsize=7, rotation=45)

    ax6 = fig.add_subplot(3, 4, 6)
    sc = ax6.scatter(x_obs.cpu().numpy(), t_obs.cpu().numpy(),
                     c=u_obs.cpu().numpy(), cmap='viridis', s=18)
    plt.colorbar(sc, ax=ax6, label='u obs')
    ax6.set_xlabel('x'); ax6.set_ylabel('t')
    ax6.set_title(f'Observations (N={N_OBS})'); ax6.grid(alpha=0.3)

    best_r = R_VALUES[int(np.argmin([results[r]['best_mean'] for r in R_VALUES]))]
    solver  = results[best_r]['solver']
    m       = solver.models[0]
    x_plot  = np.linspace(0, m['L'], 200)
    x_t     = torch.tensor(x_plot, dtype=torch.float64).view(-1, 1).to(device)

    for idx, t_val in enumerate(EVAL_TIMES):
        ax = fig.add_subplot(3, 4, 9 + idx)
        t_t = torch.full_like(x_t, t_val)
        u_e = exact_u(x_plot, t_val)
        ax.plot(x_plot, u_e, 'k--', lw=1.8, label='Exact')
        for r in R_VALUES:
            s   = results[r]['solver']
            u_p = s.predict(0, x_t, t_t).detach().cpu().numpy().flatten()
            ax.plot(x_plot, u_p, lw=1.2, color=r_color[r], alpha=0.85, label=f"r={r}")
        tol  = 0.08
        mask = np.abs(t_obs.cpu().numpy().flatten() - t_val) < tol
        if mask.sum():
            ax.scatter(x_obs.cpu().numpy().flatten()[mask],
                       u_obs.cpu().numpy().flatten()[mask],
                       s=18, c='steelblue', zorder=5, label='Obs')
        ax.set_title(f"t={t_val}"); ax.set_xlabel("x")
        if idx == 0: ax.set_ylabel("u(x,t)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax7 = fig.add_subplot(3, 4, 7)
    best_s  = results[best_r]['solver']
    best_kp = results[best_r]['kappa_param']
    rec     = best_kp.item()
    t_plot  = torch.full_like(x_t, 1.0)
    u_p1    = best_s.predict(0, x_t, t_plot).detach().cpu().numpy().flatten()
    u_e1    = exact_u(x_plot, 1.0)
    ax7.semilogy(x_plot, np.abs(u_p1 - u_e1) + 1e-16, color='purple', lw=1.5)
    ax7.set_xlabel('x'); ax7.set_ylabel('|error|')
    ax7.set_title(f'Pointwise error t=1  (best r={best_r})'); ax7.grid(alpha=0.3)

    ax8 = fig.add_subplot(3, 4, 8)
    ax8.axis('off')
    best_rel = results[best_r]['rel_kappa']
    rows = [
        ("Best r",       f"{best_r}"),
        ("True κ",       f"{TRUE_KAPPA}"),
        ("Init κ",       f"{KAPPA_INIT}"),
        ("Recovered κ",  f"{rec:.6f}"),
        ("Rel err κ",    f"{best_rel:.3f}%"),
        ("N obs",        f"{N_OBS}"),
        ("Noise",        f"{NOISE_LEVEL*100:.1f}%"),
        ("Scheme",       SCHEME),
        ("α",            f"{ALPHA}"),
    ]
    for k, (lbl, val) in enumerate(rows):
        y = 0.95 - k * 0.1
        ax8.text(0.02, y, lbl + ":", fontsize=10, va='top', fontweight='bold')
        ax8.text(0.55, y, val, fontsize=10, va='top',
                 color='crimson' if 'Recovered' in lbl else 'black')

    plt.tight_layout()
    plt.savefig("inverse_sweep.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → inverse_sweep.png")


def print_table(results):
    header = f"\n{'r':>5}  κ_rec     rel_err_κ  " + \
             "  ".join(f"t={t}" for t in EVAL_TIMES) + "  mean_best"
    print(header)
    print("-" * (len(header) + 10))
    for r in R_VALUES:
        res  = results[r]
        errs = [res['final_err'].get(t, float('inf')) for t in EVAL_TIMES]
        row  = (f"{r:>5.1f}  {res['kappa_rec']:.6f}  {res['rel_kappa']:>8.3f}%  "
                + "  ".join(f"{e:.3e}" for e in errs)
                + f"  {res['best_mean']:.3e}")
        print(row)


if __name__ == "__main__":
    run_sweep()
