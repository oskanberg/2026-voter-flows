"""Posterior predictive checks for the transition matrix.

Runs the PPC for the baseline model (idata.nc) and, if the latent-model
outputs exist on disk, also for the latent-Reform model (idata_latent.nc).
Each PPC compares posterior-mean predictions against observed ward shares.

For each output category we report RMSE, R^2, mean bias, and the best-fit
observed-on-predicted line (slope + intercept + Pearson). The slope is the
key diagnostic for whether the model's predicted *magnitude* of variation
matches the observed range, independent of how well it ranks wards (which
is what Pearson captures).

Outputs:
    figures/ppc.png             baseline PPC
    figures/ppc_latent.png      latent-model PPC (if available)
    ppc.csv                     per-category metrics for the baseline
    ppc_latent.csv              per-category metrics for the latent (if available)
"""

import csv
from pathlib import Path

import numpy as np
import arviz as az
import matplotlib.pyplot as plt

from wards import SHORT, K, load_pool


HERE = Path(__file__).parent
REF_IDX = SHORT.index("Ref")


def compute_metrics(Y_obs, Y_hat):
    """Return per-category list of (short, rmse_pp, r2, bias_pp, slope, intercept_pp, pearson)."""
    metrics = []
    for j in range(K):
        pred = Y_hat[:, j]
        obs = Y_obs[:, j]
        resid = obs - pred
        rmse = float(np.sqrt(np.mean(resid**2)) * 100)
        bias = float(np.mean(resid) * 100)
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((obs - obs.mean())**2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        if pred.std() > 1e-9:
            slope, intercept = np.polyfit(pred, obs, 1)
            pearson = float(np.corrcoef(pred, obs)[0, 1])
        else:
            slope, intercept, pearson = float("nan"), float("nan"), float("nan")
        metrics.append((SHORT[j], rmse, r2, bias, float(slope), float(intercept) * 100, pearson))
    return metrics


def print_metrics(label, metrics):
    print(f"\n=== {label} ===")
    print(f"{'Cell':>6} {'RMSE (pp)':>11} {'R^2':>8} {'Bias (pp)':>11} "
          f"{'BFslope':>10} {'BFint (pp)':>12} {'Pearson':>10}")
    for s, rmse, r2, bias, slope, intercept_pp, pearson in metrics:
        print(f"{s:>6} {rmse:>11.2f} {r2:>8.3f} {bias:>+11.2f} "
              f"{slope:>10.2f} {intercept_pp:>+12.2f} {pearson:>10.3f}")


def save_csv(metrics, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "rmse_pp", "r2", "bias_pp", "slope", "intercept_pp", "pearson"])
        for row in metrics:
            w.writerow([row[0],
                        f"{row[1]:.3f}", f"{row[2]:.4f}", f"{row[3]:+.3f}",
                        f"{row[4]:.4f}", f"{row[5]:+.3f}", f"{row[6]:.4f}"])
    print(f"Wrote {out_path}")


def plot_ppc(Y_obs, Y_hat, metrics, n_wards, title, subtitle, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=180)
    fig.patch.set_facecolor("white")
    axes = axes.flatten()
    for j in range(K):
        ax = axes[j]
        obs = Y_obs[:, j] * 100
        pred = Y_hat[:, j] * 100
        short, rmse, r2, _, slope, intercept_pp, pearson = metrics[j]
        ax.scatter(pred, obs, s=8, alpha=0.45, color="#1f77b4", edgecolor="none")
        lo = float(min(pred.min(), obs.min()))
        hi = float(max(pred.max(), obs.max()))
        pad = max(0.5, (hi - lo) * 0.05)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color="#888", linewidth=0.8, linestyle="--", zorder=1)
        xs = np.array([lo - pad, hi + pad])
        ax.plot(xs, slope * xs + intercept_pp, color="#d62728",
                linewidth=1.0, linestyle="-", zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(short, fontsize=12, fontweight="bold", pad=4)
        ax.text(0.04, 0.96,
                f"RMSE {rmse:.2f}pp\n$R^2$ {r2:.2f}\nslope {slope:.2f}\nPearson {pearson:.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="#444",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#ddd", alpha=0.9))
        ax.tick_params(labelsize=8)
        ax.grid(True, color="#eee", linewidth=0.5, zorder=0)
        for s in ax.spines.values():
            s.set_color("#ccc")
    axes[K].axis("off")
    fig.text(0.5, 0.02, "Predicted 2026 share-of-electorate (%)",
             ha="center", fontsize=10, color="#444")
    fig.text(0.02, 0.5, "Observed 2026 share-of-electorate (%)",
             va="center", rotation="vertical", fontsize=10, color="#444")
    fig.suptitle(f"{title} — fit across {n_wards} wards",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=9, color="#666")
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Load data + baseline posterior.
# -----------------------------------------------------------------------------
pool, X, Y = load_pool()
N = len(pool)


# -----------------------------------------------------------------------------
# Baseline PPC: y_hat_w = E[T] @ x_w.
# -----------------------------------------------------------------------------
idata = az.from_netcdf(HERE / "idata.nc")
T_mean = idata.posterior["T"].mean(("chain", "draw")).values
Y_hat = X @ T_mean
metrics = compute_metrics(Y, Y_hat)

print_metrics(f"Baseline PPC ({N} wards)", metrics)
save_csv(metrics, HERE / "ppc.csv")
plot_ppc(
    Y, Y_hat, metrics, N,
    title="Posterior predictive check — homogeneous T",
    subtitle="Each point is one ward. Red line: best-fit observed-on-predicted. Dashed grey: y=x.",
    out_path=HERE / "figures" / "ppc.png",
)


# -----------------------------------------------------------------------------
# Latent-model PPC (if outputs exist): y_hat_w = E[T] @ E[x_corr_w].
# -----------------------------------------------------------------------------
LATENT_IDATA = HERE / "idata_latent.nc"
if LATENT_IDATA.exists():
    idata_l = az.from_netcdf(LATENT_IDATA)
    T_mean_l = idata_l.posterior["T"].mean(("chain", "draw")).values
    sigma_sub_mean = idata_l.posterior["sigma_sub"].mean(("chain", "draw")).values
    sigma_full = np.zeros(K)
    for slot, j in enumerate([i for i in range(K) if i != REF_IDX]):
        sigma_full[j] = sigma_sub_mean[slot]
    lam_mean = idata_l.posterior["lam"].mean(("chain", "draw")).values

    stood = X[:, REF_IDX] > 0
    notstood_indices = np.where(~stood)[0]
    e_ref = np.zeros(K); e_ref[REF_IDX] = 1.0

    X_corr = X.copy()
    X_corr[notstood_indices] = X[notstood_indices] + lam_mean[:, None] * (e_ref[None, :] - sigma_full[None, :])
    Y_hat_l = X_corr @ T_mean_l
    metrics_l = compute_metrics(Y, Y_hat_l)

    print_metrics(f"Latent-Reform PPC ({N} wards)", metrics_l)
    save_csv(metrics_l, HERE / "ppc_latent.csv")
    plot_ppc(
        Y, Y_hat_l, metrics_l, N,
        title="PPC for the latent-Reform model",
        subtitle="Red line: best-fit observed-on-predicted. Dashed grey: y=x. Slope close to 1 = well-calibrated.",
        out_path=HERE / "figures" / "ppc_latent.png",
    )
