"""Render the transition matrix as a heatmap.

Outputs:
    figures/heatmap.png         baseline T
    figures/heatmap_latent.png  latent-model T (if available)
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

HERE = Path(__file__).parent

CATS = [
    "Labour Party",
    "Conservative and Unionist Party",
    "Liberal Democrats",
    "Green Party",
    "Reform UK",
    "Other",
    "Abstain",
]
SHORT = ["Lab", "Con", "LD", "Grn", "Ref", "Oth", "DNV"]
COLORS = {
    "Lab": "#E4003B",
    "Con": "#0087DC",
    "LD": "#FAA61A",
    "Grn": "#6AB023",
    "Ref": "#12B6CF",
    "Oth": "#9C9C9C",
    "DNV": "#4A4A4A",
}
K = len(SHORT)


def load_T(csv_path):
    """Read a transition_matrix*.csv into a (K, K) numpy array of post means."""
    T = np.zeros((K, K))
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            T[CATS.index(r["from_2024"]), CATS.index(r["to_2026"])] = float(
                r["post_mean"]
            )
    return T


def plot_heatmap(T, title, footer, out_path):
    """Render and save one heatmap."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=180)
    fig.patch.set_facecolor("white")
    WHITE = np.array([1.0, 1.0, 1.0])
    GAMMA = 0.85  # gently emphasises small flows

    for i in range(K):
        party_rgb = np.array(to_rgb(COLORS[SHORT[i]]))
        for j in range(K):
            p = T[i, j]
            intensity = p**GAMMA
            cell_rgb = WHITE * (1 - intensity) + party_rgb * intensity
            ax.add_patch(
                plt.Rectangle(
                    (j, K - 1 - i),
                    1,
                    1,
                    facecolor=cell_rgb,
                    edgecolor="white",
                    linewidth=2,
                )
            )
            text_color = "white" if intensity > 0.55 else "#1a1a1a"
            if p < 0.005:
                label, text_color = ".", "#bbb"
            elif p < 0.01:
                label = "<1%"
            else:
                label = f"{p * 100:.0f}%"
            ax.text(
                j + 0.5,
                K - 1 - i + 0.5,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
                fontweight="bold" if p >= 0.1 else "normal",
                family="DejaVu Sans",
            )

    ax.set_xlim(0, K)
    ax.set_ylim(0, K)
    ax.set_aspect("equal")
    ax.set_xticks([j + 0.5 for j in range(K)])
    ax.set_xticklabels(SHORT, fontsize=12, family="DejaVu Sans")
    ax.set_yticks([K - 1 - i + 0.5 for i in range(K)])
    ax.set_yticklabels(SHORT, fontsize=12, family="DejaVu Sans")
    ax.tick_params(axis="both", which="both", length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(
        "voted in 2026", fontsize=13, color="#444", labelpad=12, family="DejaVu Sans"
    )
    ax.set_ylabel(
        "voted in 2024", fontsize=13, color="#444", labelpad=12, family="DejaVu Sans"
    )
    ax.xaxis.set_label_position("bottom")

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
        family="DejaVu Sans",
        color="#1a1a1a",
        y=0.97,
    )
    fig.text(
        0.5,
        0.92,
        "Posterior-mean P(2026 = j | 2024 = i) — rows sum to 100%",
        ha="center",
        fontsize=12,
        color="#555",
        family="DejaVu Sans",
    )
    fig.text(
        0.5, 0.04, footer, ha="center", fontsize=10, color="#888", family="DejaVu Sans"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.9])
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")
    plt.close(fig)


# Baseline heatmap.
with open(HERE / "transition_matrix.meta.json") as f:
    n_wards = json.load(f)["n_wards"]
T = load_T(HERE / "transition_matrix.csv")
plot_heatmap(
    T,
    title="Estimated voter transitions, 2024 → 2026",
    footer=f"Bayes RxC model · n={n_wards} wards · cells tinted by 2024 source-party colour",
    out_path=HERE / "figures" / "heatmap.png",
)


# Latent-model heatmap (if outputs exist).
LATENT_CSV = HERE / "transition_matrix_latent.csv"
LATENT_META = HERE / "transition_matrix_latent.meta.json"
if LATENT_CSV.exists() and LATENT_META.exists():
    with open(LATENT_META) as f:
        meta_l = json.load(f)
    T_l = load_T(LATENT_CSV)
    plot_heatmap(
        T_l,
        title="Voter transitions under the latent-Reform model",
        footer=(
            f"Bayes RxC model with Reform-correction · n={meta_l['n_wards']} wards "
            f"({meta_l['n_stood_2024']} stood / {meta_l['n_notstood_2024']} not stood in 2024)"
        ),
        out_path=HERE / "figures" / "heatmap_latent.png",
    )
