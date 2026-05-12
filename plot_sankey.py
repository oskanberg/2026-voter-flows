"""Render the transition matrix as Sankey diagrams.

Produces:
    figures/sankey_full.png         all flows from the baseline pool fit
    figures/sankey_full_latent.png  all flows from the latent-Reform fit
                                    (if transition_matrix_latent.csv exists)

Flow widths are share-of-electorate magnitudes (T[i,j] * mean 2024 share[i]),
so visual mass reflects actual voter populations. Tiny flows are suppressed.
"""

import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from wards import CATS, SHORT, K, load_pool

HERE = Path(__file__).parent
REF_IDX = SHORT.index("Ref")

COLORS = {
    "Lab": "#E4003B",
    "Con": "#0087DC",
    "LD": "#FAA61A",
    "Grn": "#6AB023",
    "Ref": "#12B6CF",
    "Oth": "#9C9C9C",
    "DNV": "#4A4A4A",
}
MIN_FLOW = 0.003  # share-of-electorate threshold below which we suppress flows


def hex_to_rgba(h, alpha):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def build_sankey(T_use, x_mean_use, title, subtitle, footer):
    """Render a Sankey from a transition matrix and 2024 cohort means.

    Omits the "Other" bucket from the rendering (its flows are small and
    visually cluttering); the model is unchanged.
    """
    order = [s for s in SHORT if s != "Oth"]
    pos = {s: i for i, s in enumerate(order)}
    n = len(order)
    labels = list(order) + list(order)
    node_colors = [COLORS[s] for s in order] * 2

    sources, targets, values, link_colors = [], [], [], []
    for i in range(K):
        if SHORT[i] not in pos:
            continue
        for j in range(K):
            if SHORT[j] not in pos:
                continue
            flow = x_mean_use[i] * T_use[i, j]
            if flow < MIN_FLOW:
                continue
            sources.append(pos[SHORT[i]])
            targets.append(n + pos[SHORT[j]])
            values.append(flow * 100)
            alpha = 0.3 if i == j else 0.6
            link_colors.append(hex_to_rgba(COLORS[SHORT[i]], alpha))

    fig = go.Figure(
        go.Sankey(
            arrangement="perpendicular",
            node=dict(
                label=labels,
                color=node_colors,
                pad=24,
                thickness=22,
                line=dict(color="rgba(0,0,0,0)", width=0),
                hovertemplate="%{label}<br>%{value:.1f}pp of electorate<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate="%{source.label} → %{target.label}<br>%{value:.1f}pp<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.96,
            font=dict(
                family="Helvetica Neue, Arial, sans-serif", size=22, color="#1a1a1a"
            ),
        ),
        font=dict(family="Helvetica Neue, Arial, sans-serif", size=14, color="#1a1a1a"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=80, r=100, t=120, b=80),
        width=950,
        height=720,
        annotations=[
            dict(
                text=subtitle,
                x=0.5,
                y=1.06,
                xref="paper",
                yref="paper",
                xanchor="center",
                showarrow=False,
                font=dict(size=13, color="#555"),
            ),
            dict(
                text=footer,
                x=0.5,
                y=-0.05,
                xref="paper",
                yref="paper",
                xanchor="center",
                showarrow=False,
                font=dict(size=10, color="#888"),
            ),
        ],
    )
    return fig


def load_T(csv_path):
    """Read a transition_matrix*.csv into a (K, K) numpy array of post means."""
    T = np.zeros((K, K))
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            T[CATS.index(r["from_2024"]), CATS.index(r["to_2026"])] = float(
                r["post_mean"]
            )
    return T


# -----------------------------------------------------------------------------
# Load data and transition matrices.
# -----------------------------------------------------------------------------
pool, X, _ = load_pool()
x_mean = X.mean(0)
T_main = load_T(HERE / "transition_matrix.csv")
with open(HERE / "transition_matrix.meta.json") as f:
    n_wards = json.load(f)["n_wards"]


# -----------------------------------------------------------------------------
# Baseline sankey.
# -----------------------------------------------------------------------------
fig = build_sankey(
    T_main,
    x_mean,
    title="Voter flows, UK local elections 2024 → 2026",
    subtitle="",
    footer=(
        f"Bayes RxC model · n={n_wards} wards · "
        f"flows < {MIN_FLOW * 100:.1f}pp suppressed · minor parties (Oth) omitted"
    ),
)
out = HERE / "figures" / "sankey_full.png"
out.parent.mkdir(exist_ok=True)
fig.write_html(out.with_suffix(".html"), include_plotlyjs="cdn")
fig.write_image(out, scale=2)
print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Latent-model sankey (if outputs exist). The corrected cohort sizes pool the
# observed x_w with the latent λ_w correction: for each not-stood ward, λ_w
# is added to Reform and λ_w·σ is subtracted from the other categories.
# Aggregating with the pool-mean λ̄ across not-stood wards gives a faithful
# reconstruction of the latent model's implied 2024 cohorts.
# -----------------------------------------------------------------------------
LATENT_CSV = HERE / "transition_matrix_latent.csv"
LATENT_META = HERE / "transition_matrix_latent.meta.json"
if LATENT_CSV.exists() and LATENT_META.exists():
    T_latent = load_T(LATENT_CSV)
    with open(LATENT_META) as f:
        meta_l = json.load(f)

    sigma_full = np.zeros(K)
    for slot, value in meta_l["sigma_substitution_mean"].items():
        sigma_full[SHORT.index(slot)] = value
    e_ref = np.zeros(K)
    e_ref[REF_IDX] = 1.0
    mean_correction = (
        meta_l["n_notstood_2024"]
        / meta_l["n_wards"]
        * meta_l["lambda_posterior_mean_overall"]
        * (e_ref - sigma_full)
    )
    x_mean_latent = x_mean + mean_correction

    fig = build_sankey(
        T_latent,
        x_mean_latent,
        title="Voter flows, 2024 intent → 2026 vote",
        subtitle="Latent-Reform model: 2024 Reform cohort includes voters who would have chosen Reform had a candidate stood",
        footer=(
            f"Latent-Reform Bayes RxC · n={meta_l['n_wards']} wards "
            f"(2024 Reform cohort reconstructed in {meta_l['n_notstood_2024']} not-stood wards) · "
            f"flows < {MIN_FLOW * 100:.1f}pp suppressed · minor parties (Oth) omitted"
        ),
    )
    out = HERE / "figures" / "sankey_full_latent.png"
    fig.write_html(out.with_suffix(".html"), include_plotlyjs="cdn")
    fig.write_image(out, scale=2)
    print(f"Wrote {out}")
