"""Estimate the 2024 -> 2026 UK local-election voter transition matrix.

Outputs:
    transition_matrix.csv        long-form posterior summary, one row per (i, j)
    transition_matrix.meta.json  sample size + mean 2024 cohort shares + prior
    idata.nc                     full posterior trace (ArviZ NetCDF)

See README.md for the full write-up
"""

import csv
import json
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
from wards import CATS, SHORT, K, load_pool

HERE = Path(__file__).parent


# -----------------------------------------------------------------------------
# 1. Load the comparable ward pool.
# -----------------------------------------------------------------------------
pool, X, Y = load_pool()
print(f"N wards in pool: {len(pool)}")
print(f"Mean 2024 shares: {dict(zip(SHORT, X.mean(0).round(3)))}")
print(f"Mean 2026 shares: {dict(zip(SHORT, Y.mean(0).round(3)))}")


# -----------------------------------------------------------------------------
# 2. The Bayesian model.
# -----------------------------------------------------------------------------
# Each row of T gets a Dirichlet prior. The prior structure has two pieces.
#
# (a) A single shared retention prior of 70% applied to every row -- including
#     DNV. Anchored to Fieldhouse et al. (2020) and Mellon (2021), which find
#     typical UK 2-year retention in the 65-80% range. Implemented as a
#     Dirichlet with effective sample size ~35 voters per row. Applied
#     symmetrically to avoid pre-baking party-specific assumptions.
#
# (b) Each non-DNV row's flow into DNV is set to a prior mean of ~8.5%,
#     encoding bidirectional turnout churn. UK GE 2019->2024 saw a 7.6pp
#     net turnout drop (House of Commons Library) -- gross drop-out is
#     necessarily larger. Fieldhouse et al. (2020) document pronounced
#     individual-level turnout volatility in UK panel data.
#
# Likelihood: y_w ~ Normal(T @ x_w, sigma_obs).
RETENTION_PRIOR_MEAN = 0.70
PARTY_TO_DNV_MEAN = 0.085
ESS_NON_DNV = 35.0
DNV_IDX = K - 1

alpha_T = np.zeros((K, K))
# Non-DNV rows: retention on the diagonal, party->DNV bump, others spread evenly.
for i in range(K):
    if i == DNV_IDX:
        continue
    other_mean = (1 - RETENTION_PRIOR_MEAN - PARTY_TO_DNV_MEAN) / (K - 2)
    alpha_T[i, :] = other_mean * ESS_NON_DNV
    alpha_T[i, i] = RETENTION_PRIOR_MEAN * ESS_NON_DNV
    alpha_T[i, DNV_IDX] = PARTY_TO_DNV_MEAN * ESS_NON_DNV

# DNV row: retention on the diagonal (DNV->DNV), no separate party->DNV bump
# because that's already on the diagonal here.
other_mean = (1 - RETENTION_PRIOR_MEAN) / (K - 1)
alpha_T[DNV_IDX, :] = other_mean * ESS_NON_DNV
alpha_T[DNV_IDX, DNV_IDX] = RETENTION_PRIOR_MEAN * ESS_NON_DNV

with pm.Model() as model:
    T = pm.Dirichlet("T", a=alpha_T, shape=(K, K))
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.1)
    pm.Normal("obs", mu=pm.math.dot(X, T), sigma=sigma_obs, observed=Y)

    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=42,
        progressbar=False,
    )


# -----------------------------------------------------------------------------
# 3. Summarise + sanity-check the posterior.
# -----------------------------------------------------------------------------
T_samples = idata.posterior["T"].stack(sample=("chain", "draw")).values
T_mean = T_samples.mean(axis=2)
T_lo = np.percentile(T_samples, 10, axis=2)
T_hi = np.percentile(T_samples, 90, axis=2)

print("\nPosterior transition matrix (rows = 2024 -> cols = 2026)")
print("Each cell: posterior mean% [80% credible interval]")
print()
print("       " + "".join(f"{c:>20}" for c in SHORT))
for i, c in enumerate(SHORT):
    row = f"{c:>5}: "
    for j in range(K):
        row += f"{T_mean[i, j] * 100:>5.0f} ([{T_lo[i, j] * 100:>3.0f},{T_hi[i, j] * 100:>3.0f}]) "
    print(row)

summary = az.summary(idata, var_names=["T"], round_to=4)
print(f"\nMax r_hat across T cells: {summary['r_hat'].max():.3f}  (want <= 1.01)")
print(f"Min ESS bulk:             {summary['ess_bulk'].min():.0f}  (want >= 400)")
print(f"Divergences:              {idata.sample_stats.diverging.sum().item()}")


# -----------------------------------------------------------------------------
# 4. Save outputs.
# -----------------------------------------------------------------------------
with open(HERE / "transition_matrix.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        ["from_2024", "to_2026", "post_mean", "post_p10", "post_p90", "ci_width"]
    )
    for i in range(K):
        for j in range(K):
            w.writerow(
                [
                    CATS[i],
                    CATS[j],
                    f"{T_mean[i, j]:.4f}",
                    f"{T_lo[i, j]:.4f}",
                    f"{T_hi[i, j]:.4f}",
                    f"{T_hi[i, j] - T_lo[i, j]:.4f}",
                ]
            )

with open(HERE / "transition_matrix.meta.json", "w") as f:
    json.dump(
        {
            "n_wards": len(pool),
            "x_mean": {SHORT[i]: float(X.mean(0)[i]) for i in range(K)},
            "prior": {
                "retention_prior_mean": RETENTION_PRIOR_MEAN,
                "party_to_dnv_mean": PARTY_TO_DNV_MEAN,
                "ess_non_dnv": ESS_NON_DNV,
            },
        },
        f,
        indent=2,
    )

idata.to_netcdf(HERE / "idata.nc")
print("\nWrote transition_matrix.csv, transition_matrix.meta.json, idata.nc")
