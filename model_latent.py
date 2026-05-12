"""Reform-aware extension of the homogeneous transition model.

Outputs:
    transition_matrix_latent.csv         posterior summary for T
    transition_matrix_latent.meta.json   prior + sigma posterior + lambda summary
    idata_latent.nc                      full posterior trace

(Figures: ppc.py and plot_heatmap.py both pick up the latent-model outputs
automatically and generate ppc_latent.png and heatmap_latent.png.)
"""

import csv
import json
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from wards import CATS, SHORT, K, load_pool

HERE = Path(__file__).parent
REF_IDX = SHORT.index("Ref")
DNV_IDX = SHORT.index("DNV")


# -----------------------------------------------------------------------------
# 1. Load the comparable ward pool.
# -----------------------------------------------------------------------------
pool, X, Y = load_pool()
N = len(pool)


# -----------------------------------------------------------------------------
# 2. Identify the candidate-availability gap.
# -----------------------------------------------------------------------------
stood = X[:, REF_IDX] > 0
n_stood = int(stood.sum())
n_notstood = int((~stood).sum())
notstood_indices = np.where(~stood)[0]

# Observed Reform share in stood wards, kept for reporting only.
stood_ref = X[stood, REF_IDX]
m_ref = float(stood_ref.mean())
v_ref = float(stood_ref.var())

print(f"N wards: {N}  (Reform stood 2024: {n_stood} | not stood: {n_notstood})")
print(
    f"Reform share in stood wards: mean={m_ref * 100:.2f}%  sd={np.sqrt(v_ref) * 100:.2f}%"
)
print(f"lambda_w prior: HalfNormal(sigma=0.05)")


# -----------------------------------------------------------------------------
# 3. T prior — identical to baseline.
# -----------------------------------------------------------------------------
RETENTION_PRIOR_MEAN = 0.70
PARTY_TO_DNV_MEAN = 0.085
ESS_NON_DNV = 35.0

alpha_T = np.zeros((K, K))
for i in range(K):
    if i == DNV_IDX:
        continue
    other_mean = (1 - RETENTION_PRIOR_MEAN - PARTY_TO_DNV_MEAN) / (K - 2)
    alpha_T[i, :] = other_mean * ESS_NON_DNV
    alpha_T[i, i] = RETENTION_PRIOR_MEAN * ESS_NON_DNV
    alpha_T[i, DNV_IDX] = PARTY_TO_DNV_MEAN * ESS_NON_DNV
other_mean = (1 - RETENTION_PRIOR_MEAN) / (K - 1)
alpha_T[DNV_IDX, :] = other_mean * ESS_NON_DNV
alpha_T[DNV_IDX, DNV_IDX] = RETENTION_PRIOR_MEAN * ESS_NON_DNV


# -----------------------------------------------------------------------------
# 4. The Bayesian model.
# -----------------------------------------------------------------------------
e_ref = np.zeros(K)
e_ref[REF_IDX] = 1.0

with pm.Model() as model:
    T = pm.Dirichlet("T", a=alpha_T, shape=(K, K))

    # sigma over the 6 non-Reform categories. Then embed into a 7-vector with
    # 0 in the Reform slot so we can write the correction in vector form.
    sigma_sub = pm.Dirichlet("sigma_sub", a=np.ones(K - 1), shape=K - 1)
    sigma_full = pt.concatenate([sigma_sub[:REF_IDX], pt.zeros(1), sigma_sub[REF_IDX:]])

    # Per-ward latent Reform support, only for not-stood wards. HalfNormal
    # with sigma=0.05 — implied mean ~4%, sd ~3%. Empirical sweet spot:
    # loose enough to let per-ward lambda_w range from 0.6% to 34%, tight
    # enough to keep the model in the high-Reform-retention mode (Ref->Ref
    # ~92%). Looser priors drift the model toward a substantively
    # implausible low-retention mode (see header docstring).
    lam_notstood = pm.HalfNormal("lam", sigma=0.05, shape=n_notstood)

    # Scatter lambda_notstood into a length-N tensor (zeros at stood positions).
    lam_full = pt.set_subtensor(pt.zeros(N)[notstood_indices], lam_notstood)

    # Corrected 2024 input: x_corr_w = x_w + lambda_w * (e_Ref - sigma).
    correction = lam_full[:, None] * (e_ref[None, :] - sigma_full[None, :])
    X_corr = X + correction

    sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.1)
    pm.Normal("obs", mu=pm.math.dot(X_corr, T), sigma=sigma_obs, observed=Y)

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
# 5. Summarise.
# -----------------------------------------------------------------------------
T_samples = idata.posterior["T"].stack(sample=("chain", "draw")).values
T_mean = T_samples.mean(axis=2)
T_lo = np.percentile(T_samples, 10, axis=2)
T_hi = np.percentile(T_samples, 90, axis=2)

sigma_sub_mean = idata.posterior["sigma_sub"].mean(("chain", "draw")).values
sigma_full_mean = np.zeros(K)
for slot, j in enumerate([i for i in range(K) if i != REF_IDX]):
    sigma_full_mean[j] = sigma_sub_mean[slot]

lam_post = idata.posterior["lam"].stack(sample=("chain", "draw")).values
lam_mean_per_ward = lam_post.mean(axis=1)

print("\n=== sigma (substitution distribution for hidden 2024 Reform voters) ===")
for j in range(K):
    if j == REF_IDX:
        continue
    print(f"   {SHORT[j]:>4}: {sigma_full_mean[j] * 100:>5.1f}%")
print(
    f"\nMean latent lambda across not-stood wards: {lam_mean_per_ward.mean() * 100:.2f}%"
)
print(
    f"  (per-ward range {lam_mean_per_ward.min() * 100:.2f}% to {lam_mean_per_ward.max() * 100:.2f}%)"
)

print("\nPosterior T (rows = 2024 -> cols = 2026)")
print("       " + "".join(f"{c:>20}" for c in SHORT))
for i, c in enumerate(SHORT):
    row = f"{c:>5}: "
    for j in range(K):
        row += f"{T_mean[i, j] * 100:>5.0f} ([{T_lo[i, j] * 100:>3.0f},{T_hi[i, j] * 100:>3.0f}]) "
    print(row)

summary = az.summary(idata, var_names=["T", "sigma_sub"], round_to=4)
print(f"\nMax r_hat (T, sigma): {summary['r_hat'].max():.3f}  (want <= 1.01)")
print(f"Min ESS bulk:         {summary['ess_bulk'].min():.0f}  (want >= 400)")
print(f"Divergences:          {idata.sample_stats.diverging.sum().item()}")


# -----------------------------------------------------------------------------
# 6. Save outputs.
# -----------------------------------------------------------------------------
with open(HERE / "transition_matrix_latent.csv", "w", newline="") as f:
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

with open(HERE / "transition_matrix_latent.meta.json", "w") as f:
    json.dump(
        {
            "n_wards": N,
            "n_stood_2024": n_stood,
            "n_notstood_2024": n_notstood,
            "x_mean": {SHORT[i]: float(X.mean(0)[i]) for i in range(K)},
            "T_prior": {
                "retention_prior_mean": RETENTION_PRIOR_MEAN,
                "party_to_dnv_mean": PARTY_TO_DNV_MEAN,
                "ess_non_dnv": ESS_NON_DNV,
            },
            "lambda_prior": {"type": "HalfNormal", "sigma": 0.05},
            "stood_ref_share_mean": m_ref,
            "sigma_substitution_mean": {
                SHORT[j]: float(sigma_full_mean[j]) for j in range(K) if j != REF_IDX
            },
            "lambda_posterior_mean_overall": float(lam_mean_per_ward.mean()),
        },
        f,
        indent=2,
    )

idata.to_netcdf(HERE / "idata_latent.nc")
print(
    "\nWrote transition_matrix_latent.csv, transition_matrix_latent.meta.json, idata_latent.nc"
)
