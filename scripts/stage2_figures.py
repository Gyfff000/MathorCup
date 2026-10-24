"""AI-assisted scientific plots: OpenAI Codex / GPT-6, 2026-10-24."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage2"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})

def finish(fig, name):
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close(fig)

metrics = pd.read_csv(OUT / "q1/metrics_summary.csv").set_index("model")
names = ["mean", "ridge", "pls", "extra_trees", "gbr_fixed", "fixed_ridge_tree_blend"]
fig, ax = plt.subplots(figsize=(9.5, 4), layout="constrained")
ax.bar(np.arange(6), metrics.loc[names, "mae"], color=["#999999", "#3974a6", "#549978", "#bb8439", "#a45f87", "#728c99"])
ax.set_xticks(range(6), ["Mean", "Ridge", "PLS", "ExtraTrees", "GBR", "Fixed blend"])
ax.set(ylabel="Macro MAE on supplied continuous targets", title="Identical repeated held-out splits")
finish(fig, "model_comparison.png")

g = pd.read_csv(OUT / "q1/grade_rule_sensitivity.csv")
c = pd.read_csv(OUT / "q1/conformal_coverage.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
ax[0].bar(np.arange(6) + 1, 100 * g.any_rule_disagreement_fraction, color="#3974a6")
ax[0].set(xlabel="Held-out fold", ylabel="Disagreement across grading assumptions (%)", ylim=(0, 100))
ax[1].hist(c.empirical_test_coverage, bins=np.linspace(.6, 1, 9), color="#549978", edgecolor="white")
ax[1].axvline(.9, ls="--", color="black", label="Nominal marginal coverage")
ax[1].set(xlabel="Empirical coverage (50 test beers)", ylabel="Number of sensory targets")
ax[1].legend(fontsize=8)
finish(fig, "calibration_uncertainty.png")

t = pd.read_csv(OUT / "q2/top3_evidence_review.csv")
good = t.evidence_status == "comparatively_supported_association"
fig, ax = plt.subplots(figsize=(9.5, 4), layout="constrained")
ax.scatter(t.loc[~good, "bootstrap_top3_frequency"], t.loc[~good, "heldout_permutation_MAE_increase"], s=23, alpha=.65, label="Weak or unstable")
ax.scatter(t.loc[good, "bootstrap_top3_frequency"], t.loc[good, "heldout_permutation_MAE_increase"], s=45, color="#b26044", label="Passed current descriptive criteria")
ax.axhline(0, color="grey", lw=.8)
ax.axvline(.5, color="grey", lw=.8, ls="--")
ax.set(xlabel="Bootstrap Top-3 frequency", ylabel="Held-out MAE increase after permutation", title="Association evidence for 150 selected factors")
ax.legend(fontsize=8)
finish(fig, "factor_evidence.png")

d = pd.read_csv(OUT / "q3/generator_comparison_by_split.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
styles = ["Tripel", "Kriek", "Lambic"]
methods = ["bootstrap", "local_convex", "PCA_Gaussian"]
for j, method in enumerate(methods):
    group = d[d.method == method].groupby("style").agg(mmd=("MMD2", "mean"), sd=("MMD2", "std"), prob=("target_style_probability", "mean")).reindex(styles)
    xx = np.arange(3) + (j - 1) * .23
    ax[0].bar(xx, group.mmd, width=.22, yerr=group.sd, capsize=3, label=method)
    ax[1].bar(xx, group.prob, width=.22, label=method)
for a in ax:
    a.set_xticks(range(3), styles)
ax[0].set(ylabel="MMD squared (mean and split SD)", title="Held-out distribution distance")
ax[1].set(ylabel="Target-style model probability", title="Internal classifier diagnostic", ylim=(0, 1.05))
ax[0].legend(fontsize=7)
finish(fig, "generator_comparison.png")

a = pd.read_csv(ROOT / "outputs/stage1/q3/transfer_diagnostics.csv").set_index("beer")
b = pd.read_csv(OUT / "q3/expanded_transfer_diagnostics.csv")
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
xx = np.arange(len(b))
ax.plot(xx, b.source_trip_probability, "o--", color="#999999", label="Source")
ax.plot(xx, a.loc[b.beer, "candidate_target_probability"], "s-", color="#3974a6", label="PCA direction search")
ax.plot(xx, b.candidate_trip_probability, "^-", color="#b26044", label="Expanded conditional-anchor search")
for i, row in b.iterrows():
    if not row.all_source_defects_observed:
        ax.annotate("missing source diacetyl", (i, row.candidate_trip_probability), xytext=(i-.8, .72-.06*(i%2)), fontsize=7, arrowprops={"arrowstyle":"-", "lw":.5})
ax.set_xticks(xx, b.beer.astype(str), rotation=45)
ax.set(xlabel="Source Pils/Lager beer", ylabel="Tripel classifier probability", ylim=(0, 1.09), title="Model-based transfer; actual sensory response remains unverified")
ax.legend(fontsize=8, loc="center right")
finish(fig, "transfer_comparison.png")

counts = pd.read_csv(OUT / "audit/style_counts.csv").sort_values("n")
fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
ax.barh(counts.tasting_category_fine, counts.n, color="#3974a6")
ax.set(xlabel="Observed beers", title="Small and unbalanced style samples")
finish(fig, "style_counts.png")
print("Created six stage-two figures")
