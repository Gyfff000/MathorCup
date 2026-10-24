"""AI-assisted implementation: OpenAI Codex / GPT-6, 2026-10-24."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import progress, sha256, write_json


def make_report(data, out, cfg):
    out = Path(out)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    progress("Report: preparing experiment summaries and scientific figures")
    metrics = pd.read_csv(out / "q1/metrics_summary.csv").set_index("model")
    per_target = pd.read_csv(out / "q1/metrics_by_target.csv")
    robustness = pd.read_csv(out / "q1/missingness_robustness.csv")
    top = pd.read_csv(out / "q2/top3_by_sensory_dimension.csv")
    jaccard = pd.read_csv(out / "q2/method_jaccard.csv")
    generation = pd.read_csv(out / "q3/generation_diagnostics.csv")
    transfers = pd.read_csv(out / "q3/transfer_diagnostics.csv")
    checks = pd.read_csv(out / "q3/defect_nonincrease_checks.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140, "savefig.facecolor": "white"})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    order = ["mean", "ridge", "pls", "extra_trees"]
    ax[0].bar(order, metrics.loc[order, "mae"], color=["#9b9b9b", "#286eaa", "#549778", "#bd833a"])
    ax[0].set(ylabel="Macro MAE (provided continuous scale)", title="Repeated out-of-fold prediction")
    byrate = robustness.groupby("additional_missing_fraction").mae.agg(["mean", "std"])
    ax[1].errorbar(byrate.index * 100, byrate["mean"], yerr=byrate["std"], marker="o", capsize=4, color="#286eaa")
    ax[1].set(xlabel="Additional missing cells (%)", ylabel="Fold MAE", title="Ridge missingness sensitivity")
    ax[1].text(.02, .98, "Bars: fold SD, not a confidence interval", transform=ax[1].transAxes, va="top", fontsize=8)
    fig.savefig(figures / "q1_validation.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    ax[0].hist(top.bootstrap_top3_frequency, bins=np.linspace(0, 1, 11), color="#286eaa", edgecolor="white")
    ax[0].set(xlabel="Bootstrap Top-3 inclusion frequency", ylabel="Number of selected factors", title="Stability of 150 selected factors")
    cols = ["graph_vs_ridge_jaccard", "graph_vs_spearman_jaccard", "ridge_vs_spearman_jaccard"]
    ax[1].boxplot([jaccard[c] for c in cols], tick_labels=["Graph / ridge", "Graph / rank corr.", "Ridge / rank corr."])
    ax[1].set(ylabel="Top-3 Jaccard", ylim=(-.05, 1.05), title="Agreement across methods")
    fig.savefig(figures / "q2_stability.png", dpi=170)
    plt.close(fig)
    portraits = pd.read_csv(out / "q3/generated_sensory_continuous.csv")
    selected = ["A_malt_all", "A_hops_all", "A_esters_all", "F_malt_all", "F_hops_all", "F_esters_all", "acidity", "bitternes", "sweetness", "body", "diacetyl", "dms", "stale_hops"]
    values = portraits.groupby("style")[selected].mean().reindex(cfg["generation_styles"])
    fig, ax = plt.subplots(figsize=(12, 3.4), layout="constrained")
    cap = max(.5, float(np.max(np.abs(values.to_numpy()))))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-cap, vmax=cap)
    ax.set_xticks(range(len(selected)), selected, rotation=45, ha="right")
    ax.set_yticks(range(len(values)), values.index)
    ax.set_title("Generated sensory portraits: model predictions on supplied continuous scale")
    fig.colorbar(im, ax=ax, label="Predicted continuous intensity", shrink=.85)
    fig.savefig(figures / "q3_sensory_portraits.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    x = np.arange(len(transfers))
    ax[0].plot(x, transfers.source_target_probability, "o--", color="#999999", label="Source")
    ax[0].plot(x, transfers.candidate_target_probability, "o-", color="#286eaa", label="Candidate")
    ax[0].set_xticks(x, transfers.beer.astype(str), rotation=45)
    ax[0].set(xlabel="Source beer", ylabel="Tripel probability (model only)", title="Candidate style movement")
    ax[0].legend()
    groups = checks.groupby("defect").nonincrease.agg(["sum", "count"])
    ax[1].barh(groups.index, groups["sum"] / groups["count"], color="#549778")
    ax[1].set(xlim=(0, 1.05), xlabel="Nonincrease fraction in checked quantity", title="Chemical measurements and stale-hops proxy")
    ax[1].text(.02, -.2, "Unchanged failures are included; actual tasting unverified", transform=ax[1].transAxes, fontsize=8)
    fig.savefig(figures / "q3_transfer_checks.png", dpi=170)
    plt.close(fig)
    facts = json.loads((out / "audit/verified_facts.json").read_text(encoding="utf-8"))
    q3status = json.loads((out / "q3/status.json").read_text(encoding="utf-8"))
    successful = int((transfers.status == "nontrivial_model_feasible_candidate").sum())
    heldout = transfers[transfers.generator_role == "holdout"]
    heldout_success = int((heldout.status == "nontrivial_model_feasible_candidate").sum())
    baseline = float(metrics.loc["mean", "mae"])
    ridge_mae = float(metrics.loc["ridge", "mae"])
    improvement = (baseline - ridge_mae) / baseline
    low = int((top.bootstrap_top3_frequency < .5).sum())
    report = f"""# A 题阶段成果报告

本阶段为 AI 辅助研究基线，三问均有可运行实现和阶段产物。数值只对应当前配置与当前附件。原始 0–6 分标定、真实感官改善及化学因果仍未完成验证，不能将整份报告直接作为最终参赛结论。

## 已核实的数据事实

实际有 {facts['n_samples']} 个样本、{facts['n_features']} 个化学或理化数值字段、{facts['n_targets']} 个连续感官目标和 {facts['n_styles']} 种风格。以 beer 唯一键一对一关联，不使用受损 beer_id。附件相关矩阵与重算 Spearman 的平均绝对差为 {facts['spearman_matrix_mean_absolute_difference']:.6g}，与 Pearson 为 {facts['pearson_matrix_mean_absolute_difference']:.6g}。逐字段变换证据、模板检查、哈希和字典保存在 audit/。这些是文件证据，关于合理性的解释仍需队内复核。

## 第一问 处理 方法与结果

输入为附件 1 的数值字段，目标为附件 4 的 50 个连续感官字段。样本编号、条码和风格标签均不作为化合物预测特征。训练折内拟合 1%—99% 截尾、缺失中位数与标准化。采用按风格分层的 {cfg['outer_folds']} 折、{cfg['outer_repeats']} 次重复外层验证，岭回归和 PLS 的超参数在外层训练数据内再做 {cfg['inner_folds']} 折选择。每个样本的重复留出预测取平均。预先指定岭回归作为下游参考，未根据外层成绩选择冠军再声称独立测试性能。

| 方法 | 宏平均 MAE | 宏平均 RMSE | 宏平均 R² |
|---|---:|---:|---:|
"""
    for name in order:
        v = metrics.loc[name]
        report += f"| {name} | {v.mae:.6f} | {v.rmse:.6f} | {v.r2:.6f} |\n"
    report += f"""
岭回归相对于训练均值参照的宏平均 MAE 变化为 {improvement:.2%} 的下降。该结果衡量给定连续标签的插值预测，不能当作真实 0–6 分精度，也不证明对全新风格的迁移能力。逐维误差、各风格误差、折分配和附加缺失压力实验均已保存。图中的折间标准差不是置信区间。

`ASSUMPTION_percentile_grades_0_6.csv` 采用训练折目标的七等频经验分位边界，只是有明确假设的参考等级。它不能恢复“未感知”等绝对评分语义，集中于同一值的目标也会出现重复分位边界。绝对标定仍为 UNRESOLVED。最终模型的全数据拟合系数仅供复现与下游画像，不用于报告样本外精度。

![第一问验证](figures/q1_validation.png)

## 第二问 处理 方法与结果

按全矩阵秩相关一致性核对附件 1 与 2 的变量别名。网络边保留绝对相关大于等于 {cfg['graph_threshold']} 的关系，保存符号与绝对权重。用符号拉普拉斯约束岭回归，求解 (ZᵀZ + αI + λL)B = Zᵀ(Y − 均值)。L 的非对角元素为负的有符号相关权，度数取绝对权和，因而为半正定。固定 α={cfg['graph_ridge_alpha']}、λ={cfg['graph_ridge_lambda']}；这是探索参数，尚未证明最优。

Top-3 仅从明确的化合物或元素字段中选择，排除 pH、色度、综合总量及混合指标。每个目标给出 3 项，共 {len(top)} 条，附方向、秩相关、网络模块、来源元数据及人工机理核查字段。{cfg['bootstrap_repeats']} 次样本 bootstrap 中重新拟合预处理和网络，再计算入选频率。与无图岭回归、边际秩相关 Top-3 比较 Jaccard。

当前选定因素中，入选频率低于 0.5 的有 {low}/{len(top)} 条，提示单个因素排序存在明显不稳定性。Graph/ridge 的平均 Jaccard 为 {jaccard.graph_vs_ridge_jaccard.mean():.4f}，Graph/Spearman 为 {jaccard.graph_vs_spearman_jaccard.mean():.4f}。这些差异应保留，不宜只展示一致的维度。来源与缺陷描述来自附件元数据，每个目标的具体化学机理论证仍需队员查阅原始文献并复核。全数据关联分析不构成独立预测验证或因果识别。

较高频候选包括 4VG 感官目标与 X4VG、金属味与 iron、烘烤麦芽感官与 methylpyrazine；它们仍只是当前重采样设计下的候选。某些目标排名出现反直觉化合物，说明模型和化学论证需要继续改进，不能把完整 Top-3 表当作已经确定的机理清单。

![第二问稳定性](figures/q2_stability.png)

## 第三问 处理 方法与结果

分层留出 20% 样本作为生成模型诊断参考。训练数据上拟合预处理和 {cfg['generation_components']} 维 PCA；各风格采用独立均值，协方差按 n/(n+10) 与全体训练协方差部分汇聚。采样后还原到附件特征尺度并限制在训练 1%—99% 范围，生成每种指定风格 {cfg['generation_samples_per_style']} 组化合物或理化描述符。全数据拟合的第一问参考模型输出感官画像，属于模型推断。

风格分类器在生成模型留出集上的平衡准确率为 {q3status['style_classifier_holdout_balanced_accuracy']:.4f}。该分类器与生成模型共享训练资料，并不是外部独立鉴定；其保留样本表现与生成物风格概率仅作模型诊断。Kriek、Lambic 等风格的真实留出样本很少，KS/MMD 指标无法证明完整分布已被可靠学会。KS 仅比较各字段实际非缺失的留出观测，字段数量另列；MMD 使用训练预处理后的表示。详见 generation_diagnostics.csv，并保留真实训练样本重采样的参照 MMD。

跨风格迁移先在潜空间施加风格均值差，再搜索连续插值候选。DMS、双乙酰、4VG 和 trans-2-nonenal 的附件字段逐项不得增加；Stale hops 没有直接化合物观测，以第一问的预测值不增加作为单独代理约束。候选还需使固定分类器的 Tripel 概率相对于原始样本以及仅做截尾的对照均增加超过 0.01，避免把异常清理效果混为风格迁移。未找到满足条件的非零改变候选时，保留原样并标为未找到，不计成功。

当前共有 {len(transfers)} 个 Pils/Lager 来源样本，找到非平凡模型内可行候选 {successful} 个；其中生成模型留出来源样本为 {len(heldout)} 个，找到候选 {heldout_success} 个。任何缺陷源值缺失、越出训练经验边界或仅能依赖预测代理的情况均在 transfer_diagnostics.csv 标注。所有字段检查保存在 defect_nonincrease_checks.csv。即使这些检查全部通过，也只能说明被检查的测量量或模型代理满足条件，不能声称实际品评无恶化。

![第三问感官画像](figures/q3_sensory_portraits.png)

![第三问迁移检查](figures/q3_transfer_checks.png)

## 保留结果与下一阶段

可保留：原件哈希和样本键审计、逐字段尺度证据、既定验证下的误差表、Top-3 与重采样频率、生成样本与诊断、逐项迁移约束核查。它们均绑定 config.json 与运行环境。

需要继续研究：真实评分标定锚点；未确定尺度与异常值的来源；目标相关的化学解释；网络与生成参数敏感性；新的独立验证；聚合指标的物理一致性；完整阈值和真实感官实验。原始研究或其他项目中的已发表成绩不属于本次实测结果。所有采纳的 AI 辅助成果须依赛事要求披露并由队内独立验证。

方法详见 docs/，来源详见 docs/SOURCES.md，运行日志详见 docs/RESEARCH_LOG.md。
"""
    (out / "STAGE_REPORT.md").write_text(report, encoding="utf-8")
    checksums = [{"file": str(p.relative_to(out)).replace("\\", "/"), "sha256": sha256(p)} for p in sorted(out.rglob("*")) if p.is_file() and "private" not in p.parts and p.suffix in {".csv", ".json", ".md", ".png"} and p.name not in ["result_checksums.json", "run_manifest.json"]]
    write_json(out / "result_checksums.json", checksums)
