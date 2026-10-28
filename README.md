# MathorCup 2026 A 题：啤酒风味预测

本项目整理啤酒化合物、历史感官评分、关键因素及风格迁移研究。代码和文字由 OpenAI Codex 辅助完成；参赛队员需要独立理解、复跑、核验并记录实际工作。项目用于研究参考，正式提交前应核对赛题、格式、匿名和 AI 使用规定。

## 从这里开始

| 需要什么 | 入口 |
|---|---|
| 最新论文 | [PDF](paper/latex_20261028/paper_stage5.pdf) · [LaTeX 源码](paper/latex_20261028/main.tex) · [模板与赛规适配](docs/期刊模板来源与赛事适配.md) |
| 论文问题与实验建议 | [审查报告](docs/review_20261028/论文问题与实验改进方案.pdf) · [可编辑内容](docs/review_20261028/论文问题与实验改进方案.md) |
| 目前结果与剩余工作 | [第四阶段研究报告](outputs/stage4/STAGE_REPORT.md) · [本轮改进记录](docs/交付与诊断记录_20261028.md) · [队内复核清单](docs/TEAM_REVIEW.md) |
| 原始 0—6 等级和预测 | [众数等级](outputs/stage4/ordinal/oof_modal_grades_0_6.csv) · [中位数等级](outputs/stage4/ordinal/oof_median_grades_0_6.csv) · [面板均分预测](outputs/stage4/prediction/oof_fixed_half_blend_0_6.csv) |
| 关键因素和迁移 | [50 项 Top-3](outputs/stage4/raw_graph/q2/top3_with_evidence.csv) · [冻结候选复核](outputs/stage4/transfer/frozen_stage3_candidates_raw_score_check.csv) · [浓度约束证书](outputs/review_20261028/mixture_certificates.csv) |
| 预测诊断 | [等级混淆表](outputs/stage5_diagnostics_20261028/modal_confusion.csv) · [逐目标诊断](outputs/stage5_diagnostics_20261028/modal_by_target.csv) · [分组覆盖](outputs/stage5_diagnostics_20261028/coverage_by_group.csv) |
| 化学异常与阈值联合复核 | [实验汇总](outputs/joint_20261028/chemical_sensitivity_summary.csv) · [逐目标结果](outputs/joint_20261028/chemical_sensitivity_by_target.csv) · [候选复核](outputs/joint_20261028/joint_candidate_evaluation.csv) · [角色汇总](outputs/joint_20261028/joint_summary_by_role.csv) |
| 非零等级先导 | [两阶段模型结果](outputs/stage5_two_stage_20261028/metrics.csv) · [非零命中率配对区间](outputs/stage5_two_stage_20261028/paired_nonzero_accuracy.csv) |

`outputs/stage1` 至 `stage3` 和旧论文是历史研究记录；第四阶段在公开逐人评分基础上重建了原始评分端点。不要把早期标准化尺度结果与原始 0—6 分结果直接比较。`outputs/review_20261028` 保存后续诊断先导；`outputs/stage5_diagnostics_20261028` 保存已有留出预测的描述性诊断。

## 最终模型：训练与无标签预测

Python 3.12，依赖版本见 `requirements-lock.txt`。先自行获取官方 A 题附件与作者公开数据，并完成第四阶段审计和分布构造。模型训练读取官方 `1.csv` 与已审计的历史面板结果；预测输入只需要与 `1.csv` 一致的 231 个数值字段，字段次序可不同、单元格可缺失，但不能缺列或更换单位和对数尺度。

```powershell
python run_final_model.py train --chemical "D:\your_data\A附件\1.csv" --model "local_models\panel_model.joblib"
python run_final_model.py predict --model "local_models\panel_model.joblib" --input "new_samples.csv" --out "local_predictions\run1"
```

训练输出包含预处理的可信本地模型和同名 `manifest.json`，后者记录字段、已核验的参考单位与变换、目标、数据哈希和版本。预测输出 `panel_mean.csv`、`modal_grade.csv`、`median_grade.csv` 和 `rating_distribution.csv`。模型文件仅应从可信来源加载，`joblib` 文件不上传仓库。训练用已导出的历史评分表保留有限小数精度；模型是全数据拟合，精度仍应引用按酒留出实验，而非训练样本演示。

## 复现与论文编译

```powershell
python scripts/fetch_stage4_sources.py
python run_stage4.py all --data-dir "D:\your_data\A附件" --config config_stage4.json --out "outputs\my_run"
python scripts/joint_review_experiments.py --data-dir "D:\your_data\A附件"
python -m pytest -q
cd paper\latex_20261028
xelatex -jobname=paper_stage5 -interaction=nonstopmode -halt-on-error main.tex
xelatex -jobname=paper_stage5 -interaction=nonstopmode -halt-on-error main.tex
```

第四阶段生成依赖第三阶段现有输出，路径可用 `--stage3` 指定；`--config` 和 `--out` 可以隔离实验。非空输出目录会被拒绝，只有显式 `--allow-overwrite` 才允许覆盖。重新构造旧生成谱需要先运行第三阶段。保形区间、模型选择和实际执行参数见运行记录；旧版结果文件保留原有指标名称，`outputs/review_20261028/ordinal_metrics.csv` 给出修正后的分布距离和期望个人评分。

LaTeX 依赖 XeLaTeX 和中文字体；`main.tex` 是人工编辑入口。`scripts/build_latex_revision.py` 会由阶段 Markdown 重新生成并覆盖它，使用前需先保存人工修改。官方附件及原始逐人记录不放入公开仓库。

## 尚待完成

非零等级的整体计算预算与校准对照、缺失机制和完整参数重选稳定性仍需进一步实验。已增加化学异常逐单元屏蔽敏感性及阈值候选的冻结代理复核，均基于历史数据。检测限与完整化学标定、新批次制备和盲评需要真实实验条件。队员独立复核记录、AI 使用详情与最终提交材料由队伍按实际情况完成。

