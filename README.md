# MathorCup 2026 A 题研究记录

围绕啤酒化合物与感官数据建立三问的可复现研究基线，保存文件证据、处理决定、模型实验、限制和阶段结果。本仓库由 OpenAI Codex 辅助建立，核心研究判断、最终算法选择和竞赛成果须由参赛队员独立完成与验证。详见 [AI 使用记录](docs/AI_USAGE.md)。

当前三问均可运行。第四阶段已补入作者v4逐人员原始评分，重建并交叉核验250款酒的匿名对应，提供真实0–6评分分布的整数等级、原始面板均分和预测区间；第二问重新运行原始端点图对照与全部Top-3；第三问补齐原始评分画像、冻结候选复查及浓度混合检验。**完整化学测量认证和新实物品评仍未完成，条件、失败和弱项均保留。**

## 从哪里开始阅读

- **最新：原始评分完善版**：[论文PDF（含完整源程序）](paper/revised_absolute_20261025/论文完善稿.pdf) · [可编辑正文](paper/revised_absolute_20261025/论文完善稿.md) · [AI使用更新说明](paper/revised_absolute_20261025/AI工具使用详情说明_更新版.pdf) · [第四阶段报告](outputs/stage4/STAGE_REPORT.md) · [完整实施记录](docs/第四阶段原始评分与论文完善记录.md)。48项测试通过；原始评分数据路径已实质解决，来源重建边界和未完成实测继续说明。
- [0–6整数众数等级](outputs/stage4/ordinal/oof_modal_grades_0_6.csv) · [中位数等级](outputs/stage4/ordinal/oof_median_grades_0_6.csv) · [七级概率与实测分布](outputs/stage4/ordinal/oof_rating_distributions.csv) · [原始面板均分](outputs/stage4/audit/observed_panel_means_0_6.csv) · [均分OOF预测](outputs/stage4/prediction/oof_fixed_half_blend_0_6.csv)。众数、中位数与均分含义不同，不能把概率未经校准或总体一致率当精细等级保证。
- [完整原始端点Top-3](outputs/stage4/raw_graph/q2/top3_with_evidence.csv) · [浓度例外](outputs/stage4/audit/chemical_source_exceptions.csv) · [真实品评方案](docs/真实品评与化学复核实施方案.md)。方案与空表不是已完成实测。
- **历史：第三阶段修订版**：[论文PDF](paper/revised_20261025/论文修订稿.pdf) · [正文](paper/revised_20261025/论文修订稿.md) · [修订说明](docs/论文修订说明_20261025.md) · [第三阶段报告](outputs/stage3/STAGE_REPORT.md)。保留当时40项检查和未取得原始评分的状态，不代表第四阶段当前结论。
- **2026-10-25 新增：[六个缺口的实验与算法方案](docs/六个缺口的实验与算法方案.md)**，含 12 项实验、参数、对照、评价、停止规则及三人分工；[设计阶段实际检查](outputs/stage3_design/DESIGN_STATUS.md)。这是设计与先导检查，尚未运行第三阶段完整训练。
- **第一次参赛先读：[从数据到论文的解题教程](docs/从数据到论文的解题教程.md)**
- [第二阶段历史论文 PDF](paper/论文参考稿.pdf) · [历史 Markdown](paper/论文参考稿.md) · [参考稿使用说明](paper/阅读说明.md)
- [第二阶段实验与实际结论](outputs/stage2/STAGE_REPORT.md) · [扩展方法与结果索引](docs/STAGE2_METHOD.md)
- [阶段成果与实测数字](outputs/stage1/STAGE_REPORT.md)
- [数据处理及证据分级](docs/DATA_DECISIONS.md)
- [第一问方法与限制](docs/Q1_METHOD.md)
- [第二问方法与限制](docs/Q2_METHOD.md)
- [第三问方法与限制](docs/Q3_METHOD.md)
- [GitHub 与原始研究检索记录](docs/SOURCES.md)
- [研究过程日志](docs/RESEARCH_LOG.md)
- [需要队内确认的事项](docs/TEAM_REVIEW.md)

## 复现

第四阶段的来源获取、计算与核验：

```powershell
python scripts/fetch_stage4_sources.py
python run_stage4.py all --data-dir "D:\your_data\A附件"
python scripts/stage4_graph_comparison.py --data-dir "D:\your_data\A附件"
python scripts/prepare_trial_protocol.py
python -m pytest -q --junitxml=tmp/stage4_tests.xml
python scripts/verify_stage4.py
python scripts/stage4_figures.py
python scripts/prepare_absolute_paper.py
```

完整重建冻结的历史生成谱须先运行第三阶段q1/q2/q3。作者公开原始文件按固定MD5下载到仓库外，支持`--external-dir`；官方附件仍自行取得，原件不改动。论文PDF另需requirements-paper.txt及本机中文字体；运行环境和代码哈希在manifest中。历史入口如下。

建议 Python 3.12。先取得赛事官方 A 题附件，将 `1.csv`—`8.xlsx` 和数据说明放入同一个本地目录。原始赛题、签名承诺书和原始样本数据不随本仓库公开分发。文件哈希见 `outputs/stage1/audit/source_manifest.csv`。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe run.py all --data-dir "D:\your_data\A附件" --out outputs/stage1
.\.venv\Scripts\python.exe -m pytest -q
```

通用依赖范围见 requirements.txt，本阶段实际环境见 requirements-lock.txt。也可依次运行 `audit`、`q1`、`q2`、`q3`、`report`。第三问依赖第一问在本地产生的模型文件；二进制模型不上传，可从原始数据重新训练。测试中的数据与结果检查在缺少对应材料时会明确跳过，算法性质测试仍能运行。

```powershell
python run.py audit --data-dir "D:\your_data\A附件"
python run.py q1 --data-dir "D:\your_data\A附件"
python run.py q2 --data-dir "D:\your_data\A附件"
python run.py q3 --data-dir "D:\your_data\A附件"
python run.py report --data-dir "D:\your_data\A附件"
```

三折两次重复验证、100 次 bootstrap、生成样本数等均写在 config.json。改变配置应写入新的输出目录和研究日志，保留已有阶段，避免覆盖后混用结果。

第二阶段在第一阶段的基础上增加 GBR、分级假设敏感性、分割保形区间、留出置换贡献、五次划分的生成对照和多锚点迁移。保留第一阶段完整结果，再运行：

```powershell
python run_stage2.py --data-dir "D:\your_data\A附件" --base outputs/stage1 --out outputs/stage2
python -m pytest -q
python scripts/stage2_figures.py
```

配套参考稿已生成，共 13 页（摘要、目录各 1 页，后续 11 页含正文、参考文献和复现索引），含 14 个公式、6 张图和实际实验表。模型、假设、求解、验证及局限均按三问展开。最新实测为 23 项检查通过；GBR 宏平均 MAE 0.3398；150 条因素候选中 11 条通过当前探索性证据条件；扩展搜索找到 11 个模型内迁移候选。**论文是供队员独立学习和复核的参考研究，AI 披露、完整源程序附录、队员贡献和正式提交格式仍须按实际使用补全。**

论文再生成需另装 `requirements-paper.txt`，并提供宋体、黑体；当前 PDF 脚本使用 Windows 系统字体。按顺序运行 `scripts/prepare_paper.py`、`scripts/render_equations.py`、`scripts/build_paper_pdf.py`。数值模型环境不需要安装文档依赖。

第三阶段已有算法原型和只读先导检查，可单独运行：

```powershell
python -m pytest -q --junitxml=tmp/stage3_tests.xml
python scripts/stage3_preflight.py --data-dir "D:\your_data\A附件"
```

当前本地有对应原附件时，全部 **29 项检查通过**；原先第二阶段的 23 项记录保留。先导检查发现：核心 30 维与全 50 维的模型比较方向不同；3 个评分字段别名、12 处疑似 Barnyard 汇总误引用已经另存映射。未修改原工作簿。完整第三阶段训练入口尚未实现，不应把 preflight 当作全部实验运行器。

以上29项与“入口尚未实现”为设计阶段快照。**后续已经实现并运行**第三阶段执行版，当前40项通过。复现新研究使用：

```powershell
python run_stage3.py q1 --data-dir "D:\your_data\A附件"
python run_stage3.py q2 --data-dir "D:\your_data\A附件"
python run_stage3.py q3 --data-dir "D:\your_data\A附件"
python -m pytest -q --junitxml=tmp/stage3_full_tests.xml
python scripts/verify_stage3.py
python scripts/stage3_figures.py
python scripts/revised_equations.py
python scripts/prepare_revised_paper.py
python scripts/build_revised_pdf.py --content paper/revised_20261025/paper_content.json --output paper/revised_20261025/论文修订稿.pdf
```

执行版配置为 config_stage3.json，设计版配置保留为历史计划。验证脚本默认从 data/raw 或相邻 materials/A/附件 找原件；若使用其他位置，可将材料放入上述默认目录再验证。论文生成需要 requirements-paper.txt 与宋体、黑体、Consolas；原始附件不公开分发。模型训练约束与真实化学、真实品评认证分开记录，不把40项检查通过当作科学问题全部解决。

## 结果的使用边界

- `audit/` 为当前附件的计算核查；其中“变换匹配”只针对可比的原始正值配对，不证明零替代值是精确浓度。
- `q1/oof_*` 为留出连续预测，`ASSUMPTION_*` 为未经绝对标定的经验分级。
- `q2/top3_by_sensory_dimension.csv` 为探索候选，低入选频率、反直觉因素和跨方法不一致全部保留。
- `q3/transfer_diagnostics.csv` 会区分找到非平凡候选、原样保留失败及源值缺失；预测代理通过不等于实际品评通过。
- `run_manifest.json`、`result_checksums.json` 和原件哈希用于追踪本次运行。重新运行时模型文件应来自同一阶段。

当前仓库为用户指定的公开仓库。公开代码不自动等于许可复用其他项目；本仓库未复制同题仓库的代码、论文或结果。第三方来源与实际借鉴范围见 SOURCES.md。
