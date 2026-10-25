# MathorCup 2026 A 题研究记录

围绕啤酒化合物与感官数据建立三问的可复现研究基线，保存文件证据、处理决定、模型实验、限制和阶段结果。本仓库由 OpenAI Codex 辅助建立，核心研究判断、最终算法选择和竞赛成果须由参赛队员独立完成与验证。详见 [AI 使用记录](docs/AI_USAGE.md)。

当前三问均可运行。第一问提供连续感官预测及明确标为假设的经验等级；第二问提供逐维 Top-3 候选与稳定性；第三问提供风格条件生成与逐项约束迁移。**绝对 0–6 分标定、目标相关化学机理、真实品评与酿造可行性尚未完成验证。**

## 从哪里开始阅读

- **最新：2026-10-25 实验与论文修订版**：[论文PDF（含完整源程序附录）](paper/revised_20261025/论文修订稿.pdf) · [可编辑正文](paper/revised_20261025/论文修订稿.md) · [AI工具使用详情阶段版](paper/revised_20261025/AI工具使用详情说明_阶段版.pdf) · [修订说明](docs/论文修订说明_20261025.md) · [第三阶段实测报告](outputs/stage3/STAGE_REPORT.md)。主要计算实验已执行，40项检查通过；原始评分锚点、完整化学认证和真实品评仍未完成。
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
