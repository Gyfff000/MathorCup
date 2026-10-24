# MathorCup 2026 A 题研究记录

围绕啤酒化合物与感官数据建立三问的可复现研究基线，保存文件证据、处理决定、模型实验、限制和阶段结果。本仓库由 OpenAI Codex 辅助建立，核心研究判断、最终算法选择和竞赛成果须由参赛队员独立完成与验证。详见 [AI 使用记录](docs/AI_USAGE.md)。

当前三问均可运行。第一问提供连续感官预测及明确标为假设的经验等级；第二问提供逐维 Top-3 候选与稳定性；第三问提供风格条件生成与逐项约束迁移。**绝对 0–6 分标定、目标相关化学机理、真实品评与酿造可行性尚未完成验证。**

## 从哪里开始阅读

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

## 结果的使用边界

- `audit/` 为当前附件的计算核查；其中“变换匹配”只针对可比的原始正值配对，不证明零替代值是精确浓度。
- `q1/oof_*` 为留出连续预测，`ASSUMPTION_*` 为未经绝对标定的经验分级。
- `q2/top3_by_sensory_dimension.csv` 为探索候选，低入选频率、反直觉因素和跨方法不一致全部保留。
- `q3/transfer_diagnostics.csv` 会区分找到非平凡候选、原样保留失败及源值缺失；预测代理通过不等于实际品评通过。
- `run_manifest.json`、`result_checksums.json` 和原件哈希用于追踪本次运行。重新运行时模型文件应来自同一阶段。

当前仓库为用户指定的公开仓库。公开代码不自动等于许可复用其他项目；本仓库未复制同题仓库的代码、论文或结果。第三方来源与实际借鉴范围见 SOURCES.md。
