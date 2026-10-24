# MathorCup 2026 A 题研究框架

本仓库保存 A 题三问的 AI 辅助参考代码、方法说明和公开来源记录。三问框架已在本地运行，样本级预测、生成谱、迁移结果、审计表和图表目前保留本地，公开发布等待用户明确授权。

核心研究判断、最终算法设计、求解与分析需由参赛队员独立完成和验证；AI 使用须按赛事要求披露。绝对 0–6 分标定、化学因果、真实品评和酿造可行性均未获得完整验证。

## 阅读顺序

- [公开资料检索及借鉴范围](docs/SOURCES.md)
- [第一问方法](docs/Q1_METHOD.md)
- [第二问方法](docs/Q2_METHOD.md)
- [第三问方法](docs/Q3_METHOD.md)
- [队内复核清单](docs/TEAM_REVIEW.md)
- [AI 使用阶段记录](docs/AI_USAGE.md)

## 复现

请自行取得官方 A 题原始附件，本仓库不公开分发原始资料、签名承诺书或个人信息。建议使用 Python 3.12，在本地安装锁定依赖并运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe run.py all --data-dir "D:\your_data\A附件" --out outputs/stage1
.\.venv\Scripts\python.exe -m pytest -q
```

可分别执行 audit、q1、q2、q3、report，第三问依赖第一问的本地模型。改变实验配置应另建输出目录。样本关联、训练折预处理、嵌套调参、bootstrap 和逐项迁移约束均在代码与方法记录中说明。

当前公开版本包含代码与方法，不包含需单独授权公开的派生结果。未采用其他同题仓库的完整代码、论文或结果。
