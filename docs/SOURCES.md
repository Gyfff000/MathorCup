# 公开检索与借鉴记录

检索日期：2026-10-24。先检索 GitHub，再开展本阶段实现。查询包括 `beer machine learning`、`beer sensory`、`beer flavor prediction` 及 `tasting_category_fine`。按数据来源、输入目标是否一致、验证方法是否可核对进行筛选，没有把仓库排名或名称视为质量证明。

| 来源 | 已阅读内容 | 与本题关系和借鉴范围 |
|---|---|---|
| [scikit-learn 相关特征示例](https://github.com/scikit-learn/scikit-learn/blob/main/examples/inspection/plot_permutation_importance_multicollinear.py) | 官方完整示例，BSD-3-Clause 标记 | 相关特征会稀释单变量重要度，应该查看模块与方法稳定性。没有直接复制分类示例作为本题的回归结果 |
| [scikit-learn 防止泄漏](https://scikit-learn.org/1.9/common_pitfalls.html) | Pipeline 与训练/测试预处理说明 | 预处理、插补、特征选择在训练折内拟合；本阶段实现采用嵌套训练与验证 |
| [jenmusdat/BeerMachineLearning](https://github.com/jenmusdat/BeerMachineLearning) | README，blob SHA `2a85d2633021a5ebbb9c8067601bfd4a9e402cf0` | 使用评论文本、Word2Vec、回归与聚类；可参考记录清洗与实验的方式，输入并非本题化合物测量，未采用其成绩与模型 |
| [rmwoods/beer.ai](https://github.com/rmwoods/beer.ai) | README，blob SHA `f232ba88cb86802c1d8639563a4277a21d8e7d59` | 以 BeerXML 配方和原料向量研究生成，强调名称与单位标准化。配方生成不等于化合物谱生成，本阶段没有复用其代码或数据 |
| [PyrmontBrewery/beersensory](https://github.com/PyrmontBrewery/beersensory) | README，blob SHA `9fc131bd350bffa8dd4e7dae6a4e6ccb0e0cbf12` | 感官描述的分类整理；未将其 README 当作阈值或具体化学机理的权威依据 |
| [Schreurs 等原始研究](https://www.nature.com/articles/s41467-024-46346-0) | 研究主题、数据规模与来源说明 | 与附件的 250 款啤酒和 50 个感官目标高度吻合；用于来源核对，不能把论文已有贡献归于本队 |
| [作者 Zenodo 仓库](https://zenodo.org/records/10653704) | 文件清单与 [模型训练脚本](https://zenodo.org/records/10653704/files/Machine%20learning%20models%20transcript.py?download=1) | 原作者比较线性、PLS、树模型等，保存模型和评价。参考基线比较与风格分层思路。本阶段独立实现 Pipeline 嵌套验证，没有复用其预训练模型 |

代码搜索还命中公开同题仓库（如 dd2006tqf/beer）以及包含同名字段的其他项目。仅查看了搜索返回的片段，未采用同题完整代码、论文或结果。片段中的成绩与自评没有独立核验，未作为本阶段结论或可靠化学证据。也未联系其他参赛队伍讨论赛题。

原始完整工作簿的下载在此前尝试中受访问限制，本阶段没有核验其中是否提供可恢复绝对 0–6 分的原始锚点。原作者脚本可读，不代表已获取或有权复用其中提到的所有外部数据，尤其是受限的 RateBeer 评分。

本阶段代码使用标准依赖 API 重新实现，未逐段复制第三方代码。若队员后续加入外部实现，应另外核对许可证、记录具体文件和版本，并按赛事要求引用。
