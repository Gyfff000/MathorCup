# 2026-10-25 新版原始评分补查

- [作者Zenodo v4](https://zenodo.org/records/23624915)，2026-10-24发布，新增Raw sensory data anonymized.csv，MD5 a8f911d8c0061e096abf3b087678e164；补充工作簿MD5 52645e38d90aafda272532e19e2fea1c。两文件已下载核验，原件在仓库外，未重新公开分发。v3先前下载障碍不再代表当前获取状态。
- [Meilgaard 1993原始阈值研究](https://doi.org/10.1016/0950-3293(93)90158-3)：多实验室与品评人员的啤酒阈值试验，作为外部参考，不替代本批次标定。
- [Santos等2003 nonenal测量研究](https://pubmed.ncbi.nlm.nih.gov/12580508/)：约0.1 ug/L的感知背景值用于敏感性，对应0.0001 mg/L；不据此认证本附件全部浓度或新候选缺陷。
- 原作者全文通过Europe PMC的PMC10966102 fullTextXML核对，重点为Trained panel与Chemical dataset；公开R/Python脚本仅用于来源追踪，未逐段复制成自有算法。

当前匿名对应、零方差和化学例外见outputs/stage4/audit。以下为历史检索记录，其“尚未获得”只代表各阶段当时状态。

# 公开检索与借鉴记录

## 2026-10-25 论文修订补充来源

- [Breiman 1996 Stacked Regressions，作者机构全文](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf)：交叉验证组合权重与过拟合的已有研究；本题融合仅作应用变体，不声称首创。
- [Li和Li 2010图约束回归原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3423227/)：拉普拉斯二次惩罚的先例，不将本题符号相关图称为新的回归理论。
- [Vanbeneden等2008，4VG陈化](https://pubmed.ncbi.nlm.nih.gov/19053838/)：支持丁香相关香气的合理性，不提供本批次因果或新加标证据。
- [Omur-Ozbek等2012，口腔金属味实验](https://pubmed.ncbi.nlm.nih.gov/22304665/)：铁/铜溶液与脂质氧化，研究区分铁价态；不将其阈值直接套用总iron或啤酒基质。
- [Pater等2024，芫荽籽添加低醇啤酒](https://pubmed.ncbi.nlm.nih.gov/38398596/)及[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10892622/)：linalool相关气味活性背景，不证明本附件特定标签的单化合物因果。

只有明确支持的案例写入chemical_evidence_ledger，其余保留元数据或未充分核实状态；没有为150条候选补造独立文献。

## 2026-10-25 实验设计的方法依据

以下均核对原始研究页面；用于设计与界定证据范围，未复制其代码或把其结果归为本项目实验。

- [Bengio 与 Grandvalet，交叉验证方差研究](https://jmlr.org/papers/v5/grandvalet04a.html)：重叠训练/测试产生误差相关，不能把重复折当独立样本简单检验。本阶段固定 OOF 重采样明确只作条件敏感性。
- [Lei 等，回归保形预测](https://arxiv.org/abs/1604.04173)：用于设计拟合/校准分离的逐维区间；映射到固定相对等级不等于恢复原始等级。
- [Debeer 与 Strobl，Conditional permutation importance revisited](https://pmc.ncbi.nlm.nih.gov/articles/PMC7362659/)：用于区分边际贡献、相关替代与条件解释。拟议风格内/模块置换不冒充严格完整条件置换。
- [Gretton 等，A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html)：分布比较依据；设计中所有生成方法共用训练确定的表示与带宽，不以小样本不显著等同分布一致。
- [Schreurs 等原始研究的公开全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10966102/)：核对数据来源和研究范围。原作者已有模型和加标实验不能作为本队新增实测证据。

完整设计见 docs/六个缺口的实验与算法方案.md，科学效果尚待完整第三阶段实验。

## 2026-10-24 初始检索记录

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
