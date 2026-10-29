"""AI-assisted manuscript preparation: OpenAI Codex / GPT-6, 2026-10-24.
Every reported result is read from saved experiment tables.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage2"
PAPER = ROOT / "paper"
PAPER.mkdir(exist_ok=True)
status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
metrics = pd.read_csv(OUT / "q1/metrics_summary.csv").set_index("model")
top = pd.read_csv(OUT / "q2/top3_evidence_review.csv")
gen = pd.read_csv(OUT / "q3/generator_comparison_mean.csv")
transfer = pd.read_csv(OUT / "q3/expanded_transfer_diagnostics.csv")
facts = json.loads((OUT / "audit/verified_facts.json").read_text(encoding="utf-8"))
coverage = pd.read_csv(OUT / "q1/conformal_coverage.csv")
blocks = []
def H(text, level=1): blocks.append({"type":"heading", "level":level, "text":text})
def P(text): blocks.append({"type":"paragraph", "text":text})
def E(text, number): blocks.append({"type":"equation", "text":text, "number":number})
def T(headers, rows, widths=None): blocks.append({"type":"table", "headers":headers, "rows":rows, "widths":widths})
def F(name, caption, height=6): blocks.append({"type":"figure", "path":"outputs/stage2/figures/"+name, "caption":caption, "height_cm":height})

baseline = float(metrics.loc["mean", "mae"])
gbr = float(metrics.loc["gbr_fixed", "mae"])
reduction = (baseline-gbr)/baseline
abstract = f"针对啤酒化合物到感官画像的预测、关键因素分析和风格生成问题，构建数据审计、多输出预测、网络关联解释及约束迁移的统一研究流程。依据实际附件，以 beer 为唯一键关联 250 个样本、231 个数值字段和 50 个连续感官目标，逐列核查对数尺度与缺失处理。原始研究采用按品评人员标准化的评分，缺少原始参数时绝对 0—6 分不可唯一恢复，故将连续预测与分级假设分开验证。在训练折内预处理和调参的三折两次重复验证中，固定配置 GBR 的宏平均 MAE 为 {gbr:.4f}，相对均值参照下降 {reduction:.2%}；独立训练、校准和测试分割下，名义 90% 的逐维边际区间获得 {status['conformal_empirical_average_coverage']:.2%} 的平均测试覆盖率。三类分级假设在 {status['grade_sensitivity_disagreement']:.2%} 的预测位置存在分歧。第二问以相关网络和图约束岭回归产生 Top-3 候选，结合 bootstrap 与留出置换贡献，仅 {status['comparatively_supported_top3_associations']}/150 条通过当前描述性证据条件。第三问比较重采样、局部凸组合和 PCA 高斯生成；局部凸组合保留非复制样本优势，但分布距离未全面优于重采样。扩展迁移搜索为 11 个源样本找到模型内可行候选，其中 2 个含缺失缺陷源值。结果支持相对感官预测与受约束的统计设计，尚不支持绝对评分恢复、化学因果或真实酿造效果的结论。"

H("1 问题重述与研究范围")
P("啤酒风味由原料、发酵过程和多种化合物相互作用共同形成。原始研究已将实测化学资料与训练品评联系起来，说明此类数据具有建模价值[1]。本题进一步要求处理缺失、分级评分、稳定因素识别和跨风格生成，因此研究重点不仅是拟合精度，还包括目标定义是否清楚、解释是否稳健，以及生成谱满足什么层次的约束。")
P("第一问以化合物和理化测量为输入，估计品评表各感官维度；第二问建立化合物关系网络并对每维给出三个候选因素；第三问按 tasting_category_fine 生成风格条件样本，并将 Pils/Lager 向 Tripel 调整，要求指定缺陷项不恶化。本文对实际附件的 50 个连续目标完整输出，Aroma 与 Flavour 维度单独标识；最终提交维度应依据官方明确口径核对。")
P("本文研究范围为对给定附件的统计建模与可复现验证。原始 0—6 分锚点、部分浓度尺度、聚合量公式和真实感官试验均存在信息缺口。对这些缺口采用明确假设、敏感性和失败记录处理，保留可核查结果，不以预测代理替代实测。")

H("2 模型假设与符号说明")
H("2.1 基本假设", 2)
P("假设一，beer 能稳定标识同一款啤酒，不同文件按该字段正确对应。该假设经唯一性、集合和风格一致性检查支持；受损的 beer_id 不用于合并。假设二，在相同数据获取机制下，现有样本可用于研究新样本的相对感官预测；此假设不保证未见风格或不同酿造工艺下的外推。")
P("假设三，训练分布的分位截尾与中位数插补可作为噪声、极端值与缺失的研究基线。它们不证明被处理值是错误观测，后续需要针对检测限与非随机缺失改进。假设四，相关网络可作为参数正则化信息，但不具有先验因果含义。假设五，生成与迁移的经验范围和代理模型只定义统计可行性，不构成物理、化学或生产可行性的充分条件。")
H("2.2 主要符号", 2)
T(["符号", "含义"], [["n、p、q", "样本数 250、输入字段数 231、感官目标数 50"], ["X、Y", "附件尺度输入矩阵与连续感官目标矩阵"], ["Z", "仅由训练数据拟合预处理得到的标准化输入"], ["B、α、λ", "回归系数、岭惩罚和网络惩罚强度"], ["W、L", "截断有符号相关权与对应拉普拉斯矩阵"], ["f、g", "连续感官预测器与风格诊断分类器"], ["c、h", "目标风格条件与低维潜变量"], ["D、D̂", "缺陷化学字段集合与学习得到的感官缺陷代理"]], [.23,.77])

H("3 数据审计与预处理")
H("3.1 样本与字段核查", 2)
P("实际附件中 1.csv 有 250 行、234 列，前 3 列为元信息，其余 231 列包括化合物、理化和综合字段；4.csv 含 50 个连续感官目标。化合物条码在感官文件中存在精度损失，直接用条码合并可能产生多对多错配。程序要求 beer 非空、唯一、两表集合相同，并在对齐后检查风格标签一致。")
T(["资料", "实际结构", "研究用途"], [["1.csv", "250×234，含 3 列元信息", "化合物及理化输入"], ["2.csv", "去行名后 231×231", "描述性相关网络"], ["3.xlsx", "12 个 beer 页及 summary", "感官定义与空模板审计"], ["4.csv", "250×53，含 50 个目标", "连续感官监督信息"], ["6.csv", "去行名后 44×50", "跨域关联参考，不作新样本"], ["8.xlsx", "阈值、名称、校准等工作表", "尺度和术语证据"]], [.16,.38,.46])
P("完整原件以 SHA-256 绑定版本。字典、字段尺度、缺失统计和样本键检查随支撑文件保存。题面与文件的数量差异按实际证据记录，不补造缺失行列。8.xlsx 中已有 SHAP 与重要度属于既有研究结果，未作为本文重新训练得到的结论，也不用于第一问的外层变量筛选。")
F("style_counts.png", "图1 风格样本数分布 显示小样本与类别不平衡", 8)
H("3.2 量纲与缺失处理", 2)
P("对可比的原始正值配对，逐字段比较恒等关系与 log10 关系。检查支持 37 个字段在这些配对上符合 log10，包括题面列为可能保留原量纲的部分理化指标；另有字段只在多数配对一致，或缺少可比原值。正值配对一致性并不能识别零替代值与检测限。故保留附件尺度建模，不对所有字段统一逆变换。")
P(f"给定相关矩阵与重算 Spearman 的平均绝对差为 {facts['spearman_matrix_mean_absolute_difference']:.6g}，与 Pearson 为 {facts['pearson_matrix_mean_absolute_difference']:.6g}，更支持采用秩相关解释。矩阵的完整结构一致性也用于核对字段别名次序。自然缺失由训练中位数填补，训练 1% 和 99% 分位定义截尾范围，再作训练均值与标准差标准化。所有操作置于 Pipeline 中[2]。")
E("z_ij = [clip(x_ij; a_j, b_j) − m_j] / s_j", 1)
P("式（1）中的边界、插补值、均值和标准差均只在当前训练数据估计。验证与测试样本仅执行变换，不参与参数学习。对极端值的处理属于可检验的建模假设，而非原始观测纠错。")

H("4 第一问 感官预测与分级不确定性")
H("4.1 评分尺度的可识别性", 2)
P("原始研究明确说明按品评人员分别中心化和标准化，并非仅对汇总后的每个属性作一次统一线性变换[1]。设品评人员 t 的原始评分为 s_it，标准化后为 z_it，若其均值 μ_t 与标准差 σ_t 未知，则不能从 z_it 唯一得到 s_it；不同品评人员的出席与汇总权重进一步增加信息需求。")
E("z_it = (s_it − μ_t) / σ_t，s_it = μ_t + σ_t z_it", 2)
P("对于缺少锚点的数据，多组满足原始评分范围的 μ_t、σ_t 都可能产生相同标准化值。空模板公式返回的零也不等于真实打分。因此本文把可验证的连续预测和有假设的相对分级分开，未把简单拉伸、裁剪或经验分位声称为原始评分恢复。")
H("4.2 多输出预测模型与求解", 2)
P("设第 k 个感官目标的预测为 f_k(x)。比较训练均值、岭回归、PLS、ExtraTrees[3]、固定配置梯度提升回归[4]以及固定比例的岭回归与 ExtraTrees 融合。岭回归通过 L2 惩罚控制高维共线，PLS 学习与目标相关的低维方向，树模型捕捉非线性。输入中不包含编号、条码、风格标签或样本的感官评论信息。")
E("B̂ = argmin_B {∥Yc − ZB∥²_F + α∥B∥²_F}", 3)
P("岭回归 α 在 {1,10,100,1000} 中由内层三折选择；PLS 成分数在 {3,8} 中选择。ExtraTrees 使用 200 棵树、最小叶样本 3、特征比例 0.8。GBR 对每个目标分别拟合 100 棵深度为 2 的树，学习率 0.04，最小叶样本 5，特征比例 0.7。GBR 参数为第二阶段固定探索配置，未根据本阶段外层目标逐折调整。")
H("4.3 验证设计与性能", 2)
P("采用按风格分层的三折、两次重复外层验证。每个样本每次重复仅被留出一次，预处理和调参在外层训练数据内完成。重复留出预测取均值后计算逐维指标，并对 50 维等权宏平均。最少风格有 3 个样本，选用三折避免强行采用超过该样本数的分层折数。第二阶段是在第一阶段观察之后的探索改进，不能视为未接触过数据的外部验证。")
E("MAE = (1/nq) Σ_i Σ_k |y_ik − ŷ_ik|", 4)
T(["方法", "宏平均 MAE", "宏平均 RMSE", "宏平均 R²"], [[name, f"{metrics.loc[key,'mae']:.4f}", f"{metrics.loc[key,'rmse']:.4f}", f"{metrics.loc[key,'r2']:.4f}"] for key,name in [("mean","均值参照"),("ridge","岭回归"),("pls","PLS"),("extra_trees","ExtraTrees"),("gbr_fixed","固定 GBR"),("fixed_ridge_tree_blend","固定比例融合")]], [.28,.24,.24,.24])
P(f"GBR 的 MAE 为 {gbr:.4f}，相对均值参照下降 {reduction:.2%}。它与 ExtraTrees 的差距较小，不能仅据单个平均值断言普遍优越。宏平均 R² 约 {metrics.loc['gbr_fixed','r2']:.3f}，说明仍有较大未解释变异；逐维性能可能差别明显。上述单位均为附件的连续目标尺度，不能改写成 0—6 分的平均误差。")
F("model_comparison.png", "图2 六类方法在相同重复留出划分下的连续目标误差", 6)
H("4.4 分级敏感性与预测区间", 2)
P("在训练目标上分别构造七等频经验分位、正态分布累计概率和以 5%—95% 分位为端点的线性映射三类等级规则。它们均输出 0 至 6，但语义不同；重复分位边界也会影响离散分级。对每个外层留出预测比较三者，完整保存差异与跨规则范围。")
P(f"三种规则在 {status['grade_sensitivity_disagreement']:.2%} 的预测位置给出不同等级，显示绝对标定对假设高度敏感。跨规则的最小值和最大值仅表示假设范围，不是统计置信区间。若获得参考酒原始评分或逐品评人员转换参数，应重新标定和验证，再向原始 0—6 分作实质性解释。")
P("另用 150 个训练、50 个校准、50 个测试样本展示分割保形预测[5]。用训练模型在校准集的绝对残差定义逐维分位，再为测试预测添加对称区间。名义水平为 90%，目标是单个感官维度的边际覆盖，并非所有 50 维同时覆盖。")
E("r_ik = |y_ik − f_k(x_i)|；r*_k = 第 ceil[(m+1)(1−α)] 个校准残差", 5)
E("C_k(x) = [f_k(x) − r*_k，f_k(x) + r*_k]", 6)
P(f"本次平均测试覆盖率为 {status['conformal_empirical_average_coverage']:.2%}，平均区间宽度为 {coverage.interval_width.mean():.4f}。该结果只对应一次分割和当前连续标签；交换性假设、样本筛选与原作者预处理都会限制其外推。不能把边际覆盖率解释为每种风格或每款啤酒都达到同一保障，更不能据此推导原始等级的覆盖率。")
F("calibration_uncertainty.png", "图3 分级规则分歧与逐维连续预测区间的测试覆盖分布", 6)

H("5 第二问 化合物网络与解释证据")
H("5.1 共表达图及图约束回归", 2)
P("以 231 个数值字段为节点，给定相关矩阵中 |r|≥0.7 时建立边，同时保存符号与绝对权重。当前有 109 条边；0.6、0.7、0.8、0.9 阈值的网络变化另存。模块包含孤立节点，不能直接当作生化通路。对每个目标最终只从化合物或元素字段中选择候选，pH、色度、混合和聚合指标不冒充单一化合物。")
E("W_ij = r_ij I(|r_ij|≥τ)，L = diag(Σ_j |W_ij|) − W", 7)
E("B̂ = argmin_B {∥Yc−ZB∥²_F + α∥B∥²_F + λ tr(BᵀLB)}", 8)
P("符号拉普拉斯的二次型为相关边上系数差异或相反关系的加权惩罚，矩阵为半正定。固定 α=20、λ=5 后，求解 (ZᵀZ+αI+λL)B=ZᵀYc。该模型提供一种利用关系结构的正则化基线，尚未证明固定参数最优。每个目标按标准化特征的系数绝对值取前三项，方向和相关性均保留。")
H("5.2 重采样稳定性与留出置换贡献", 2)
P("采用 100 次样本有放回重采样，每次重新估计截尾、插补、标准化和秩相关网络，重新拟合并计数 Top-3。另比较图约束回归、无图岭回归和边际 Spearman 的名单 Jaccard。相关因素可能相互替代，所以单个因素频率低与模块关联存在并不矛盾。")
E("π_jk = bootstrap 中化合物 j 进入目标 k 的 Top-3 次数 / 重采样次数", 9)
P("第二阶段对每个外层训练拟合的岭回归，在其留出样本上单独置换某一特征，观察逐维 MAE 增量，重复 5 次并汇总六个外层划分。对线性预测利用固定预处理后的变量差与系数精确计算，避免反复拟合同一模型。置换贡献取决于模型与相关结构，不是化合物的内在因果效应[6]。")
E("Δ_jk = MAE_k(置换第 j 个变量后的留出预测) − MAE_k(原留出预测)", 10)
P("定义探索性证据条件：入选频率至少 0.5，平均留出 MAE 增量大于零，且至少三分之二外层划分增量为正。阈值为本研究的描述性判据，不是显著性检验或官方评分标准。由于候选名单包含全数据探索信息，筛选后的支持数仍存在选择适应性，不能作未经选择的独立假设检验。")
P(f"150 条候选中，{status['comparatively_supported_top3_associations']} 条同时满足当前条件。第一阶段有 137 条入选频率低于 0.5，说明多数 Top-3 还不适合写成确定因素。完整 150 条及所有变量的贡献保存在支撑表中，弱证据项和反直觉项不删除。")
selected = top[top.evidence_status == "comparatively_supported_association"]
T(["感官目标", "候选字段", "入选频率", "留出 MAE 增量"], [[r.target,r.field,f"{r.bootstrap_top3_frequency:.2f}",f"{r.heldout_permutation_MAE_increase:.5f}"] for r in selected.itertuples()], [.23,.43,.16,.18])
F("factor_evidence.png", "图4 Top-3 候选的重采样频率与留出置换贡献 通过描述条件仍不表示因果", 6)
H("5.3 化学合理性及反例", 2)
P("4VG 感官目标与 X4VG 的联系具有较强的统计一致性。原始化学研究指出 4VG 与特种啤酒的丁香气味相关，并研究了其陈化降解[7]，可作为该关联的外部合理性依据。金属味与 iron、烘烤麦芽与 methylpyrazine 等候选也可结合附件的名称、来源和缺陷描述继续论证，但本文未开展加标、盲评或反应机理实验。")
P("表中也存在不宜直接解释为气味来源的关联，例如苦味与 iron、乳酸相关描述与某酯类字段。它们可能反映风格共变、共线替代或模型不足。源于同一附件的元数据只能提供术语与来源线索，不能独立验证每个目标的机制。对这些结果应保留待证状态，不能人为换成期望化合物来美化模型。")

H("6 第三问 风格生成与约束迁移")
H("6.1 小样本条件生成与对照", 2)
P("按风格分层留出 20% 样本，训练预处理和生成模型仅使用训练部分。比较三类方法：风格内 bootstrap、同风格近邻之间的局部凸组合，以及十维 PCA 中的条件高斯采样。PCA 模型以风格均值为中心，协方差按 n_c/(n_c+10) 将风格内估计与汇聚估计混合。")
E("h|c ~ N(μ_c，w_cΣ_c + (1−w_c)Σ_pool)，w_c = n_c/(n_c+10)", 11)
P("局部凸组合在相同风格中选择一个训练样本和其近邻，以 0.1—0.9 的随机权重插值。该方法是一种条件经验生成模型，生成的新点不等同于化合物可直接混配的物理过程，尤其在对数尺度上只代表统计插值。")
E("x_new = u x_a + (1−u)x_b，u ~ Uniform(0.1，0.9)，a、b 属于同一风格", 12)
H("6.2 生成验证与结果", 2)
P("用五种随机划分重复比较，每次为 Tripel、Kriek、Lambic 各生成 40 个谱。指标包括对真实非缺失留出测量的平均 KS、训练表示上的有偏 RBF MMD²、目标风格分类概率及精确复制比例。分类器与生成器共享训练集，留出表现用于说明诊断模型能力；生成概率并不是外部独立风格鉴定。五次划分也会重复使用同一批酒。")
T(["风格", "方法", "平均 MMD²", "精确复制比例"], [[r.style,r.method,f"{r.MMD2:.4f}",f"{r.exact_training_copy_fraction:.2f}"] for r in gen.itertuples()], [.18,.38,.24,.20])
P("局部凸组合没有精确复制训练样本，目标风格分类概率较高；但其 MMD 未全面优于直接重采样。PCA 高斯的复杂性也没有带来分布距离优势。因每风格只有极少留出样本，尤其 Kriek 和 Lambic 每次通常只有两款，距离指标不能证明整个分布已被学会。生成模型的评价应同时看接近程度、复制风险与可解释约束，而不能只追求分类器高分。")
F("generator_comparison.png", "图5 五次划分下生成方法的分布距离与内部风格概率 误差线是划分标准差", 6)
P("第一种划分的局部凸组合样本及连续感官画像完整保存。画像来自第一问预先指定的全数据岭回归代理，因此不能用作独立的真实感官验证。五个综合指标随谱联合生成，原作者聚合公式尚未完整恢复，精确代数一致性也未被证明。")
H("6.3 逐项缺陷约束与迁移搜索", 2)
P("第一阶段采用 PCA 风格均值方向搜索，11 个源样本中只有 7 个找到非平凡候选。第二阶段扩大为目标风格条件训练锚点搜索：对每个源样本，沿每个 Tripel 训练锚点方向取 30 个权重，再作经验范围投影与四项缺陷字段的逐项上界投影。四项分别为 DMS、双乙酰、4VG 和 trans-2-nonenal。相比单一潜方向，该搜索保留更多目标风格变化方向。")
E("x_candidate,d ≤ x_source,d，d∈{DMS、Diacetyl、4VG、trans-2-nonenal}", 13)
E("f_stale(x_candidate) ≤ f_stale(x_source)", 14)
P("同时要求固定分类器的 Tripel 概率超过原始样本和仅做截尾对照至少 0.01。该阈值是操作性的非平凡变化标准，并非真实风格成功阈值。候选选择按模型概率进行，因此会适应诊断分类器；高概率不能作为优化之后仍独立的验证证据。未找到候选时保留原样并记失败，原样不算成功迁移。")
P(f"扩大搜索后，11 个源样本都找到满足当前模型条件的候选，其中 {int(transfer.all_source_defects_observed.sum())} 个具有四项缺陷源字段实测，另 2 个缺少双乙酰值，只能相对插补值检查；Stale hops 对全部样本都是学习得到的代理。源样本中 2 个属于生成器留出集，其余为训练来源。增加搜索范围带来更多模型可行候选，但没有增加真实品评证据。")
F("transfer_comparison.png", "图6 扩展搜索与单一 PCA 方向搜索的内部分类概率比较 缺失源值另作标注", 6)
P("DMS 与其他缺陷的单位并不统一，部分尺度和 trans-2-nonenal 阈值也未确定。本文因此不声称全部绝对 OAV 阈值已通过。测量字段逐项不增加是较保守的模型条件，但若来源值有检测限替代、缺失或尺度歧义，仍需逐项补证。下一步应恢复基本成分的浓度和聚合公式，加入发酵过程与酿造可行性，再以盲评检验感官效应。")

H("7 稳健性 可复现性与模型评价")
H("7.1 已完成的稳健性检查", 2)
P("本流程分别保存了分层重复留出、训练折调参、额外随机缺失、网络阈值变化、bootstrap 网络重建、跨方法排名、留出置换、分级规则变化和生成划分变化。第一阶段与第二阶段分目录存储，保留方法变化前后的成绩和失败样本。23 项程序检查覆盖主键对齐、原件未改、预处理隔离、符号拉普拉斯性质、区间残差排序、留出角色、逐项缺陷和失败标记；这些检查证明实现满足指定性质，不证明全部科学假设成立。")
H("7.2 优点与适用范围", 2)
P("流程的主要价值是把尺度审计、预测验证、关联证据和生成约束分开评估。简单参照使性能提升可比较，弱因素与失败案例保留使解释不过度确定，模型内约束也明确限定在可观测字段或代理。对类似小样本、高维、混合量纲的风味研究，这些步骤比单纯增加模型复杂度更有实际参考价值。")
H("7.3 局限与改进方向", 2)
P("其一，原始逐人评分与绝对锚点缺失使真实 0—6 分不可恢复；分级规则高度分歧，不能把假设等级包装成真值。其二，噪声、检测限、品评差异和有限样本限制预测能力，且源数据预处理范围未完全明确。其三，Top-3 多数不稳定，反直觉项缺少因果证据；图约束参数与解释阈值也需要更系统敏感性。")
P("其四，生成验证的稀有风格留出过少，分类器共享训练信息且会被迁移搜索适应；重复分割不能代替新数据。其五，聚合量公式、物理约束、缺陷阈值和感官实验未完成。其六，第二阶段已经基于第一阶段经验设计，缺少完全封存的最终外部测试。因此本文定位为可复核的参考研究，后续应以新增测量、品评锚点和独立实验补充关键证据。")

H("8 结论与建议")
P(f"在当前附件尺度上，化合物对感官画像具有一定预测信息：固定配置 GBR 的连续宏平均 MAE 为 {gbr:.4f}，优于均值参照；但原始 0—6 分缺少可识别条件，不应声称已恢复。加入训练、校准和测试隔离的区间示例，可补充连续预测的不确定性表达，仍需遵守边际覆盖及交换性限制。")
P(f"化合物网络可组织共变关系并提供正则化信息，但 150 条 Top-3 仅 {status['comparatively_supported_top3_associations']} 条满足当前相对支持判据，说明稳定解释和化学机制是后续重点。生成比较显示局部凸组合在非复制与风格诊断之间有实用折中，复杂高斯模型尚未表现出全面优势。扩展搜索扩大了模型内可行迁移范围，却不能替代真实浓度阈值、酿造过程或品评验证。")
P("后续研究优先顺序为：获取原始评分或参考酒锚点；复核尺度、检测限与异常；对关键因素开展浓度和文献核查；用封存的新样本验证生成与迁移；最后据实更新论文与 AI 使用记录。")

H("参考文献")
references = [
"[1] Schreurs M, Piampongsant S, Roncoroni M, et al. Predicting and improving complex beer flavor through machine learning. Nature Communications, 15:2368, 2024. DOI:10.1038/s41467-024-46346-0.",
"[2] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12:2825-2830, 2011.",
"[3] Geurts P, Ernst D, Wehenkel L. Extremely randomized trees. Machine Learning, 63:3-42, 2006. DOI:10.1007/s10994-006-6226-1.",
"[4] Friedman J H. Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5):1189-1232, 2001. DOI:10.1214/aos/1013203451.",
"[5] Lei J, G'Sell M, Rinaldo A, et al. Distribution-Free Predictive Inference for Regression. Journal of the American Statistical Association, 113(523):1094-1111, 2018. DOI:10.1080/01621459.2017.1307116.",
"[6] Scikit-learn developers. Permutation Importance with Multicollinear or Correlated Features. https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html，访问日期：2026-10-24.",
"[7] Vanbeneden N, Saison D, Delvaux F, Delvaux F R. Decrease of 4-vinylguaiacol during beer aging and formation of apocynol and vanillin in beer. Journal of Agricultural and Food Chemistry, 56(24):11983-11988, 2008. DOI:10.1021/jf8019453.",
"[8] OpenAI Codex，GPT-6 家族（精确可公开版本待核验），OpenAI，使用日期：2026-10-24."
]
for reference in references: P(reference)
H("AI 工具使用声明")
P("本参考研究使用 OpenAI Codex 辅助进行资料检索、代码实现、数据审计、实验记录、图表和稿件编写[8]。正文所述计算为 AI 辅助代码的实际运行结果，不能据此声明核心工作已由参赛队员独立完成。队员后续应独立理解、验证、修改并记录实际贡献；采纳内容须按赛事要求在相应位置标注。模型家族信息可由会话核验，精确版本应按实际界面补充，不臆造版本号。重要交互与采纳修改记录见随附 AI 使用阶段记录，最终提交前须补全规定的 PDF 说明。")
H("附录 复现与结果索引")
P("第一阶段运行 run.py all，第二阶段运行 run_stage2.py，配置分别为 config.json 和 config_stage2.json。原始附件由使用者自行取得并经 source_manifest.csv 核对哈希。Python 与依赖版本锁定在 requirements-lock.txt；生成论文的脚本位于 scripts/，原始数据未随公开仓库分发。")
T(["材料", "路径与内容"], [["数据证据", "outputs/stage2/audit/：字典、原件哈希、尺度和模板检查"], ["第一问", "outputs/stage2/q1/：连续误差、GBR 留出预测、分级敏感性和区间覆盖"], ["第二问", "outputs/stage2/q2/：完整置换贡献与 150 条候选证据；原始网络见 stage1/q2"], ["第三问", "outputs/stage2/q3/：生成对照、局部样本、画像、迁移与缺陷检查"], ["研究源代码", "src/、run.py、run_stage2.py；算法与完整运行代码随支撑包提供"], ["入门教程", "docs/从数据到论文的解题教程.md；按数据、模型、验证与论文顺序阅读"]], [.20,.80])
P("本稿未逐页排入全部源代码，也未填写队号、承诺书和队员独立贡献。若转为正式参赛作品，应依据官方要求把全部实际使用源程序纳入附录，并完成匿名、页数、引用与 AI 披露检查。参考稿中的条件、失败结果和未证实事项不得在修改时无依据删除。")

title = "基于尺度审计与稳定关联的啤酒感官预测及风格约束迁移"
payload = {"title":title, "abstract":abstract, "keywords":"感官预测；数据尺度；图约束回归；重采样稳定性；条件生成；约束迁移", "blocks":blocks}
(PAPER / "paper_content.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
md = ["# "+title,"","## 摘要",abstract,"","关键词："+payload["keywords"],"","## 目录","见 PDF 的自动生成目录。正文以下各节与实际结果文件对应。",""]
for b in blocks:
    if b["type"]=="heading": md += ["#"*(b["level"]+1)+" "+b["text"],""]
    elif b["type"]=="paragraph": md += [b["text"],""]
    elif b["type"]=="equation": md += [f"公式（{b['number']}）：{b['text']}",""]
    elif b["type"]=="table":
        md += ["| "+" | ".join(b["headers"])+" |","|"+"---|"*len(b["headers"])]
        md += ["| "+" | ".join(map(str,row))+" |" for row in b["rows"]]
        md += [""]
    elif b["type"]=="figure": md += [f"![{b['caption']}](../{b['path']})",""]
(PAPER / "论文参考稿.md").write_text("\n".join(md), encoding="utf-8")
print(json.dumps({"blocks":len(blocks),"characters":len('\n'.join(md)),"paper":str(PAPER/'论文参考稿.md')},ensure_ascii=False))
