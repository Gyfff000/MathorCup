# 第二阶段：从研究基线到论文参考稿

日期：2026-10-24。代码及稿件由 OpenAI Codex 辅助完成。当前阶段是观察第一阶段之后的探索扩展，不是完全封存数据上的外部确认。完整数值以 outputs/stage2 下的文件为准。

## 第一问

沿用第一阶段三折两次分层划分，在每折训练数据内估计截尾、中位数插补和标准化。增加逐目标 GBR（100 棵深度 2 的树，学习率 0.04，叶节点至少 5 个样本，特征比例 0.7）及 50% 岭回归、50% ExtraTrees 固定融合，不另在外层挑选融合权重。重复留出预测按样本平均，逐目标计算 MAE、RMSE、R² 后宏平均。

在各训练折的目标分布上拟合三种 0—6 等级假设：七等频经验分位、正态累计分布七分位和 5%—95% 分位端点线性映射。比较留出位置的规则分歧，范围输出仅是跨假设范围。原始研究按品评人员标准化，缺少逐人转换参数、出席和汇总信息，现有聚合标签不足以唯一恢复绝对原始评分。

另作独立角色的分割示例：150 训练、50 校准、50 测试，逐维绝对残差的第 ceil[(50+1)×0.9]=46 顺序统计量确定区间。测试不参与区间校准。该实验是逐维边际连续覆盖，不是 50 维同时保障，也不是原始等级保障；一次分割和交换性假设限制结论。

主要文件：q1/metrics_summary.csv、metrics_by_target.csv、oof_gbr_continuous.csv、split_assignments.csv、grade_rule_sensitivity.csv、ASSUMPTION_grade_ranges.csv、conformal_split.csv、conformal_coverage.csv。

## 第二问

保留第一阶段完整 150 条候选与 bootstrap 频率，不人为替换弱项。第二阶段在每个外层训练集拟合固定岭回归，再对留出输入逐变量置换 5 次。利用线性预测的精确差量计算逐目标 MAE 变化，预处理始终冻结。

以入选频率≥0.5、平均留出 MAE 增量>0、至少 2/3 外层划分贡献为正定义描述性支持条件。阈值不表示显著性；候选已有全数据选择信息，因此筛选后的结果不能解释为独立假设检验。相关变量的替代性和风格共变均可影响排名。

主要文件：q2/heldout_permutation_importance.csv 为全部 231×50 组合，q2/top3_evidence_review.csv 连接第一阶段名单并保存支持条件。网络和逐维原始 Top-3 仍见 outputs/stage1/q2。

## 第三问

五次不同种子按风格分层留出 20%；每次只用训练部分拟合预处理、分类器和生成器。比较风格内直接 bootstrap、同风格最近三个邻居间局部凸组合和第一阶段条件 PCA 高斯。每风格每方法每次生成 40 个谱。局部权重范围 0.1—0.9，输出仍是附件尺度的统计插值。

保存对非缺失真实留出字段的 KS、有偏 RBF MMD²、目标风格概率和精确复制比例。第一划分保存 120 个局部生成谱及连续感官画像；画像由第一阶段全数据岭回归代理给出，不能构成独立品评证据。Kriek、Lambic 通常只有两款留出酒，重复划分也不产生新的独立样本。

迁移沿全部 22 个 Tripel 训练锚点各搜索 30 个权重，再投影经验范围及四项源值上界。四项字段为 DMS、X2.3.butanedione、X4VG、X.E..2.nonenal；还要求 stale_hops 预测不增加、目标风格概率超过原样和只截尾对照至少 0.01。11 款源酒均有模型内非平凡候选，9 款四个源字段均有实测，beer 119、28 缺少双乙酰源值。分类器受搜索适应，高概率不能当作外部成功证明。

主要文件：q3/generator_comparison_by_split.csv、generator_comparison_mean.csv、local_convex_generated_profiles.csv、local_convex_sensory_continuous.csv、expanded_transfer_profiles.csv、expanded_transfer_diagnostics.csv、expanded_transfer_defect_checks.csv。

## 运行、审阅与论文

入口 run_stage2.py，参数 config_stage2.json；环境与配置存 run_manifest.json，关键判断存 status.json。实际运行完成 23 项检查，覆盖主键、原件、泄漏隔离、图矩阵、区间顺序统计量和迁移逐项约束；检查通过不等于科学假设已被证实。

paper/论文参考稿.md 可编辑；PDF 已逐页渲染审阅，摘要首页、目录次页、正文从 1 编号，无个人身份页眉。公式图由标准数学排版工具生成，正文与所有数值对应实际结果。正式参赛前仍须独立复核、完善源程序附录、准确披露 AI 使用及队员贡献。
