## 5.1 实验设置（Experimental Setup）

**水印配置（Watermark Configuration）。** 沿用 SFWMark 的 HSQR 密钥与中心感知嵌入 [1]。QR 版本为 1，包含 $21\times21$ 个模块，模块边长为 2，纠错等级为 H。水印嵌入零起始索引为 3 的潜变量通道，在中央 $44\times44$ 区域进行频域处理；检测时提取 $42\times21$ 个复数系数，将实部与虚部依次展平为 1,764 维特征。嵌入保留各实、虚分量的幅值，依据 QR 图案调整符号，并维持 Hermitian 对称，幅值偏移参数 $\delta=0$；检测模板使用 $\pm45$。所有配置使用相同的密钥区域和 HSQR 嵌入规则。

**模型与数据集（Models and Datasets）。** 基础生成模型为 Stable Diffusion v2.1-base，输出分辨率为 $512\times512$，生成时无分类器引导（CFG）强度为 7.5，采用 50 步 DDIM。反演同样使用 50 步，输入空提示词，引导强度为 0。基础生成与反演模型及 FFT 均使用 FP32。以 MS-COCO 图像描述为提示词，按来源图像 ID 划分互不重叠的拟合、阈值校准和测试集，规模分别为 1,024、2,000 和 1,000；同一图像的描述不跨集合。G0/G1 配对控制提示词、初始潜变量种子和密钥；同池内不同反演配置复用保存的生成图像。

**反演配置与基线（Inversion Configurations and Baselines）。** 表 1 给出 Q0–Q5 配置。G0 的生成与反演均关闭 FreeU，G1 的两个阶段均开启 FreeU，参数 $(s_1,s_2,b_1,b_2)=(0.9,0.2,1.4,1.6)$ [2]。DDIM/Complex-L1 是本文执行的 HSQR 基线（Reproduced SFWMark）；Exact-DDIM 使用受 Exact-DPM 启发的一阶前向步修正 [3]，步长为 0.5，每个反演步最多评估 5 个候选，包含 DDIM 初值。精修在批内最大残差 RMSE 不超过 $10^{-5}$ 时提前停止，逐图像保留局部 MSE 最小的已评估候选。GNRI 使用面向 SD2.1/DDIM 的适配实现 [4]，每步最多迭代 2 次，容差为 $10^{-4}$，先验权重为 0.1。GNRI 作为反演比较方法，其结果对应本实验的适配与预算。G1 的 DDIM/Complex-L1 是匹配 FreeU 配置的基线；G0/G1 的跨池差异同时涉及生成端和反演端。

**表 1. 反演与生成配置。** FreeU 的两列分别表示其在生成阶段和反演阶段的开关。每个 Q 配置均评价 Complex-L1、L2、Diag 和 Mahalanobis 四种距离。

| 配置 | 生成池 | 反演器 | 生成 FreeU | 反演 FreeU |
| --- | --- | --- | --- | --- |
| Q0 | G0 | DDIM | OFF | OFF |
| Q1 | G0 | Exact-DDIM | OFF | OFF |
| Q2 | G0 | GNRI | OFF | OFF |
| Q3 | G1 | DDIM | ON | ON |
| Q4 | G1 | Exact-DDIM | ON | ON |
| Q5 | G1 | GNRI | ON | ON |

**攻击设置（Attack Settings）。** 当前报告采用清洗后的 24 个条件。常规条件 Original-10 包含 clean、对比度（系数 0.5）、JPEG（质量 25）、高斯模糊（$5\times5$ 核，$\sigma=1$）、BM3D（$\sigma=0.1$）、VAE-B、VAE-C、Diffusion、中心裁剪和随机裁剪。VAE-B 与 VAE-C 分别采用 Bmshj18 和 Cheng20 压缩模型，质量等级均为 3。Diffusion 再生成采用 SD2.1，噪声时刻为 60、调度步数为 50、CFG 为 7.5，使用 FP16；噪声时刻与调度步数含义不同。中心和随机裁剪分别保留原图面积的 0.5 和 0.7，黑色填回原位置，不缩放。双线性旋转采用 $\{0,\pm5,\pm10,\pm15,\pm30,\pm45\}^{\circ}$，保持 $512\times512$ 画布，不扩展，边界填黑；$90^{\circ}$、$180^{\circ}$ 和 $270^{\circ}$ 采用无插值转置并单独报告。因此，24 个条件由 10 个常规条件、11 个双线性角度和 3 个正交角度组成，包含 clean 与双线性 $0^{\circ}$ 两个对照。历史实验中的亮度、噪声、双线性 $\pm75^{\circ}$ 和最近邻 $75^{\circ}$ 不计入本节结果或均值，历史 12 个非零旋转角度的均值也不用于当前报告。

**评价协议与指标（Evaluation Protocol and Metrics）。** 每个条件使用 1,000 对含水印/无水印测试图像，判断图像是否匹配指定密钥。四种距离共用恢复特征与密钥区域：Complex-L1 为复数残差模长的均值，L2 为未中心化欧氏距离，Diag 为中心化后按收缩协方差对角方差标准化的 L2，Mahalanobis 则使用完整协方差白化。每个 Q 配置分别以正确密钥残差估计均值与 Ledoit–Wolf 收缩协方差，所得统计量共用于全部测试条件。1,024 个拟合样本分为干净、JPEG、噪声（$\sigma=0.05$）、两种裁剪及 $|\theta|=5^{\circ}$、$15^{\circ}$、$30^{\circ}$ 旋转共八组，每组 128 例，旋转组内正负角度各半。拟合分布包含噪声，并不意味着噪声属于本节的 24 个测试条件；拟合过程只估计统计参数，不更新扩散模型权重。

主要指标为 TPR@1%FPR，按实际评价代码定义为测试 ROC 上满足 $\mathrm{FPR}<0.01$ 的最大 TPR，不对 ROC 作插值。以负距离作为正类分数，另报告 AUC。Frozen 指标使用独立的干净无水印校准集：对每个 Q 配置和距离，将 2,000 个校准距离从小到大排列，取第 20 个顺序统计量为阈值 $\gamma$，严格以 $s<\gamma$ 判为含指定水印，并将该阈值冻结用于所有攻击。Frozen TPR 和 Frozen FPR 因而反映固定阈值在分布变化后的实际检出与误报；名义 1% 校准预算不保证攻击后的 Frozen FPR 仍低于 1%。其余记录指标包括 Frozen Balanced Accuracy $=(\mathrm{TPR}+1-\mathrm{FPR})/2$ 和测试 ROC 最优阈值下的 MaxAcc。多密钥 Identification 在 2,048 个候选中排序，与指定密钥 Verification 的判别目标不同，本节不将两者的准确率混用。

所有跨条件均值均先逐条件计算指标，再对条件等权取算术平均，不将不同条件的样本混合重算 ROC。Original-10 均值包含 clean；双线性旋转均值仅包含十个非零角度，$0^{\circ}$ 和正交旋转均不混入。以下检测表基于最终逐条件数值重算，AUC 保留四位小数，百分数保留两位小数，百分数的差值使用百分点。逐条件源表已舍入，汇总末位可能与未舍入底层分数的计算略有差异；这些跨条件均值是点估计，不表示置信区间或显著性检验结果。

生成质量实验在 G0 和 G1 下各评价 1,000 张无水印图像与 1,000 张含 HSQR 水印图像，共 4,000 张。按相同 sample ID、提示词和潜变量种子形成 G1/G0 配对，分别在两种水印状态下计算 MUSIQ-AVA、LAION Aesthetic 和 CLIP 图文余弦相似度，三者均为高分更好。MUSIQ-AVA 使用 PyIQA 的 AVA 权重配置，LAION Aesthetic 使用 ViT-L/14 图像特征及相应美学回归头；CLIP 使用仓库的 OpenCLIP ViT-g-14、`laion2b_s12b_b42k` 权重配置，对归一化图文特征取内积，保持原始余弦尺度。质量评价实现对应研究仓库的 FreeU 评测提交 [5]，直接读取保存的生成图像及其提示词，不经过攻击或后续 DDIM 反演。对各组分数统计均值与样本标准差（$\mathrm{ddof}=1$），以每个 sample ID 的配对差值为重采样单位，采用 10,000 次有放回 Bootstrap、随机种子 1729，构造平均 $G1-G0$ 的逐项 95% 百分位置信区间。区间未进行多重比较校正。

## 5.2 与基线比较（Comparison with Baselines）

### 5.2.1 几何旋转下的水印验证鲁棒性（Rotation Robustness of Watermark Verification）

我们先在同一生成池内比较 DDIM/Complex-L1 与 Exact-DDIM/Mahalanobis，以检验反演精修与统计评分的联合配置能否改善旋转后的指定密钥验证。表 2 展示十个非零双线性角度的 TPR@1%FPR，同时列出 GNRI/Mahalanobis 比较结果。表中 L1 指 Complex-L1，Exact 指 Exact-DDIM，Mah 指 Mahalanobis。

**表 2. 双线性旋转下的 TPR@1%FPR（%）。** 每个角度评价相同的 1,000 对测试样本；末行为十个角度的等权均值。

| 旋转角度 | G0 DDIM/L1 | G0 Exact/Mah | G0 GNRI/Mah | G1 DDIM/L1 | G1 Exact/Mah | G1 GNRI/Mah |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| -45° | 5.10 | 10.70 | 10.50 | 4.30 | 9.60 | 7.80 |
| -30° | 5.80 | 18.60 | 14.20 | 5.50 | 22.60 | 16.80 |
| -15° | 3.90 | 18.00 | 15.80 | 3.30 | 20.80 | 16.20 |
| -10° | 1.20 | 8.00 | 6.70 | 0.50 | 9.80 | 10.80 |
| -5° | 8.90 | 28.90 | 30.90 | 7.60 | 31.30 | 26.50 |
| +5° | 4.20 | 32.90 | 30.50 | 9.60 | 29.00 | 25.70 |
| +10° | 0.80 | 9.60 | 8.10 | 0.70 | 10.80 | 8.30 |
| +15° | 3.40 | 19.10 | 18.00 | 5.10 | 17.50 | 16.90 |
| +30° | 5.70 | 8.80 | 8.30 | 3.40 | 15.60 | 14.40 |
| +45° | 4.60 | 8.80 | 6.70 | 4.20 | 10.80 | 11.90 |
| 10 个非零角度均值 | 4.36 | 16.34 | 14.97 | 4.42 | 17.78 | 15.53 |

G0 中，DDIM/Complex-L1 的平均 TPR 为 4.36%，Exact-DDIM/Mahalanobis 提高至 16.34%，增加 11.98 个百分点；G1 中由 4.42% 提高至 17.78%，增加 13.36 个百分点。改善覆盖表 2 的全部十个角度，但联合配置的平均检出率仍低于 20%，表明旋转后的低误报验证仍有明显困难。例如，$-30^{\circ}$ 下 G0 从 5.80% 提高至 18.60%，G1 从 5.50% 提高至 22.60%；$-45^{\circ}$ 下相应结果为 5.10% 至 10.70% 和 4.30% 至 9.60%。这些差异对应反演器与评分方式同时变化，单个组件的贡献需通过独立消融判断。

**表 3. 十个非零双线性角度上的 AUC 与固定阈值结果。** 各列分别对逐角度指标取平均，Frozen 指标使用 clean 校准阈值。

| 配置 | 平均 AUC | 平均 Frozen TPR (%) | 平均 Frozen FPR (%) |
| --- | ---: | ---: | ---: |
| Q0 Complex-L1 | 0.6418 | 2.72 | 0.62 |
| Q1 Mahalanobis | 0.8336 | 10.02 | 0.36 |
| Q2 Mahalanobis | 0.8237 | 10.75 | 0.51 |
| Q3 Complex-L1 | 0.6405 | 2.92 | 0.51 |
| Q4 Mahalanobis | 0.8355 | 11.11 | 0.41 |
| Q5 Mahalanobis | 0.8252 | 9.89 | 0.52 |

G0 的平均 AUC 从 0.6418 提高至 0.8336，G1 从 0.6405 提高至 0.8355，说明联合配置同时改善了总体区分能力。固定阈值下，G0 的平均 Frozen TPR 从 2.72% 提高至 10.02%，Frozen FPR 为 0.36%；G1 的 Frozen TPR 从 2.92% 提高至 11.11%，Frozen FPR 为 0.41%。Frozen TPR 低于对应的 TPR@1%FPR，体现出 clean 标定阈值与攻击条件 ROC 工作点的差异。在相同 Mahalanobis 评分下，GNRI 的旋转平均 TPR 为 G0 的 14.97% 和 G1 的 15.53%，分别低于 Exact-DDIM 的 16.34% 和 17.78%；这比较的是本节各反演器在已声明预算下的表现。

**表 4. 双线性零角度与正交旋转的 TPR@1%FPR（%）。** 这些条件单列，不参与表 2 的非零双线性均值。

| 条件 | G0 DDIM/L1 | G0 Exact/Mah | G1 DDIM/L1 | G1 Exact/Mah |
| --- | ---: | ---: | ---: | ---: |
| Bilinear 0° | 100.00 | 100.00 | 100.00 | 100.00 |
| Orthogonal 90° | 1.70 | 2.40 | 2.60 | 3.10 |
| Orthogonal 180° | 0.80 | 9.70 | 1.60 | 5.10 |
| Orthogonal 270° | 0.90 | 2.40 | 2.30 | 2.90 |

双线性 $0^{\circ}$ 对照下四个主要配置均为 100.00%；正交旋转后的检出率明显降低，即使没有插值，频域坐标对应关系的变化仍使验证困难。Exact-DDIM/Mahalanobis 的正交三角度平均 TPR 在 G0 为 4.83%、G1 为 3.70%，不能将双线性旋转结果外推为任意角度上的高检出率。旋转下的多密钥 Identification 仍较弱，指定密钥验证的改善不代表已经解决 2,048 候选的密钥身份识别。

### 5.2.2 常规攻击下的水印验证性能（Verification Performance under Conventional Distortions）

常规条件检验的是旋转验证收益是否伴随基础验证性能变化。表 5 在 G0/G1 内分别比较 DDIM/Complex-L1、Exact-DDIM/Complex-L1 和 Exact-DDIM/Mahalanobis，以区分保留原评分的反演精修配置与联合统计验证配置。以下数值均来自本研究执行的实验；SFWMark 发表论文（Published SFWMark）[1] 的攻击集合、部分实现与汇总协议不同，其发表值不并入本表作无条件优劣比较。

**表 5. Original-10 条件下的 TPR@1%FPR（%）。** 表中保留全部十个常规条件，包括 clean；负向结果与均值采用相同范围。

| 条件 | G0 DDIM/L1 | G0 Exact/L1 | G0 Exact/Mah | G1 DDIM/L1 | G1 Exact/L1 | G1 Exact/Mah |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| contrast | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| jpeg | 100.00 | 100.00 | 96.40 | 100.00 | 100.00 | 94.90 |
| blur | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| bm3d | 100.00 | 100.00 | 99.90 | 100.00 | 100.00 | 100.00 |
| vae_b | 99.90 | 100.00 | 91.70 | 99.50 | 100.00 | 89.00 |
| vae_c | 100.00 | 100.00 | 91.20 | 100.00 | 100.00 | 92.50 |
| diff | 100.00 | 100.00 | 99.50 | 100.00 | 100.00 | 98.90 |
| cc | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| rc | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Original-10 均值 | 99.99 | 100.00 | 97.87 | 99.95 | 100.00 | 97.53 |

Exact-DDIM/Complex-L1 在两池的十个条件中均达到 100.00%，保持了这一测试范围内原 HSQR 基线的验证表现。换用 Mahalanobis 后，G0 和 G1 的均值分别为 97.87% 和 97.53%，低于各自 DDIM/Complex-L1 的 99.99% 和 99.95%，差值分别为 −2.12 和 −2.42 个百分点。代价主要出现在 JPEG 与 VAE 再生成：G0 的 Exact-DDIM/Mahalanobis 在 JPEG、VAE-B 和 VAE-C 下为 96.40%、91.70% 和 91.20%，G1 为 94.90%、89.00% 和 92.50%，而相应 Exact-DDIM/Complex-L1 均为 100.00%。因此，旋转中更合适的残差权重并未在全部常规失真下保持相同的低误报检出率。

**表 6. Original-10 条件上的验证指标均值。** GNRI/Mahalanobis 同时列作反演比较；Frozen TPR 与 Frozen FPR 是分别取均值，不能据 AUC 替代固定阈值评价。

| 配置 | 平均 TPR@1%FPR (%) | 平均 AUC | 平均 Frozen TPR (%) | 平均 Frozen FPR (%) |
| --- | ---: | ---: | ---: | ---: |
| Q0 Complex-L1 | 99.99 | 1.0000 | 100.00 | 1.61 |
| Q1 Complex-L1 | 100.00 | 1.0000 | 100.00 | 1.56 |
| Q1 Mahalanobis | 97.87 | 0.9985 | 97.18 | 2.09 |
| Q2 Mahalanobis | 98.21 | 0.9986 | 97.74 | 2.28 |
| Q3 Complex-L1 | 99.95 | 0.9998 | 99.95 | 1.51 |
| Q4 Complex-L1 | 100.00 | 1.0000 | 100.00 | 1.30 |
| Q4 Mahalanobis | 97.53 | 0.9987 | 97.08 | 2.50 |
| Q5 Mahalanobis | 97.64 | 0.9988 | 97.15 | 2.39 |

较高 AUC 也不保证冻结阈值下的表现无损。G0 的 Exact-DDIM/Mahalanobis 平均 Frozen TPR/FPR 为 97.18%/2.09%，G1 为 97.08%/2.50%，误报均值超过名义 1% 校准预算。在 G1 的 BM3D 和 VAE-C 条件下，其 Frozen FPR 分别达到 8.20% 和 7.80%；Diffusion 条件的 Frozen TPR 为 85.20%，即便其 ROC 指标仍较高。这些结果说明，联合配置带来旋转验证收益的同时，常规攻击下的低误报检出与固定阈值迁移仍存在代价。

### 5.2.3 生成质量与水印验证的兼容性分析（Compatibility Analysis of Generation Quality and Watermark Verification）

兼容性分析回答两个相连的问题：FreeU 在含 HSQR 水印的生成条件下能否获得美学评分收益，以及反演与统计验证组合能否在相应配置下继续改善旋转验证。我们使用 RTX 4090D 上已完成的质量测试，在无水印与含水印状态下分别比较 1,000 对 G0/G1 图像，共 4,000 张。表 7 给出质量结果；验证部分沿用前述相同生成池的结果。

**表 7. FreeU 生成质量的配对比较。** 每一水印状态、每项指标均为 1,000 对 G0/G1 样本。CI 对应平均配对差值，分数及差值保留六位小数。

| 指标 | 图像类型 | G0 均值 | G0 SD | G1 均值 | G1 SD | 配对差值 G1−G0 | 差值的 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MUSIQ-AVA | 无水印 | 5.446280 | 0.499514 | 5.488538 | 0.492161 | +0.042258 | [0.017203, 0.067252] |
| MUSIQ-AVA | 含 HSQR 水印 | 5.439008 | 0.469536 | 5.471005 | 0.489975 | +0.031997 | [0.008107, 0.055921] |
| LAION-Aes | 无水印 | 5.499057 | 0.340628 | 5.526639 | 0.351266 | +0.027582 | [0.011038, 0.044362] |
| LAION-Aes | 含 HSQR 水印 | 5.493650 | 0.328696 | 5.510149 | 0.353016 | +0.016499 | [0.000356, 0.032819] |
| CLIP | 无水印 | 0.328547 | 0.038470 | 0.328486 | 0.039482 | −0.000061 | [−0.002020, 0.001880] |
| CLIP | 含 HSQR 水印 | 0.329180 | 0.039657 | 0.330251 | 0.039620 | +0.001070 | [−0.000744, 0.002898] |

表中 SD 为组内样本标准差（$\mathrm{ddof}=1$），由各组的 1,000 个图像分数计算；CI 对应配对平均差值的不确定性，与组内标准差含义不同。配对差值先在未舍入分数上计算，再与均值分别舍入。因此，含水印 CLIP 的两个已舍入均值相减与所报配对差值相差 $10^{-6}$。

含水印图像的 MUSIQ-AVA 从 5.439008 增至 5.471005，平均差值为 +0.031997，95% CI 为 [0.008107, 0.055921]；LAION-Aes 从 5.493650 增至 5.510149，差值为 +0.016499，95% CI 为 [0.000356, 0.032819]。两个逐项区间均在零以上，支持本配置下小幅正向的美学评分变化；区间未经多重比较校正，不能据此作全部比较的总体显著性判断。无水印条件下两项平均增量分别为 +0.042258 和 +0.027582。虽然含水印增量在数值上更小，尚未检验两类增量之差，不能据此断言水印显著削弱了 FreeU 效果。

CLIP 的无水印差值为 −0.000061，含水印差值为 +0.001070，两个 95% CI 均包含零。当前结果未排除零差异，既不支持语义一致性显著改善，也不足以证明两种配置严格等效。两项美学分数的增量较小，因而质量结论限于所用评价器上的平均收益，不将其描述为大幅画质提升。

结合表 2 与表 5，G1 配置的质量收益与旋转验证改善可以同时出现：相对于 G1 的 DDIM/Complex-L1，Exact-DDIM/Mahalanobis 的平均旋转 TPR 从 4.42% 提高至 17.78%，但 Original-10 均值由 99.95% 降至 97.53%。与 G0 的联合配置相比，G1 的旋转平均 TPR 从 16.34% 增至 17.78%，Original-10 均值从 97.87% 变为 97.53%。这些跨池差异同时包含生成与反演阶段的 FreeU 开关变化，不能将其全部归因于生成阶段。

这些观察支持所评估 SD2.1/HSQR 配置下的系统级兼容性：采用已有 FreeU 调制时，含水印生成可获得小幅美学评分收益，检测端联合配置在对应生成池中仍保持相对于匹配基线的旋转验证改善。其权衡是部分常规攻击上的验证性能下降，以及 clean 阈值在攻击分布下的误报变化。质量评分直接来自生成图像，反演器与统计评分只参与验证，结果不涉及新的水印感知 FreeU、联合训练或质量反馈优化。

**参考文献**

[1] Sung Ju Lee, Nam Ik Cho. [Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity](https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Semantic_Watermarking_Reinvented_Enhancing_Robustness_and_Generation_Quality_with_Fourier_ICCV_2025_paper.html). ICCV, 18759–18769, 2025.

[2] Chenyang Si, Ziqi Huang, Yuming Jiang, Ziwei Liu. [FreeU: Free Lunch in Diffusion U-Net](https://openaccess.thecvf.com/content/CVPR2024/html/Si_FreeU_Free_Lunch_in_Diffusion_U-Net_CVPR_2024_paper.html). CVPR, 4733–4743, 2024.

[3] Seongmin Hong, Kyeonghyun Lee, Suh Yoon Jeon, Hyewon Bae, Se Young Chun. [On Exact Inversion of DPM-Solvers](https://openaccess.thecvf.com/content/CVPR2024/html/Hong_On_Exact_Inversion_of_DPM-Solvers_CVPR_2024_paper.html). CVPR, 7069–7078, 2024.

[4] Dvir Samuel, Barak Meiri, Haggai Maron, Yoad Tewel, Nir Darshan, Shai Avidan, Gal Chechik, Rami Ben-Ari. [Lightning-Fast Image Inversion and Editing for Text-to-Image Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5f8ccd2cf8ca5876de06104e5f59f902-Abstract-Conference.html). ICLR, 2025.

[5] funfan-gif. [FreeU 图像质量评价实现](https://github.com/funfan-gif/SFWMark-Research/tree/3559e5fc908525120fb5e28be3abcbf23384c04c)，`src/freeu_quality_eval.py`、`src/freeu_quality_data.py` 与 `docs/freeu_quality_eval.md`。
