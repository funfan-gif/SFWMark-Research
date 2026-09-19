# SFWMark V3：双反演、FreeU、白化马氏距离与旋转鲁棒性

日期：2026-09-17
状态：实验设计与 Codex 实施规范；不代表已在用户仓库中实现，亦不代表已有性能提升。
本文件替代旧任务书的“首次只审计、分多次等待批准”安排。按用户最新需求，用一次本机实施任务完成审计、改码、测试和研究分支推送，再在 4090 主机运行小规模配置。网络/权限审批仍按 Codex 环境规则办理。

## 1. 研究问题与边界

主问题：保持 SFWMark 的 HSQR 嵌入、中心区域、key 容量不变，更准确的反演、FreeU 特征重加权和误差协方差驱动的匹配，是否改善旋转后的水印验证与识别，并且不增加误报或破坏生成质量？

本轮选择的两种“新增反演”为：
- ExactDPM 的 DDIM Algorithm 1 求解部分，采用论文 forward-step 方法。工程名 `exact_ddim_budgeted`。
- Guided Newton-Raphson Inversion 的 DDIM 版本。工程名 `gnri_ddim_budgeted`。

同时必须保留原始 `ddim` 作为第三个水平的基线，不把两个新增方法串联。
本轮不实现 Direct/PnP、PGID、Attention、自适应嵌入或旋转校正/角度搜索。
主模型保持 SD2.1-base，生成 DDIM50、CFG7.5；检测空 prompt、guidance=0。原始 prompt、攻击真角度、真实生成噪声、正确身份标签不能传入反演器。验证时可接收声明的 reference key；识别时接收完整注册池而不接收正确 key 标签。

不能预先声称白化必然改善旋转。马氏距离具有在 query、reference 和 covariance 同时作匹配坐标变换时的协变性质，并不意味着仅旋转输入图片后得分不变。插值、裁边和非线性 VAE/U-Net 更不能自动由白化逆转。

“无需模型训练”需要精确表述为：冻结扩散模型权重，允许逐图潜变量优化，并需要离线协方差拟合与独立阈值校准。不能写成完全没有任何拟合数据。

## 2. 事实依据与本地审计范围

已核查的上游是 thomas11809/SFWMark。研究仓库 funfan-gif/SFWMark-Research 本次通过连接器读取返回 404，无法据此判断它不存在，也不能声称已审计用户最新改动。本地代码是实施时的真实依据。

已知用户在 WSL Ubuntu、conda `sfw`、RTX 4090 系列 24GB 环境跑通 HSQR。曾成功使用：
`sd2-community/stable-diffusion-2-1-base`
revision `4e63672c03103b6c636b8fb4119ba982469b2955`
实施时检查本地 cache、模型和 revision；不要覆盖现有环境、默认重新下载模型或强制升级依赖。

上游核心文件：
- src/generate.py：key 池、每图 key 分配、生成成对 non-WM / WM 图片。
- src/utils.py：make_hsqr_pattern / inject_hsqr、pil2latent / ddim_invert、fft / center_slice、get_distance_hsqr、攻击与质量函数。
- src/detect.py：读取图像和 keys，制造攻击，反演，中心 FFT，验证得分与 2048-key 搜索，保存结果。
- src/metric.py：CLIP、COCO FID。
- src/diff_attack/diff_wm_attack.py：单独制造再生成攻击图。
- src/results/results_detect.py：读取 src/results 下既有结果并汇总，不是自动消费所有新的 outputs。
- src/test.sh：批量串起生成、质量评估、再生成攻击和检测。

已核实的具体问题：
1. detect.py 将攻击条件、1000 样本、npz 文件名写死，并会无条件读取 Diff 图像。小规模 Clean/Rotation 不应被这个文件依赖阻塞。
2. detect.py 和 ddim_invert 有 no_grad，GNRI 不能沿这条路径假装做了求导。
3. generate.py 用 batch_start 派生随机数，直接改变 batch_size 可能改变每图噪声。新协议用逐样本 generator；旧行为单独保留作 legacy 对照。
4. HSTR/HSQR 是先截空间中心 [10:54,10:54] 再做 44×44 FFT，而非整个 64×64 FFT 后裁低频。
5. HSQR 实际评分使用通道 3 的 42×21 个复数点，模板实部来自 QR 左半、虚部来自右半，模板幅值为 ±45。
6. 上游所谓 L1 是复数模差的均值 mean(abs(q_complex-t_complex))。它不等于拼接 real/imag 后的普通实向量 L1。
7. 上游最终 Avg 是 Clean+11 攻击共 12 项平均，称 `Avg-original12`；仅 11 个攻击的均值另称 `Avg-attacks11`，不能混用。
8. 原 ROC 是测试分数的描述性统计，严格选 FPR<0.01。保留 paper_roc 模式，并额外提供独立阈值校准结果。
9. utils.py 导入时加载一些攻击/评价模型，需懒加载以避免不需要它们时仍占显存；保留接口并回归测试。
10. 核查 Gaussian noise 分支的 uint8 转换、PSNR 的 uint8 运算和随机攻击配对。修复时使用明确的新 protocol version，所有方法同一版本；不要悄悄只修新方法。

本轮最小可交付以 HSQR 为主。HSTR 保持原 L1 路径可用；若实现 HSTR 的马氏距离，应单独处理每个通道及 Hermitian 冗余，保留原 channel_min 汇合规则，并重新拟合。未完成时明确报 NotImplementedError，不能把 HSQR 的 1764 维白化器套给 HSTR。

## 3. 总体因果链与实验矩阵

实际管线：
生成噪声 -> 原 HSQR 嵌入 -> U-Net 生成（FreeU off/on）
-> 已保存图像 -> 同一攻击协议 -> VAE.mode()
-> 选择一个反演器（其 U-Net 同样使用约定的 FreeU 状态）
-> 中心 FFT / 原 key 特征 -> 缓存特征
-> 原始复数 L1 或白化马氏距离 -> 验证/识别。

FreeU 是 U-Net 内部设置，不是反演结束后再串接的一层。

完整矩阵为 3×2×2 = 12 个评分配置：

| 特征缓存 | 反演 | 生成 FreeU | 检测 FreeU | 评分 |
|---|---|---|---|---|
| Q0 | 原 DDIM | off | off | L1 / M |
| Q1 | Exact-DDIM | off | off | L1 / M |
| Q2 | GNRI-DDIM | off | off | L1 / M |
| Q3 | 原 DDIM | on | on | L1 / M |
| Q4 | Exact-DDIM | on | on | L1 / M |
| Q5 | GNRI-DDIM | on | on | L1 / M |

因此同一批样本只需 6 条特征提取管线，不是为 12 种评分分别跑扩散。
图像生成只有 2 个模式：G0(FreeU off)、G1(FreeU on)。
每种生成模式都有无水印和有水印图。三种反演共享同一模式的图片。
L1/M/L2/diagonal 对同一 Q 缓存离线评分，不再调用 U-Net。

不要以“某组不好看”为由删除该消融。快筛和正式测试是不同数据用途：
- 小规模先确认实现、时间、明显退化；不得把它的低误报率结果作为论文结论。
- 正式配置冻结后完整跑主矩阵；若资源不足，只能预先宣布缩小正式攻击集/样本数，并在论文如实说明。

补充机制对照（几乎不增加扩散开销）：
- 同特征的普通实向量 L2：区分从 L1 换成 L2 的效应。
- diag(Sigma_regularized)：区分按方差缩放与利用非对角相关性的效应。
- 原始 L1 与上述量不能比较绝对数值大小来说明优劣，应比较 Id-Acc / 同 FPR 约束下 TPR。
- FreeU 位置诊断：只在小规模独立 dev 子集增加“生成off、检测on”与“生成on、检测off”，一次只改一侧。若未做，主文只能归因于匹配 FreeU 管线，不能声称单独来自检测端。

## 4. 两种反演的具体实施要求

### 4.1 公共接口

`invert_latent(pipe, z0, inverter_config, conditioning) -> InversionResult`

z0 来自相同 VAE.mode()*scaling_factor。返回未 FFT 的 [B,4,64,64] zT。
结果同时包含 status、实际 U-Net forward 次数、backward 次数、耗时、峰值显存、各步残差/迭代数。
公共函数不能接收 wm_label/key_index/attack_angle/reference_zT。

冻结所有模型权重；仅 GNRI 对当前潜变量求梯度。每一步 detach，禁止构造整条50步反向图。
微批默认1。无水印与水印串行也可；禁止默认一次塞入所有攻击和 keys。
比较时统一模型权重、VAE、scheduler timesteps、prediction_type、归一化、精度。残差/FFT至少FP32，协方差拟合FP64。
如果为了梯度稳定从半精度改为FP32，要重跑同精度 DDIM 参考，不能把精度差当算法收益。

### 4.2 ExactDPM -> exact_ddim_budgeted

来源：On Exact Inversion of DPM-Solvers，Algorithm 1 / Sec.4.1；官方 smhongok/inv-dpm 的 `src/stable_diffusion/inverse_stable_diffusion.py` 中 forward_diffusion、fixedpoint_correction 可作交叉核对。

移植的是 DDIM 对应的隐式逆问题及 forward-step 更新；不是换成一个名字带 DPM 的 inverse scheduler。
固定当前低噪状态 z_low，初始化高噪候选 u，反复计算原去噪步 D(u) 与 z_low 的差，按论文方法更新 u。
必须从本项目实际生成 scheduler 构造 D；不能照抄不同 scheduler 的 timestep-20、alpha/sigma 或 epsilon/v_prediction 假设。
允许自适应减小步长，输出最优有限残差状态，记录是否收敛；不能无条件使用最后一次发散结果。
起步预算为每步最多8次修正；这是预算版，不是原文完整收敛配置，不保证 exact。
本轮不加入 decoder inversion，不改变 VAE 编码，不切换生成采样器。命名必须说明 solver-only/budgeted。

关键限制：生成有真实 prompt/CFG7.5，而检测使用空 prompt/CFG0；即使数值逆求解充分收敛，也不能因此承诺恢复原始真实噪声。

### 4.3 GNRI -> gnri_ddim_budgeted

来源：Lightning-Fast Image Inversion and Editing for Text-to-Image Diffusion Models，arXiv:2312.12540v5，Eq.(7)-(10) 和其 DDIM prior。
官方仓库 dvirsamuel/NewtonRaphsonInversion 默认示例为 SDXL-Turbo；`src/FixedPointInversion` 是固定点基线，不能改名充当 GNRI。

按论文构造标量残差范数 + lambda*prior，并采用 GNRI 的逐分量更新与稳定项。
不能把普通 Adam/MSE 梯度下降命名 GNRI，不能计算 16384×16384 完整 Jacobian。
逐一核实 DDIM 的累计 alpha 与跨采样间隔的 transition 参数；不能把 Euler prior 原封不动套入DDIM。
起步每步最多4次修正、prior weight从论文0.1参考值开始；均需在dev核验，不代表水印最优参数。
对比 lambda=0 的小诊断应命名 NRI，不当作第三种主反演。
检查 prior 是否削弱水印特征；其目标是合理潜变量而非直接保水印。

FreeU 开启时也要跑真实 autograd smoke test；若库实现的原地特征修改造成求导冲突，应采用数值等价的非原地实现并回归测试，不得悄悄关闭 FreeU 或梯度。

### 4.4 三项必须通过的验收

- 数值小测试：合成线性去噪映射有解析逆，验证 timestep、缩放和逆求解方向正确。
- oracle sanity：只作为工程诊断，用已知生成条件与真实中间latent验证逆过程，不可进入主评估。
- 真正的 prompt-free smoke：只有图像，跑 HSQR Clean/Rotation，验证返回zT、梯度、显存和得分通路。
不能因为 oracle 重建改善就宣布水印盲检改善。
无论收敛与否都记录样本，不得丢弃困难样本。出错时显式标记失败，不默默退回DDIM并冠以新方法名字。

## 5. FreeU 的位置与参数

采用当前 diffusers 可用的内置实现，并保存版本/实现标识。
SD2.1 文档参考参数：
s1=0.9, s2=0.2, b1=1.1, b2=1.2。
这只是起点，不是已在HSQR上验证的最优值。

G0/Q0-Q2 都关闭；G1/Q3-Q5 的生成与检测都开启相同参数。
生成后再只开检测FreeU不等于匹配管线，应明确命名为位置诊断。
不改变初始 HSQR 注入本身；FreeU改变生成和反演中U-Net的特征重加权。
所有开关关闭必须回到原基线；开/关FreeU需要干净状态或上下文管理，不能污染下一组模型。
FreeU参数全1作为数值诊断可以近似中性，但不能未经测试假定与disable逐bit一致。
改变生成模式就必须重新生成匹配的正负图片、攻击图、协方差和阈值。

## 6. 白化马氏距离：本轮指定实现

### 6.1 特征与距离

HSQR query和模板都使用同一个42×21复数区域。
f = [flatten(real); flatten(imag)]，维度D=1764。
对候选key k，delta_k=f_query - f_template_k。

主距离：
d_M^2 = delta_k^T Sigma_reg^{-1} delta_k。
实现可保存 d_M^2/D，取负作为验证分数。它与d_M排序等价，但所有阈值需要针对实际分数重新校准。

Sigma_reg = (1-rho)*S + rho*trace(S)/D*I + epsilon*I。
rho默认 LedoitWolf 自动估计；epsilon=1e-6*trace(S_reg_without_ridge)/D。
使用Cholesky/triangular solve或等价稳定分解，不显式逆矩阵。
两边都白化：W delta = W query - W template。不能只白化query。
白化后用普通L2；不能再套一次原Sigma^{-1}形成重复白化。
本轮主分支不再加入White-L1。

### 6.2 Sigma 从哪里来

在独立 covariance-fit 集上构造：
r_i = f_query_i - f_template_of_known_fit_key_i。
仅拟合阶段允许用已知正确fit key标签。
对r做样本中心化以估计协方差S；评分仍采用pairwise delta，不额外减去fit均值。
均值校正是另一种方法，需另立消融；本轮只记录mean norm，不偷偷增加。

使用共享的、跨key pooled residual covariance，不做2048个每key协方差。
每条Q0-Q5管线有自己的Sigma文件：改变反演、FreeU、精度、scheduler或特征提取都要重拟合。
所有候选keys共享该矩阵；测试不根据真实旋转角度选择矩阵。
同一模型的校准不能读取正式test图片、标签、角度或统计量。

拟合数据主协议：每张fit基础水印图只产生一个攻击版本，按固定随机种子平衡分配以下8个类别：
Clean、JPEG25、Noise0.05、CC0.5、RC0.7、Rot(±5,BL)、Rot(±15,BL)、Rot(±30,BL)。
旋转类别内正负角均衡，BL=bilinear固定画布黑色补角。
实际Noise/裁剪必须说明使用legacy或修复后的protocol version，并对全部比较组一致。
不把同一张图的20种攻击算作20个独立拟合样本。
75度不进入拟合，用作未见角度压力测试；Rot75-NN同时涉及未见插值，不能只归因于角度泛化。
可追加 clean-only Sigma 对照判断旋转数据校准的必要性，需另提取独立fit图的Clean缓存；不作为首次开跑阻塞项。

记录fit样本数、维数、rho、特征次序、特征区域、均值范数、特征值范围、条件数、fit manifest hash、模型/管线hash。
白化模块需要伴随fit统计文件；单有代码不能直接跑M评分。

### 6.3 自带CPU参考核

`kernels/metric_core.py` 已包含：
- 原HSQR模板转换和中心区域坐标参考；
- 原复数L1；
- LedoitWolf正则白化；
- shared query/template变换和平方马氏距离；
- 独立负样本的经验阈值校准。

这些是CPU参考核，不是已完成的SFWMark集成。生产路径建议保留原torch特征提取并按缓存块评分。模板白化结果每个Sigma只算一次。

## 7. Rotation 正式纳入实验

攻击发生在保存的最终RGB图片上，VAE之前；不是旋转key/latent，不修改prompt。
所有方法与正负样本使用相同角度、插值、补角和尺寸协议。

A. 文献兼容点：
Rot75-NN：固定+75度、nearest、expand=False、center=None、fill=(0,0,0)、保持512×512、无后续resize。
Tree-Ring的RandomRotation((75,75))是固定角，不是uniform(-75,75)；显式nearest避免默认差异。

B. 小到大角度曲线：
Rot-BL：0、±5、±10、±15、±30、±45、±75，bilinear、其他参数同上。
完整输出正负角；展示绝对角均值时保留原数据。
角度增大不保证损害或识别率单调变化，尤其接近90度时不能这样解释。

C. 方向诊断：
Rot90-exact / 180 / 270，采用像素transpose，不插值、不缩放。与普通任意角旋转分组报告。

本轮不增加逆旋转、角度估计或角度搜索。它们是额外防御模块，不能混进“白化+反演”的收益。
0度必须identity；有水印/无水印同参变换；随机数调用顺序不能随方法变化。
输出中保存完整RotationSpec及原/攻击图像hash。

主表：原12条件分别保留，追加Rot75-NN列，不将其混入官方Avg。
曲线：固定正式test子集300对，只对预先由dev选择的主方法与关键消融做；子集不得按成功/失败筛选。未经测试的方法留空，不用推测补齐。
建议曲线至少包括DDIM+L1、DDIM+M、I*+L1、I*+M、I*+FreeU+L1、I*+FreeU+M。其中同管线两距离共享反演。
本轮的rotation只是一项攻击，不叫新增“距离指标”。

## 8. 数据划分与统计评估

正式建议：
- covariance-fit：1024张独立水印图（每生成模式同样的基础样本定义），每张1个攻击版本。
- dev：200对，选择预算/FreeU少量参数、确定曲线主方法；不用于最终结论。
- threshold-calibration：2000张独立无水印图，每张1个预声明混合攻击版本。
- test：1000对，正式原攻击表与Rot75；其中固定300对用于角度曲线。
- smoke：8对，可为专用合成prompt，不进入正式统计。

COCO按原图group_id分组拆分，多条caption不可跨split。不同split的生成seed、基础图像和group_id不重复。某图的所有攻击只能处于一个split。
注册key池允许所有split共享；这不等于图像泄露。但不要把已知key池结论冒充未注册key泛化。
不同方法之间反而应共用同一样本的prompt、seed、key、attack seed，这叫配对比较。
检查实际数据量是否足够，不能硬从DB1k的1001条prompt生成彼此“独立”的4224个prompt组。不足就报告并补合法来源，不复制split。
复用旧数据/缓存必须核查manifest和完整配置指纹，无法核验的旧结果仅作探索，不冒充正式复现。

小规模pilot可用fit256、dev64、calibration512，仅为实现/时间/趋势检查，不作精确1%FPR或显著性结论。不提前打开正式test进行调参。正式配置是另一套可执行profile而非反复向用户索取多阶段批准。

主要指标：
1. 单声明key verification：paper ROC AUC/TPR@FPR<1%用于与作者的描述性协议对齐。
2. calibrated verification：在独立calibration negatives上每个完整pipeline+distance单独确定阈值，冻结后报告test TPR和实际FPR。
3. pooled calibration使用上述8类预声明混合，按样本只1种攻击，不把同图多个版本当独立negative。测试时不读取真攻击类别来选阈值。
4. 2048-key closed-set identification：返回全部候选得分的argmin；正确key标签仅由评分器计算Id-Acc。
5. 需要open-set any-key detection时，必须为整个min-over-2048规则重新校准，不能沿用单key阈值。
6. 95%配对bootstrap置信区间按基础sample/group重采样；同图各角度视图不能当独立样本。冻结阈值bootstrap只描述条件性能不确定性；阈值估计不确定性需另说明。
7. 失败样本不能丢失，报告coverage/failure rate；Id-Acc分母包含失败。TPR/FPR给出失败处理规则与保守敏感性结果，不能用删除失败获得高分。

反演与代价：
- 记录实际UNet forward/backward次数、VAE调用、每图latency、峰值显存、不收敛率。
- 对主要方案添加普通DDIM更多反演步数的开发集成本对照，在相近时间预算下比较；更多步不保证更好，需要实测。
- 计时前后cuda.synchronize，预热后再测。不能把NFE当作梯度方法全部代价。
- 解耦日志：生成/攻击/反演/特征/模板预计算/距离评分/拟合各自耗时。
- 16图实测中位时间后估算总任务规模，并写预算文件；不得承诺在一个晚上跑完。
完整6管线、1000正负对、13条件仅推理数量就是6×1000×2×13=156000次图像反演，不含fit/calibration。缩小批次不减少总调用数；缓存和先跑pilot才是合理办法。

## 9. 后续论文证据链

不要先写“必然有效”再筛数据。预设待验证问题与可支持的结论：

| 问题 | 必需对照 | 应保存的证据 | 不能越界的结论 |
|---|---|---|---|
| 新反演有用吗 | Q1/Q2 vs Q0，L1固定 | latent恢复误差、Id-Acc、TPR/FPR、耗时 | 图片重建好不等于水印恢复好 |
| M是否有用 | 每条Q的M vs L1 | 同FPR性能、正确key到最近错误key的相对间隔 | 原始距离变小不等于更好 |
| M收益来自相关性吗 | full M vs diagonal vs L2 | 同数据、同正则配置的性能 | 不能把换L2的收益全称为去相关收益 |
| FreeU是否有用 | Q3 vs Q0，Q4 vs Q1，Q5 vs Q2 | 鲁棒性+生成质量，匹配正负样本 | 没做位置诊断就不能归因于检测端 |
| 模块是否互补 | 完整12评分矩阵 | 主效应与配对差异；必要时交互项 | 联合最高不等于存在超加性协同 |
| 旋转结论怎样 | 多角度、未见角度、exact90 | 正/负角曲线、插值协议、错key分布 | 白化不等于旋转不变性 |
| 泛化性如何 | 冻结参数/统计后的新prompt来源 | 新数据集同协议结果 | 不重新在该test上调参数 |

可解释性指标：
- 正确key margin = min_wrong d - d_true，>0表示正确key更近；跨不同距离量纲比较需标准化，优先比较margin>0比例。
- 只用评估器保存的真实zT计算未对齐噪声恢复误差，不把原始zT传给算法。旋转后误差同时包含几何位移，不应解读成纯数值求解误差。
- HSQR的±45是检测模板，不是实际生成噪声幅值；不要用模板代替真实zT算恢复NMSE。
- 残差相关性/eigenspectrum可用于机制分析，但“在fit集被白化”本来就是构造结果；需要held-out分析和实际检测结果一起支持。
- 同一生成模式的图片相同，所以反演/距离变化不改变FID或CLIP。只需对G0/G1的固定生成图片评价一次；若正式对标原文FID，采用原文COCO5000等规模并注明数据来源。

论文应把ExactDPM/GNRI/FreeU明确归因于已有工作。组合和新评估本身不自动构成新颖算法。可能的贡献是严谨验证“反演误差结构—统计匹配”的交互及边界，但需完整结果和进一步文献核对。

## 10. 最小工程改法：新增研究入口，保留legacy

建议新增：
src/experiments/run_suite.py          # 统一编排、resume、profile、缓存
src/experiments/inverters.py          # ddim / exact_ddim_budgeted / gnri_ddim_budgeted
src/experiments/sfw_adapter.py        # 调用原嵌入、原空间中心FFT与特征提取
src/experiments/metric_core.py        # L1 / L2 / diagonal / Mahalanobis
src/experiments/attack_registry.py    # 原攻击包装 + 明确RotationSpec
src/experiments/fit_statistics.py    # fit协方差与calibration阈值严格分开
src/experiments/report.py            # 按metadata汇总，不依赖固定列数
configs/research_v3_pilot.json
configs/research_v3_full.json
scripts/run_research_v3.sh
tests/test_research_v3.py
docs/research_v3_audit.md
docs/research_v3_runbook.md
THIRD_PARTY_NOTICES.md

不要重复编写三套检测循环。旧detect.py保留或作为legacy wrapper，新runner把提取与评分拆开。
L1路径必须与原get_distance_hsqr数值一致；支持weights关闭时复现，允许事先声明的浮点容差。
`rotation_eval.py`若本地已存在应复用并接入统一数据/评分API，不再制造第二套逻辑。

拟支持命令（Codex实施前不可宣称现成可用）：
`bash scripts/run_research_v3.sh --profile pilot --resume`
`bash scripts/run_research_v3.sh --profile full --resume`
内部应自动执行：检查环境与数据 -> 生成/复用 -> 攻击/缓存 -> 6种反演缓存 -> fit -> calibrate -> 多距离评分 -> 汇总。
单独支持extract/score以利用已有缓存，不可让更改distance触发重反演。
full默认必须显式启动，不由pilot自动跑到几天的工作量。pilot可以无人看守执行，但不能后台承诺而不真实启动进程。

## 11. 输出与缓存契约

建议：
outputs/research_v3/
  manifests/
  images/<generation_hash>/<split>/{wm,no_wm}/
  attacks/<generation_hash>/<split>/<attack_hash>/
  features/<pipeline_hash>/<split>/<attack_hash>/
  statistics/<pipeline_hash>/{covariance,thresholds,...}
  scores/<pipeline_hash>/<metric>/<split>/
  reports/

pipeline hash包含：模型权重/revision、VAE、scheduler完整配置、dtype、生成FreeU配置、检测FreeU配置、inverter/budget/容差/prior、特征版本、代码版本。
cache的sample部分还包含原图hash、attack hash及split。部分写入使用临时文件+原子rename；失败不能留伪成功缓存。
metric变化不改变feature hash；covariance/threshold另有独立hash及fit/calibration清单指纹。

每条评分记录至少：
sample_id, split, generation_hash, inverter, freeu_gen, freeu_det,
attack_id, angle, interpolation, metric, score_claimed_key, pred_key,
is_correct（仅评估层）, latency, status, image_hash, feature_hash, covariance_hash。
保存逐样本数据，不只保存均值；所有npz应包含attack_names和sample_ids。
实际水印key和patterns不提交公共仓库。

## 12. 一次 Codex 实施、一次 GPU 启动

### 本机一次任务的要求
先读本地AGENTS与git状态，随即完成实现，不停在审计文档等下一轮批准。
已有可复用白化/rotation实现先审计，满足本规范就复用，不因目录名不同而重写。
保留用户未提交改动；只stage本任务文件。
运行CPU测试和可用环境中的小GPU smoke；无CUDA时明确记录GPU未验证，生成4090端必跑门禁。
算法若无法核验要显式说明阻塞点，不用stub或普通DDIM冒充成功。
建立研究分支 `exp/inv-freeu-mahal-rotation`，保留基线分支不动。
允许按用户授权commit并push此研究分支，但先确认目标为用户自己的仓库。
不要push到thomas11809/SFWMark，不要force push，不要改写历史，不要上传token、.env、图像、keys、模型、缓存或整个outputs。
如果origin不是用户研究库，保留它并用明确的新remote指向用户库；没有权限时保留本地commit并准确报告，不能换用其他目标。
输出实际commit、push结果、测试状态、可复制的4090命令。

### 4090端一次启动
检查工作区是否干净和目标分支，不覆盖另一台机器未提交内容。
在实际WSL路径/conda sfw中同步分支。
跑8对smoke，验证FreeU+GNRI梯度和显存后启动pilot。
估算full工作量，正式完整测试由用户显式启动。长任务要有日志、可resume；不要求用户逐模块进行许多轮批准。

## 13. 自带核的验收范围

CPU_TEST_REPORT.txt 记录了在本会话环境运行的20项测试，包括：
复数L1定义、HSQR区域与模板映射、白化恒等式、马氏二次型一致性、秩不足正则化、
阈值ties、0度identity、exact90、nearest/bilinear区别与正负样本同参。

这些测试不验证SD2.1权重、GNRI/Exact迁移、FreeU梯度、CUDA性能或最终鲁棒性。
不能把这些CPU测试通过写成“已复现完整新方案”。

## 14. 一手来源

SFWMark论文：https://arxiv.org/abs/2509.07647
SFWMark代码：https://github.com/thomas11809/SFWMark
上游detect：https://github.com/thomas11809/SFWMark/blob/main/src/detect.py
上游utils：https://github.com/thomas11809/SFWMark/blob/main/src/utils.py
ExactDPM论文：https://arxiv.org/html/2311.18387v1
ExactDPM项目：https://smhongok.github.io/inv-dpm.html
ExactDPM代码：https://github.com/smhongok/inv-dpm
GNRI论文：https://arxiv.org/html/2312.12540v5
GNRI代码：https://github.com/dvirsamuel/NewtonRaphsonInversion
FreeU论文：https://openaccess.thecvf.com/content/CVPR2024/html/Si_FreeU_Free_Lunch_in_Diffusion_U-Net_CVPR_2024_paper.html
FreeU接口：https://huggingface.co/docs/diffusers/main/en/using-diffusers/freeu
白化理论：https://arxiv.org/abs/1512.00809
协方差估计：https://scikit-learn.org/stable/modules/covariance.html
LedoitWolf接口：https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
Tree-Ring源码：https://github.com/YuxinWenRick/tree-ring-watermark/blob/main/optim_utils.py
Rotation参数：https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomRotation.html
RST专门方法作为范围参照：https://arxiv.org/abs/2507.21195
