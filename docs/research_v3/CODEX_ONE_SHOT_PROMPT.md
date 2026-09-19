请阅读 SFWMark_V3_Experiment_Plan.md，并直接在当前本地 SFWMark 研究仓库完成实现、测试和研究分支提交，不要仅输出计划，不要把审计单独作为本轮终点。

目标：保留原 DDIM，新增 ExactDPM Algorithm 1 的预算版 DDIM 求解器与 GNRI 的 DDIM 版本；接入匹配的 FreeU 生成/检测开关；增加独立fit数据估计的LedoitWolf白化马氏距离；将Rotation纳入正式评估。其他反演、Attention、旋转校正都不加入。

先读取当前AGENTS、git状态和真实代码。实际本地代码优先；复用已存在且经测试正确的白化/旋转实现。
按任务书完成3×2×2的12评分配置，但只提取6条GPU特征管线，生成只有FreeU off/on两套成对图片。缓存复用、断点续跑和输出隔离必须做好。

关键限制：
1. HSQR原注入不变；中心44×44空间区域先FFT；key特征与±45模板不变。
2. 原L1保持复数模差均值，不替换成real/imag拼接后的实L1。
3. M使用query/template共同白化后的L2；独立fit残差、独立calibration负样本、test禁止拟合；每个pipeline独立统计。
4. 新反演按论文与官方代码实现，不以普通DDIM、固定点基线或Adam冒充GNRI；不构造完整Jacobian。保留空prompt盲检条件，冻结模型权重。
5. FreeU在U-Net内部；生成和检测匹配；GNRI+FreeU必须验证真实梯度，不能被no_grad或原地操作悄悄破坏。
6. Rot75-NN显式nearest，曲线Rot-BL显式bilinear；都固定画布512、中心旋转、黑色补角。另设exact90，不自动旋回。
7. 原Avg是Clean+11攻击的12项平均；新增rotation不混进去。
8. 不升级/覆盖现有环境，不下载大模型替换现有镜像，缺少资源明确报错。保留原legacy结果，修复攻击问题需全组统一且另标版本。
9. 提供CPU单元测试、8对GPU smoke、pilot/full两种一键配置。无CUDA可先完成CPU测试并推送，但必须明确GPU未验证，4090运行脚本先过GPU门禁。
10. 可以把附件kernels作为CPU参考移植；它们不是已验证的完整SFWMark集成。

请在本次连续完成这些工作，允许内部自行安排子任务，不需要逐模块等待我的新一轮批准。涉及权限的操作遵循当前环境批准规则。

Git授权：允许创建/使用 exp/inv-freeu-mahal-rotation，commit并push到我自己的 funfan-gif/SFWMark-Research。先核对remote，绝不push到上游作者仓库。只提交本任务代码、配置、文档和脱敏测试记录；不上传模型、图片、keys、latent、输出缓存、.env或凭证；不force push，不改写历史，不覆盖其他未提交工作。权限失败时保留本地commit并准确报告。

结束时必须输出：实际修改文件、各算法采用的论文/代码入口、测试通过和未验证事项、commit SHA和真实push结果，以及4090 WSL+sfw环境里同步分支并启动pilot的准确命令。最终命令要与你实际创建的文件一致。不要未经同意自动启动全量full。
