### Related Work 应当对已有研究进行分类、比较和归纳，让读者理解当前研究在整个领域中的位置，而不是逐篇复述摘要



### 优秀论文的 Related Work 通常不是按照“我的方法用了哪些技术”来组织，而是按照“读者需要了解哪些已有研究，才能理解本文的研究定位”来组织。



### **整体逻辑：几何失真影响水印恢复 → 恢复精修仍留下需要解释的残差 → 残差应根据任务目标形成不同决策。** FreeU 的说明保留在实现与辅助实验中即可；Related Work 无需展开迭代预算、停止阈值或重复实验数值。



### 它为什么这样写？

你需要先知道 SFWMark 的研究目标。

它的主要创新集中在水印嵌入设计，例如 Hermitian 对称与中心感知嵌入。

所以它的 Related Work 有一条非常明确的逻辑：



已有数字水印技术

空间域和频域嵌入各有哪些特点？

已有潜扩散水印

怎样结合生成过程？是否需要训练？

已有潜变量傅里叶水印

Tree-Ring、RingID 等如何设计频率图案？



SFWMark 关注的问题

现有嵌入过程中的频率完整性如何改善？

尤其注意第三小节的写法。

作者并不只是介绍 Tree-Ring 和 RingID，还指出在他们分析的先前方案中，逆傅里叶变换后的实值处理可能导致频域信息不完整。

这就自然引出了他们自己的研究：

既然某些已有频域嵌入方式会破坏频率完整性，那么就研究 Hermitian 对称能不能改善这个问题。



你可以学习它的组织方式，但不应该照搬它的三个小标题。

因为你们 TAIR-WM 的研究重点已经从水印嵌入转到了检测端。如果仍然用很大篇幅介绍频域嵌入技术，读者可能会误以为你们也准备提出一种新的嵌入机制。

## 二、Tree-Ring（NeurIPS 2023）：按照技术发展路线组织

[Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust](https://arxiv.org/html/2305.20030?utm_source=chatgpt.com)

NeurIPS 2023 · [官方论文](https://papers.nips.cc/paper_files/paper/2023/hash/b54d1757c190ba20dbc4f9e4a2f54149-Abstract-Conference.html?utm_source=chatgpt.com)

这篇论文的 Related Work 包括三个主题：

- Diffusion Models
- Watermarking Digital Content
- Fingerprinting and Watermarking Generative Models

这三个标题看起来不是按自己的算法模块来划分的，对吧？

因为作者想解释的是一种新的技术范式。

先前的许多方法是在图像生成后添加水印，而 Tree-Ring 将水印结构嵌入生成所用的初始噪声中。于是作者就需要介绍已有的数字水印方法，以及生成模型水印的不同技术路线。

这里尤其值得学习的是最后一个主题。

作者回顾了训练式水印编码器、生成模型内置水印等技术，然后强调 Tree-Ring 与这些技术在嵌入方式和训练需求上的区别。

它的写作目的不是普及生成模型知识，而是让读者理解为什么初始噪声水印构成了一条不同的技术路线。

对你们的启示是：Related Work 的小标题可以围绕真正的研究差异来设计，不需要跟 Methodology 的三个模块逐项对应。