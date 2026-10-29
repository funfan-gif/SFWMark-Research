# TAIR-WM 论文与写作材料

整理上传日期：2026-10-29。当前修订稿与实验稿更新于 2026-10-28。

当前稿件由论文前半部分和实验 5.1/5.2 两份文件组成，分别列在下表。摘要、引言和相关工作另行导出，方便在 GitHub 中阅读和继续修改。

| 内容 | 文件 |
| --- | --- |
| 当前修订稿：标题、摘要、引言、相关工作、方法、图 1、算法 1 及参考文献 | [中文 PDF](TAIR-WM_PreExperimental_Revised_CN.pdf) |
| 实验设置与基线比较，包含旋转、常规攻击及 FreeU 兼容性分析 | [实验 5.1/5.2](TAIR-WM_Experimental_5.1_5.2_CN.md) |
| 可继续编辑的当前论文源文件 | [TeX 源文件](source/TAIR-WM_PreExperimental_Revised_CN.tex) |
| 当前摘要 | [摘要 Markdown](sections/abstract_cn.md) |
| 当前引言及三项贡献 | [引言 Markdown](sections/introduction_cn.md) |
| 当前相关工作 | [相关工作 Markdown](sections/related_work_cn.md) |

## 当前研究定位与数据口径

三项贡献分别为残差引导的 Exact-DDIM 反演精修、协方差感知的 Mahalanobis 指定密钥验证，以及生成质量与水印验证的兼容性分析。FreeU 是已有推理时调制技术，在 G1 的生成与反演阶段同时开启。

当前检测结果采用清洗后的 24 个条件，双线性旋转均值只包含十个非零角度。质量实验共 4,000 张图像，在含水印与无水印两种状态下分别形成 1,000 对 G0/G1 样本。质量表已根据逐样本结果核验均值、样本标准差（ddof=1）、配对差值与 95% Bootstrap 置信区间。

| 数据 | 文件 |
| --- | --- |
| Q0–Q5 × 24 条件 × 4 距离 × 7 指标的最终汇总 | [清洗后逐条件结果](data/SFWMark_Q0-Q5_24_conditions_7_metrics_CN.md) |
| 12 个组统计与 6 个配对比较的原精度汇总 | [quality_summary.csv](data/quality_summary.csv) |
| 6,000 条 sample ID × 水印状态 × 质量指标的配对记录 | [paired_differences.csv](data/paired_differences.csv) |

## 写作笔记

下列笔记保留形成时的讨论，其中部分内容早于当前研究定位。论文内容和实验口径以本目录的当前修订稿、实验稿与数据为准。

- [引言写作](notes/introduction_writing_notes_cn.md)
- [相关工作写作](notes/related_work_writing_notes_cn.md)
- [方法与 HSQR 维度说明](notes/method_writing_notes_cn.md)
- [实验写作](notes/experiment_writing_notes_cn.md)
- [FreeU 兼容性贡献的定位讨论](notes/freeu_compatibility_notes_cn.md)

## 历史完整稿

以下完整稿来自 2026-10-24，早于当前贡献重构与 24 条件结果更新。其中的任务感知评分叙事和历史旋转均值应按版本理解，当前实验结果以新版实验稿为准。

- [历史中文完整稿](archive/2026-10-24/TAIR-WM_Chinese_Final.pdf)
- [历史英文完整稿](archive/2026-10-24/TAIR-WM_English_Final.pdf)

## 编辑与编译

当前 TeX 为中文双栏稿，图 1 使用 TikZ，参考文献随源文件保存。使用 XeLaTeX，需安装 SimSun 中文字体、TeX Gyre Termes/Herôs 字体及源文件所列 LaTeX 宏包。连续编译两次可更新交叉引用。

```bash
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build source/TAIR-WM_PreExperimental_Revised_CN.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build source/TAIR-WM_PreExperimental_Revised_CN.tex
```

研究实现见仓库的 [exp/inv-freeu-mahal-rotation](https://github.com/funfan-gif/SFWMark-Research/tree/exp/inv-freeu-mahal-rotation) 与 [exp/freeu-quality-eval](https://github.com/funfan-gif/SFWMark-Research/tree/exp/freeu-quality-eval) 分支。质量评价实现对应提交 [3559e5f](https://github.com/funfan-gif/SFWMark-Research/commit/3559e5fc908525120fb5e28be3abcbf23384c04c)。
