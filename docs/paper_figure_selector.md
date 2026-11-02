# 可独立复制的论文图片候选筛选工具

`tools/select_paper_figures.py` 只读取现成 G0/G1 含水印 PNG 和已有三项评分，输出供人工审阅的候选图墙。脚本不导入仓库模块，不使用 PyTorch、CUDA、扩散模型、水印检测器或质量评价器；不修改生成、攻击或实验算法，不写入 Q0–Q5、原始图片或评分文件。

## Ubuntu / WSL2：复制一个脚本即可运行

使用 Python 3.10（兼容更新版本）。如果已有 Pillow 和 NumPy，可直接运行。缺少时只在独立虚拟环境安装这两个依赖，不需要重新安装 PyTorch：

```bash
python3.10 -m venv .venv-figures
source .venv-figures/bin/activate
python -m pip install 'Pillow>=9.1,<13' 'numpy>=1.23,<3'
```

将 `tools/select_paper_figures.py` 单独复制到目标机器任意工作目录，运行：

```bash
python select_paper_figures.py \
  --data-root /path/to/paper_v1_r2 \
  --scores /path/to/per_image_scores.csv \
  --differences /path/to/paired_differences.csv \
  --output-dir /path/to/outputs \
  --rotation 30 \
  --num-overview 20 \
  --seed 42
```

若在仓库目录运行，只需把脚本名换成 `tools/select_paper_figures.py`。不要求目标电脑切换 Git 分支。

## 输入与拒绝条件

正式路径固定为 `{data-root}/G0/coco/HSQR/img_pil_wm/{sample_id}.png` 和对应 G1 路径。默认评分必须恰好有 1,000 对；两组含水印评分与三项配对差值的 ID 集合必须一致，且每个入池 ID 在两个目录都有匹配 PNG。正式生成池也可能含 fit/calibration 图片：目录里未参与评分的额外 PNG 仅计数并排除，不会被加入候选或读取图像内容。默认不会取缺失配对的交集、跳过缺失/坏图或接受少于 1,000 对的 smoke 结果。

CSV 使用 `exp/freeu-quality-eval` 的原生格式（仅消费 `watermark_status=wm`）：

- `per_image_scores.csv`：`sample_id,generation_pool,watermark_status,image_path,image_sha256,musiq_ava,laion_aes,clip_similarity`，允许其他列和 `no_wm` 行。
- `paired_differences.csv`：`sample_id,watermark_status,metric,g0_score,g1_score,difference`，每个含水印 ID 的 `metric` 分别为 `musiq-ava`、`laion_aes`、`clip`；允许其他列。

三项分数全部必需，不填补空值。脚本核对重复记录、缺失配对、有限数值、评分范围、差值一致性、PNG SHA256 和 512×512 RGB 格式。历史 Windows/Linux 绝对路径只用于检查末尾五段路径；实际文件按 `--data-root` 重新定位，不读取旧电脑路径。如存在 `manifests/formal_splits.json`，额外核对 test 清单。没有该文件时，仅验证 CSV/PNG 成员关系；不会声称独立证明了正式生成来源、prompt、seed 或 key。

输出目录必须是新目录或空目录，不能位于 G0/G1、Q0–Q5 或 manifests 内，已有结果不覆盖。使用不同输出目录保留不同运行。验证或候选数不足时以退出码 2 拒绝运行；不会静默放宽阈值。

## 选择规则与边界

Overview 默认选 20 对。对 G0/G1 对称计算亮度、对比度、裁切高光/阴影比例、灰度熵、梯度、4×4 空间结构和旋转后的结构保留代理；保留质量评分中间 10%–90% 的样本，轮流覆盖质量四分位层，再结合可读性和颜色/空间描述子的差异选取。不会按 FreeU 增幅或水印验证结果排序。代理无法证明主体清晰度或语义多样性，最终仍需人工看图。

旋转已对照基础提交 `3559e5fc908525120fb5e28be3abcbf23384c04c` 的 `src/research_attacks.py`：512×512 固定黑底、中心裁切/填充、Pillow 双线性、默认中心 (256,256)、不扩展、正角逆时针。先在原分辨率旋转，再缩小显示；每对显示 G0、G1 和两张对应预览。预览不等于重新运行正式攻击/水印检测。

FreeU 默认每类 3 对（可用 `--num-freeu-per-category` 调整）。以三项 G1−G0 差值的总体标准差归一化后取均值；不减均值，保留差值方向。正/负类还要求至少两项原始指标同方向，在各自类别分布的 25%–75% 条件分位数附近选真实样本。近零类同时限制合成值和每项标准化差值，避免正负大幅抵消被误标为近零，按 RMS 接近零的程度选取。所有原始分数、差值和总体经验百分位均保留；指标有分歧时同样显示。

若某类别不存在或候选不足，输出实际数量和告警，不编造负向或近零样本；类别很小时提示其分位点可能与总体极端值重合。图墙仅是候选示例，不能代替 1,000 对的均值、置信区间和胜率统计。

## 四个输出与复现

- `overview_contact_sheet.png`：sample_id、G0/G1 含水印原图、两张旋转预览。
- `freeu_contact_sheet.png`：类别、sample_id、G0/G1 含水印图、三项 G0/G1 分数与差值。
- `selected_candidates.csv`：入选顺序、类别、路径及哈希、全部分数与差值、经验百分位、分位点目标、图像统计。
- `selection_protocol.md`：算法与固定阈值、参数/种子、输入和脚本哈希、总体 ID/PNG 哈希摘要、软件版本、验证范围、缺失类别告警和所有入选 ID。

同一版本软件、输入、参数和种子产生相同入选顺序与图墙；跨 Pillow/NumPy 版本可能有像素或浮点差异，因此记录实际版本。输出到不同目录会改变协议中的路径字段。没有网络访问或权重下载。

## 只使用模拟数据的测试

```bash
python -B -m unittest discover -s tests -p 'test_select_paper_figures.py' -v
```

测试在临时目录生成模拟 PNG/CSV，验证独立复制后的 CLI、四个输出、确定性、原始文件不变、分位数选择、缺失类别、错误输入和现有旋转代码逐像素一致。小数据集必须显式使用 `--expected-pairs N`，协议会标注 `NON-FORMAL SIZE OVERRIDE`。不要用此选项将 smoke 结果冒充正式筛选。本开发工作不对真实实验运行筛选；不要将数据 PNG、评分 CSV 或生成图墙提交到 GitHub。

开发验证环境：Windows 上的 Python 3.10.11 / Pillow 10.4.0 / NumPy 1.26.4，以及 Python 3.13.12 / Pillow 12.3.0 / NumPy 2.5.2。21 项模拟测试通过，并人工检查了模拟图墙的文字和图片布局。未执行真实 1,000 对筛选；本机没有已注册的 WSL 发行版，Ubuntu WSL2 运行尚未实测。
