# FreeU 图像质量测评（G0 / G1）

本工具只评价已有生成图像。Q0/Q1/Q2 共用 G0，Q3/Q4/Q5 共用 G1；每个池只计一次。主分析为 `wm`（HSQR），`no_wm` 为控制。脚本不调用扩散生成、反演、攻击或水印检测器，不修改原始 manifest、图像或既有结果。

## 1. 学校电脑先检查环境，再安装增量依赖

```bash
git fetch origin
git switch --track origin/exp/freeu-quality-eval
conda activate sfw
python -c "import sys,torch,torchvision,importlib.metadata as m; print(sys.executable); print({n:m.version(n) for n in ['torch','torchvision','diffusers','compressai','transformers','numpy']}); print({'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

若分支已在本机创建，改用 `git switch exp/freeu-quality-eval`。以上命令必须在学校电脑的 `sfw` 环境运行。本次开发电脑没有名为 `sfw` 的 Conda 环境，不能把开发机检查当作学校环境验证。仓库依赖记录为 torch 2.5.1、torchvision 0.20.1、diffusers 0.32.2、compressai 1.2.6、transformers 4.48.1；这只是基线记录，不是强制重装清单。

新增直接依赖仅为 `pyiqa==0.1.15`。该版官方约束接受现有 torch/torchvision/transformers 基线；旧版 PyIQA 可能要求降级 transformers，不能直接套用旧版安装命令。[官方 v0.1.15 依赖](https://github.com/chaofengc/IQA-PyTorch/blob/v0.1.15/requirements.txt)。不要重新执行仓库原 `requirements.txt`，不要升级 torch、CUDA、diffusers 或 compressai。

先在独立输出目录保存环境，并用全部现有包版本约束安装器：

```bash
python -c "from pathlib import Path; Path('freeu_quality_outputs/setup').mkdir(parents=True,exist_ok=True)"
python -c "import subprocess,pathlib; pathlib.Path('freeu_quality_outputs/setup/sfw-packages-before.json').write_bytes(subprocess.check_output(['python','-m','pip','list','--format=json']))"
python -c "import importlib.metadata as m,pathlib; rows=sorted(d.metadata['Name']+'=='+d.version for d in m.distributions() if d.metadata['Name'].lower()!='pyiqa'); pathlib.Path('freeu_quality_outputs/setup/sfw-constraints.txt').write_text('\n'.join(rows)+'\n',encoding='utf-8')"
python -m pip install --dry-run --report freeu_quality_outputs/setup/install-plan.json -c freeu_quality_outputs/setup/sfw-constraints.txt -r requirements-freeu-iqa.txt
```

查看 `install-plan.json`，只允许安装缺失依赖和指定的 PyIQA 版本。上述 constraints 固定全部已有包，解析冲突会报错，不应通过删掉 torch 等约束来强行安装。PyIQA 官方元数据还声明一些其他评价器和训练工具的依赖；pip 会安装实际缺少的传递依赖。保留官方依赖解析，不用 `--no-deps` 隐藏缺包。确认 dry-run 成功后执行：

```bash
python -m pip install --upgrade-strategy only-if-needed -c freeu_quality_outputs/setup/sfw-constraints.txt -r requirements-freeu-iqa.txt
python -m pip check
python -c "import pyiqa; names=pyiqa.list_models(); assert 'musiq-ava' in names and 'laion_aes' in names; print('PyIQA metric names OK')"
python -c "import subprocess,pathlib; pathlib.Path('freeu_quality_outputs/setup/sfw-packages-after.json').write_bytes(subprocess.check_output(['python','-m','pip','list','--format=json']))"
```

若已有依赖产生冲突，或不希望在原 `sfw` 中添加官方传递依赖，先克隆环境。下列方式保留原环境：

```bash
conda create --name freeu-iqa --clone sfw
conda activate freeu-iqa
```

在克隆环境重新执行版本检查、constraints 快照和 dry-run。若仍冲突，根据 dry-run 中具体冲突在克隆环境处理，保持 torch、torchvision、diffusers、compressai 固定；不要把克隆环境的修复命令回灌原 `sfw`。本仓库不宣称已验证所有机器的传递依赖组合。

## 2. 数据目录和正式证据

`--data-root` 指向包含 G0、G1 和 manifests 的同一个实验根目录，默认数据集为 coco：

```text
experiment/
  manifests/
    formal_splits.json
    generation_plan.json
    pairing_validation.json
  G0/coco/HSQR/
    generation_config.json
    generation_manifest.json
    generation_records/<sample_id>.json
    img_pil/<sample_id>.png
    img_pil_wm/<sample_id>.png
  G1/coco/HSQR/
    ...
```

默认 split 文件为 `manifests/formal_splits.json`；实际文件名不同时，用 `--split-manifest` 显式指定。生成计划默认 `manifests/generation_plan.json`，可用 `--generation-plan` 指定。必要时用 `--metadata` 指定复制后的 COCO annotations 文件。元数据、manifest 和图像均通过对应记录中的哈希验证，不重写原文件中的旧绝对路径。`--dataset-id` 可指定其他数据集目录名。

正式 test 必须来自已有正式 split manifest 的 1,000 个 sample_id，不能从 PNG 文件名推测，也不是固定的 0–999。预检逐项核对 prompt、prompt_hash、latent_seed、key_index、生成参数和 FreeU 池路由，并验证四组 PNG 的 SHA256、可读性、尺寸和格式。记录中原机器绝对路径失效时，脚本在当前实验根目录按预期池和相对位置重新定位，并且只有哈希一致才接受。不会仅凭文件名接受图片，不读取 latent 数组或运行检测模型。

缺图、哈希不符、重复/不一致样本或配对证据不充分会导致预检失败；不会自动补图或剔除样本。报告保留具体失败细节。正式模式只使用可验证的正式生成证据。

## 3. 按 preflight → smoke → run 使用

以下命令可直接在 Bash 或 PowerShell 使用（将路径替换为学校电脑实际目录）：

```bash
python src/freeu_quality_eval.py preflight --data-root /path/to/experiment --split test --output-dir /path/to/freeu_quality_outputs/preflight
python src/freeu_quality_eval.py smoke --data-root /path/to/experiment --split test --limit 20 --output-dir /path/to/freeu_quality_outputs/smoke
python src/freeu_quality_eval.py run --data-root /path/to/experiment --split test --output-dir /path/to/freeu_quality_outputs/full
```

preflight 只检查文件、哈希、协议、软件版本和 CUDA 状态，不加载大型评价模型。PASS 表示检查通过；FAIL 需要修复输入或补齐原始证据；WARNING 说明能力或证据限制。preflight 可以在尚未安装 PyIQA 时检查数据。默认输出根为 `<data-root>/freeu_quality_outputs`；建议按上例为预检、smoke 和正式测评指定不同子目录。

smoke 固定取正式测试清单中排序后的前 20 个样本，共 80 张图片。正式数据完整性先核查，再测固定前缀；不会跳过坏样本去凑 20 个。smoke 检查评价器加载、CUDA 推理、分数合法性和落盘。其结果只用于程序验证，不能用于选择有利指标、调整正式实验配置或替代全量结果。

run 默认评价三个指标和完整测试集。默认 `--device cuda --batch-size 1`；逐评价器加载并使用 `torch.inference_mode()`，避免同时保留大模型。CLIP ViT-g-14 体积较大，首次加载需要对应权重和足够显存。smoke 如需 CPU 功能诊断，可显式加 `--device cpu`，但 CPU 测试不能作为 CUDA smoke 成功证据。显存允许时可提高 batch-size；OOM 应停止并降低批量，用同一设置重试或新建输出目录，不能偷偷跳过该图片。

单独评价某一项时使用独立目录，实际启用指标写入 manifest：

```bash
python src/freeu_quality_eval.py run --data-root /path/to/experiment --metrics musiq-ava --output-dir /path/to/freeu_quality_outputs/musiq
python src/freeu_quality_eval.py run --data-root /path/to/experiment --metrics laion_aes --output-dir /path/to/freeu_quality_outputs/laion
python src/freeu_quality_eval.py run --data-root /path/to/experiment --metrics clip --output-dir /path/to/freeu_quality_outputs/clip
```

`--metrics musiq-ava laion_aes clip` 与逗号分隔形式均可；省略则三项全开。未启用的 CSV 指标列为空，不用 0 代替未计算。

## 4. 指标定义与权重

| 输出列 | 评价器 | 协议 |
| --- | --- | --- |
| musiq_ava | PyIQA `musiq-ava` | 官方 AVA 权重；高分更好 |
| laion_aes | PyIQA `laion_aes` | 官方 LAION Aesthetic 回归器；高分更好 |
| clip_similarity | 仓库 `src/open_clip` | ViT-g-14 / laion2b_s12b_b42k；匹配图文余弦相似度 |

`musiq` 默认对应 KonIQ，不能冒充 `musiq-ava`。AVA 模型权重标识为 `musiq_ava_ckpt-e8d3f067.pth`；LAION 使用 OpenAI CLIP ViT-L/14 图像特征及 `sac+logos+ava1-l14-linearMSE.pth` 回归头。[PyIQA 官方配置](https://github.com/chaofengc/IQA-PyTorch/blob/v0.1.15/pyiqa/default_model_configs.py)、[MUSIQ 实现](https://github.com/chaofengc/IQA-PyTorch/blob/v0.1.15/pyiqa/archs/musiq_arch.py)、[LAION 实现](https://github.com/chaofengc/IQA-PyTorch/blob/v0.1.15/pyiqa/archs/laion_aes_arch.py)。

PyIQA 输入为原图 RGB、float32、0–1；内部使用该评价器官方预处理，不预先对图片进行质量修复或挑选。分数必须有限；CLIP 为 -1–1 的余弦值，LAION 回归器的参考范围不是硬裁剪规则。

CLIP 沿用 `src/metric.py` 和 `src/utils.py:get_clip_score`：图像和对应 prompt 分别编码，对两种特征分别做 L2 归一化，然后逐行内积。没有 logit_scale、乘 100、负值截断或跨样本平均。使用仓库随附 open_clip 2.16.0 代码；预处理为模型返回的评测 transform（224 像素短边 bicubic resize、中心裁剪、RGB、ToTensor、CLIP mean/std），tokenizer 对应同一模型，context length 77。不能替换为 PyIQA `clipscore` 或其他 CLIP backbone。

首次 smoke/run 可能下载评价器权重到模型库缓存，缓存不写进生成池；preflight 不下载权重。学校网络需可访问相应官方源，或提前将官方缓存复制到目标机器。manifest 记录实际模型配置、软件/源码标识、预处理与权重指纹，以便续跑拒绝混用不同模型。不把模型权重、生成图像或完整测评结果提交到 GitHub。

## 5. 输出、统计和断点续跑

所有输出写进所选 `freeu_quality_outputs` 目录树。主要文件：

- `preflight_report.json`：检查状态、四组样本数、缺失/无效详情、配对证据及环境。
- `experiment_manifest.json`：代码 commit、时间、硬件/软件、评价器/权重、样本清单哈希、输出路径与实际指标。
- `per_image_scores.csv`：sample_id、generation_pool、watermark_status、image_path、image_sha256，以及三个指标列。
- `quality_summary.csv` / `quality_summary.json`：组均值和标准差、配对 G1−G0 差值、95% bootstrap CI、G1 胜率、数量和有效性。
- `paired_differences.csv`：按 sample_id、水印状态和指标保存 G0/G1 原分数及 G1−G0 差值。
- `scores.sqlite3`：每批提交的断点记录及完整性指纹，CSV/JSON 为其可读导出。

bootstrap 按 sample_id 对整对图像差值进行有放回重采样，默认随机种子 1729、10,000 次；不会独立打乱 G0 与 G1。差值为 G1−G0，高于 0 表示 FreeU ON 得分更高。区间含 0 只能说明当前证据未排除零差异，不能据此声称两种方法已证明等效。smoke 和 legacy 结果必须带各自限制，不合并为正式 1,000 样本结论。

中断后用**相同命令和输出目录**重跑，程序核查输入、设置、权重和已有记录后续算，已落盘的有效分数复用。改变指标、样本集、模型或统计设置请用新目录；不要删除原记录来强行混合。多个进程不能同时写同一输出目录。`.run.lock` 使用操作系统文件锁，进程结束后自动释放；锁文件继续存在不代表仍被占用，不需要删除它。

## 6. 历史数据：显式 legacy

没有完整 formal generation manifest 的历史图像只能显式加 `--legacy`。仍需已有的正式测试清单和生成计划，不能通过目录交集发明 test 样本。清单和计划只确定应评价的样本及原始 prompt；若缺少对应历史生成记录，不能将计划中的 seed/key 当作这张历史图像已执行的事实。可用的原始证据照实核查，无法验证的字段保持未知。脚本会记录当前图像哈希作为本次输入指纹，但新算出的哈希不能追溯证明历史生成来源。

```bash
python src/freeu_quality_eval.py preflight --data-root /path/to/legacy_experiment --split test --legacy --output-dir /path/to/freeu_quality_outputs/legacy_preflight
python src/freeu_quality_eval.py smoke --data-root /path/to/legacy_experiment --split test --legacy --limit 20 --output-dir /path/to/freeu_quality_outputs/legacy_smoke
```

legacy 的配对仅限记录中可验证的层面；相同 sample_id/文件名不证明使用相同 latent、seed 和 key。输出必须标注低可信度和缺失的历史证据，不能自动升级为 formal protocol，也不能将其配对差值视为严格受控因果比较。现有正式 manifest 若校验失败，`--legacy` 也不能绕过其哈希错误。

## 7. 不下载模型的代码测试

```bash
python -B -m unittest discover -s tests -p "test_freeu_*.py" -v
```

自动化测试使用独立临时图像和模拟评价器验证输入协议、重定位、哈希拒绝、固定样本、统计与续跑，不是三项真实预训练评价器的效果验证。学校电脑仍需执行上面的真实 CUDA smoke。不要根据 smoke 的分数趋势选择最终配置。
