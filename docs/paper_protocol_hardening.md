# Paper-ready protocol hardening：实现报告与 RTX 4090 操作说明

## 结论与边界

本次实现的是 `paper_v1.0` 协议入口、验证链、统计与导出，不是新的水印算法。
旧入口仍可用于 pilot/debug；论文流程必须通过 `--formal` 或 `paper_workflow.py`。
保留 DDIM、已批准的 first-order forward-step refinement、GNRI 单样本 scalar Newton 公式、HSQR 定义、官方攻击参数、旋转实现及固定 FreeU 参数。

当前状态：**FORMAL EXPERIMENT GATE: FAIL**。
代码实现与 CPU 合成测试不能代替真实 COCO 分组检查、真实模型验收和完整正式 raw data 验证。
本机没有执行模型下载、依赖安装、GPU 正式实验、Git commit 或 push，也没有删除旧图片/结果。

## A. 文件清单

修改已有文件：

| 文件 | 作用 |
|---|---|
| `.gitignore` | 隔离 paper_v1、dev400、formal4024 运行输出，避免生成结果污染代码工作区 |
| `src/inversion.py` | GNRI 单图执行、执行版本、逐图收敛/数值失败记录；不改 Newton 公式 |
| `src/hsqr_metrics.py` | HSQR FFT 边界 FP32；LedoitWolf(store_precision=False) |
| `src/utils.py` | FFT dtype 保护、辅助模型延迟加载并保护加载时 RNG 状态 |
| `src/generate.py` | VAE slicing、dtype 元数据、拒绝覆盖正式冻结池 |
| `src/diff_attack/diff_wm_attack.py` | 旧入口跳过已有 Diff 输出前检查输入/配置/输出哈希；拒绝操作冻结池 |
| `src/research_hsqr.py` | 正式分流、calibrate、paper_all 和正式协议 CLI 参数 |
| `src/research_summary.py` | ROC 边界统一为严格 FPR < 0.01；旧自动搜索仅用于 legacy |
| `src/runtime_profiling.py` | 请求数、缓存数、实际反演数及明确的反演计时字段 |

新增文件：

| 文件 | 作用 |
|---|---|
| `src/paper_protocol.py` | split/plan、真实 metadata 反查、哈希封存、模型/环境/代码来源、配对与 Diff 校验 |
| `src/paper_generation.py` | 独立 seed 生成、逐图 GT latent、冻结配对、旧池审计/显式 promotion |
| `src/validate_generation_pairing.py` | 配对验证 CLI |
| `src/paper_runner.py` | 正式 fit/calibrate/test、模型兼容检查、单图特征/失败记录、不可覆盖 raw |
| `src/paper_diff_attack.py` | 原 Diff 攻击的固定批次执行与严格来源校验 |
| `src/paper_statistics.py` | 严格 ROC、只用 calibration 的冻结阈值、paired bootstrap |
| `src/paper_export.py` | 显式 run_manifest 校验、原始数据重算表格/图片、来源清单 |
| `src/paper_gate.py` | CPU 测试、显式真实模型小验收、最终正式门禁 |
| `src/paper_workflow.py` | prepare/run/export 三阶段编排，无 mtime 搜索 |
| `tests/test_paper_protocol.py` | 协议拒错、可重复统计导出与可用依赖下的数值测试 |
| `docs/paper_protocol_hardening.md` | 本报告与操作说明 |

原有用户文档 `docs/research_v3/` 保留，不覆盖。涉及冲突时以本次最新要求为准：FreeU 固定 0.9/0.2/1.4/1.6、covariance distance 减 residual mean、calibration 只用 Clean no-wm。

## B. 23 项风险、修复与验证

| Task | 原风险 | 实现与验证 |
|---|---|---|
| 1 | GNRI 在批次内汇总 objective，缓存改变输入批次即改变结果 | 保留单样本 Newton 函数，逐图调用后 cat；cache_dict 加 per_sample_v2。数值测试比较 alone/batch，并调用旧 runner 的真实缓存分派函数比较全 miss/部分 hit；本机依赖不足，未执行数值部分 |
| 2 | 保留 best latent 后失去失败信息 | 记录每步迭代数、残差/objective、总次数、converged/non_finite/hit_max_iterations/failure_reason；数值失败保留 raw，正式统计拒绝。显式 allow_failed_samples 输出 diagnostic 覆盖率 |
| 3 | CUDA half 的 44×44 FFT 不受支持 | formal 锁 FP32；注入、抽取、通用 FFT/rFFT 边界保护 half。Diff 攻击仍用原 fp16。增加真实 tensor 测试，当前 skipped |
| 4 | 范围不重叠但同 COCO 图片的 caption 跨 split | 固定 dev 0–399，排除其全部 group；fit1024/cal2000/test1000 各 group 一条。保存真实字段来源、seed、dataset hash、具体 ID；启动/导出时从 metadata 重建核对。无可靠字段或数量不足直接停止。合成分组测试通过 |
| 5 | Q/FreeU/步数不同却复用 whitening | sidecar 绑定完整 base protocol、fit ID、assignment、残差数、均值、shrinkage、jitter、NPZ 哈希；加载逐项比对；错误协议测试通过 |
| 6 | 同一 fit 图片多个攻击版本重复进入 covariance | 按 V3 已声明八类 mixture，固定 seed 均衡分派，一图一个 residual；assignment 独立封存并从预声明规则重建核对。1024 个样本每类128；旋转类别正负各64。W_clean 全 Clean |
| 7 | seed 依赖批次起点，补跑改变 latent | 局部 torch.Generator，seed=42+sample_id，先逐图建 latent 再组合；G0/G1 共用 plan/key/prompt。保存 no-wm/wm 的 .npy SHA256，全部配对后冻结；独立生成数值测试 skipped，配对拒错测试通过 |
| 8 | 原图变化而旧 Diff 被静默跳过 | 每图绑定 source hash、generation manifest、攻击模型指纹及参数、output hash；固定完整批次恢复，已有输出验证后才能 skip。输入/配置/输出篡改拒错测试通过 |
| 9 | <=1% 与原基线 <1% 混用 | 新指标版本和严格 FPR < .01；人工边界 ROC 测试通过 |
| 10 | 在 test 上选 operational threshold | 校准函数只接收 Clean no-wm calibration scores；Q×metric 独立阈值，所有 test attack 共用冻结阈值；导出重算 calibration 阈值验证。ties/边界/接口隔离测试通过 |
| 11 | 依照 mtime 挑最新攻击造成混协议 | formal 只读显式 manifest，每项 raw 检查 signature、IDs、Q、NPZ 与 JSON 一致；Original-12 缺项拒绝 Avg。混协议/缺项测试通过 |
| 12 | pilot 与 paper 自动混读 | 新 paper_v1 根目录、root manifest、禁止 legacy 输出目录；正式导出不 glob 搜索结果 |
| 13 | 只有汇总值，无法审计失败和识别 | 成对逐图 JSON 保存 group/key/四距离/全部候选 argmin/反演记录/cache 状态；并保留 NPZ，绑定文件哈希。GT 仅用于 claimed-key detector 分数和事后 accuracy，不用于 identification 候选筛选 |
| 14 | 无 CI、比较时样本不配对 | sample_id 成对 bootstrap，默认10000次、固定seed1729；方法差值使用同一重采样下标；测试重复导出/相同方法差值为0通过。Original-12 Avg 只提供点估计，CI 字段明确未计算，不平均单攻击 CI 端点 |
| 15 | 手填表格、图来源不明确 | 从锁定 raw 重算 main/rotation/runtime 表、method differences 和 SVG；export_manifest 保存 run/signature/raw hashes/commit/时间/产物 hashes。合成端到端 CSV 重复生成字节一致 |
| 16 | cache/load/不同执行量混作 seconds/image | 模型加载单列；反演 CUDA 同步计时，实际图数/缓存命中/UNet calls/峰值显存单列。formal benchmark 强制 no_cache、合成灰图预热、batch1、完整请求队列。恢复时重测已完成攻击并核对原 scores/predictions，不覆盖 raw |
| 17 | import utils 就加载多组 GPU 模型 | VAE-B/VAE-C/LPIPS 首次使用才构造，原 factory/weights 保留；fork_rng 避免首次模型构造影响外部 RNG；实际模型等价性须在完整环境验收 |
| 18 | generation VAE decode 高峰 | legacy 与新 generation 均 enable_vae_slicing，写入 manifest；不作为 Q 研究变量 |
| 19 | 警告后仍误跑正式数据 | formal 要求 clean commit、通过的真实模型 acceptance、split/plan/模型/环境一致、配对冻结、兼容 whitening/阈值、合法 Diff。错误即停止；gate 失败返回非零退出码 |
| 20 | 测试 skipped 也被称为通过 | 测试与验收明确区分 skipped、FAIL、NOT_RUN；任意 skip 无法获得 implementation acceptance |
| 21 | 来源只记录可变模型名称 | 保存代码 SHA/dirty、Python/ML包/CUDA/GPU、数据/plan/split/模型/whitening hash；本地模型 hash 配置和权重，不冒称 HF revision；远端名称必须固定40位 commit 且 local-only |
| 22 | 正式参数/名称漂移 | Q0–Q5 不变、D0/D1 不允许 formal、FreeU 值及 FP32 强锁；Exact 保留完整 first-order inspired 名称；GNRI 标注 adaptation/per-sample |
| 23 | 安全修复偷偷改算法 | 保留原 reference/44×44/42×21/1764维/channel3/2048候选、DDIM/forward refinement/攻击参数/旋转，不增加去旋转或角度搜索；代码差异与对应数值测试分开报告 |

### 两个必须明确的统计语义

1. GNRI **数值执行成功不等于每步收敛**。预算用尽但输出有限，会记录 `converged=false`、`hit_max_iterations=true`，不会伪报收敛；当前保持原预算型 solver 的输出语义，不把未达 tolerance 自动改成另一种算法。非有限更新/输出、异常分母、异常执行、缺失分数属于失败，正式统计拒绝。表内 `n_nonconverged` 统计未完全收敛的图数（no-wm/wm 各算一图），需在论文方法与限制中报告。
2. `TPR@1%FPR` 是 test ROC 上用于基线比较的曲线指标；`Frozen TPR/FPR/Balanced Accuracy` 使用独立 calibration 冻结的阈值。后者不使用 test 来选阈值。2000 个 calibration negatives 最多允许19个严格小于阈值；并列距离整体排除。

## C. 旧数据的资格

- dev400：保留为开发/pilot；其图片 group 的其他 caption 同样不能进正式 split。
- 旧 formal4024：不删除，不因目录名自动承认正式资格。只有同期记录能证明两池 sample_start/batch/seed 一致、GT latent/key/prompt/输出哈希对应、没有部分重生成导致 shift、模型与参数明确，并覆盖新的分组隔离 split，才可能 promotion。
- 旧 GNRI cache：不是 per_sample_v2，不能复用到正式流程。
- 未绑定输入来源的旧 Diff、混合 mtime summary、没有独立 calibration 的旧阈值：不能直接进入正式论文。
- 单纯现在重新计算 PNG 哈希，不能证明它当年由哪个 latent 生成。证据缺失时结果是 `legacy_not_formal`，保留 pilot，建议新生成。

旧池工具（仅在掌握真实历史证据时使用；不会自动扫描或猜测证据）：

```powershell
python -B src/validate_generation_pairing.py --audit_legacy
python -B src/validate_generation_pairing.py --audit_legacy --evidence D:/evidence/legacy-evidence.json
python -B src/validate_generation_pairing.py --promote_legacy --evidence D:/evidence/legacy-evidence.json --output_dir D:/paper-legacy-frozen
```

证据 JSON 必须是 `paper_protocol.write_once` 的 sealed 格式，顶层包含 `plan`、`G0`、`G1`、`historical_ledgers` 的文件路径与 SHA256。两池 manifest 使用 `paper_generation.assemble_pool` 相同的字段结构；历史 ledger 必须绑定每幅原图/latent 的记录以及 sample_start/batch_size/seed_scheme。promotion 还要求 plan 引用真实 group-isolated split。工具只是验证证据的一致性，不能替代对历史记录真实性的人工确认。缺少记录时不能手工捏造后“通过”。

promotion 只增加冻结标记和新 manifest，不改旧图、latent 或其字节。它属于显式单独入口；下方新实验流程默认 per_sample_v2，不自动做旧数据迁移。

## D. 最终流程

```text
clean commit + 本地模型/环境 + 真实 COCO metadata
  -> split manifest（dev 排除；fit1024 / cal2000 / test1000，group 隔离）
  -> generation plan（每 sample 固定 prompt/key/seed）
  -> CPU tests + 显式 GPU 小验收（IMPLEMENTATION ACCEPTANCE）
  -> G0/G1 generation（per_sample_v2，FP32，VAE slicing）
  -> 全部 latent/key/prompt/image provenance 配对验证 -> 冻结两池
  -> G0/G1 test Diff（原 fp16 攻击；输入/输出 provenance）
  -> Q0..Q5 fit（各自1024，一图一个预声明 residual）
  -> Q0..Q5 calibration（2000 Clean no-wm；冻结4种阈值）
  -> Q0..Q5 test（1000；Original-12 + 单列 Rotation；全2048-key argmin）
  -> immutable per-sample JSON + NPZ + explicit run_manifest
  -> failed/missing/protocol/配对/来源一致性检查
  -> paired bootstrap（10000；同样本索引）
  -> summary（Original-12 Avg 与 Rotation 分开）
  -> paper export（表、SVG、所有 source hash）
  -> 最终 FORMAL EXPERIMENT GATE
```

## E. RTX 4090 的完整命令顺序（本次未执行）

以下从**已同步本次代码、已有正确依赖、模型已经下载到本地**的主机开始。这里不提供自动安装/下载，也没有替你执行实验。
路径仅作占位，必须换成4090主机真实路径。请先将本次代码审阅并正常提交，使工作区 clean；本次没有替你 commit/push。

### 1. 打开项目根目录，核实版本与环境

```powershell
Set-Location D:/SFWMark-Research
git branch --show-current
git rev-parse HEAD
git status --short
python --version
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -B -m unittest discover -s tests -v
```

`git status --short` 必须无输出；测试不得有 skipped/failure。不要为了过关忽略缺依赖。确保本地缓存已有原 VAE-B/VAE-C 压缩模型及基线所需依赖，保留原权重，不在运行中更换版本。

### 2. 设置实际路径并离线准备

```powershell
$paperRoot = Join-Path (Get-Location) 'paper_v1'
$cocoMeta = 'D:/datasets/coco/meta_data.json'
$sdBase = 'D:/models/stable-diffusion-2-1-base'
$diffModel = 'D:/models/stable-diffusion-2-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Test-Path -LiteralPath $cocoMeta
Test-Path -LiteralPath $sdBase
Test-Path -LiteralPath $diffModel
python -B src/paper_workflow.py prepare --output_dir $paperRoot --metadata $cocoMeta --model_id $sdBase --diff_attack_model_id $diffModel --generation_batch_size 1 --whitening_name W_robust
```

`$diffModel` 必须指向原 Diff attack 使用的模型：旧 runner 默认是 `stabilityai/stable-diffusion-2-1`，而生成模型是 `stabilityai/stable-diffusion-2-1-base`，两者不能为省事互换。使用对应原模型的本地完整镜像。Diff 加载 dtype 固定 fp16，正式生成/反演仍 FP32。

prepare 做三件事：创建不可变 split/plan、执行 CPU tests、执行小规模真实模型 smoke（两池各一个生成 pair、Q0–Q5 反演及 GNRI 单图/批次比较）。它**不会启动4024图正式生成**。该步骤本次未在本机执行。

metadata 自动识别 `image_id` 或 `group_id`。若报没有可靠字段，先核实真实结构，再用 `--group_field 已人工确认的原图字段`；不要用 caption 或 annotation id 顶替。数量不足必须补正确数据，不能减样本量绕过。

此时成功应看到 `IMPLEMENTATION ACCEPTANCE: PASS`。由于正式结果尚未产生，`FORMAL EXPERIMENT GATE: FAIL` 仍正常；只有最终导出后才可能全流程 PASS。prepare 在 implementation 失败时返回非零并停止。

### 3. 人工检查准备清单

```powershell
Get-Content -LiteralPath (Join-Path $paperRoot 'manifests/formal_splits.json')
Get-Content -LiteralPath (Join-Path $paperRoot 'logs/implementation_acceptance.json')
```

检查 dev400/fit1024/cal2000/test1000、真实 group-id source、dataset hash、GPU 为目标设备、所有测试无 skip、smoke PASS。不要手工改 manifest 或 acceptance；修改配置应使用独立新输出根目录。

### 4. 显式开始完整实验

```powershell
python -B src/paper_workflow.py run --output_dir $paperRoot
```

自动顺序：G0 fit/cal/test 生成 → G1 fit/cal/test 生成 → 全部配对冻结 → 两池 test Diff → Q0至Q5各自 fit/calibrate/evaluate。每个子进程独立结束后释放其模型。正式 evaluation 用 `paper_all`、`no_cache`、预热和单图反演；不进入 D0/D1，不开启 oracle。

这一步是完整正式实验，不是4090 smoke，可能耗时很长；本次没有跑，也不能承诺固定时长或峰值显存。4090 上先通过 prepare，观察实际峰值。generation_batch_size=1 表示一个 sample 的 no-wm/wm pair（pipeline 实际两图），VAE slicing 启用；Diff 保持原 batch8，不擅自缩小它来掩盖显存问题。

不要同时启动两份写同一输出目录的任务；不要在运行期间更换 commit、依赖、模型或 metadata。

### 5. 显式统计、出图和最终验收

```powershell
python -B src/paper_workflow.py export --output_dir $paperRoot
Get-Content -LiteralPath (Join-Path $paperRoot 'logs/formal_gate.json')
```

默认10000次 bootstrap。输出：

```text
paper_v1/
  manifests/       split、generation plan、pairing、六组显式产物指针
  G0/ G1/          原图、GT latents、generation records、Diff 与来源
  Q0/ ... Q5/      models、calibration、features、failures、runs、runtime
  tables/          main_results.csv/json、rotation_results.csv/json
                   runtime_results.csv、method_differences.json
  figures/         rotation_curve.svg、method_comparison.svg
  export_manifest.json
  logs/            各阶段日志、implementation_acceptance、formal_gate
```

### 6. 中断、补跑与拒错处理

- 正常中断后可重新执行同一条 `run`：已完成 generation record 逐个验 hash；Diff 按原完整批次恢复；已有 raw 不覆盖。
- 正式计时恢复会重新测量此前完成的攻击，确认 score 在容差内一致且预测完全一致，再继续缺失攻击，避免把部分运行当全程 benchmark。
- 若写文件中途硬中断留下只有 PNG/NPZ、没有 manifest 的孤立文件，程序会拒绝猜测其来源。保留现场，不自动删除或覆盖。当前实现不自动修复这种事务中断，需要人工核查后选择独立新目录。
- 若 fit 已完成但进程在写 workflow 指针前终止，原 whitening 仍保留。程序不会按最新文件自动认领；应人工验证明确产物，再恢复明确指针，或使用新根目录。不要修改现有结果冒充新协议。
- 失败的 acceptance/report 也不可覆盖。修复环境/代码后使用独立新输出目录重新验收，并确保该目录在仓库外或被忽略，避免正式阶段因工作区变脏而拒绝。
- raw 中任何数值失败默认不能正式导出。若只为排查问题，可用下方显式 diagnostic 命令；输出写入 diagnostic 子目录，不能变成正式 PASS。

```powershell
python -B src/paper_export.py --run_manifest D:/paper_v1/Q2/runs/真实run_id/run_manifest.json --output_dir D:/paper_debug --allow_failed_samples --bootstrap_resamples 10000
```

此处必须是明确的 manifest 路径，不能用“最新文件”代替。正式6组来源由 `manifests/Q0-run.json` 等指针固定。

## F. FORMAL EXPERIMENT READINESS CHECKLIST

| 检查 | 当前状态 | 原因/通过条件 |
|---|---|---|
| Python syntax/static | PASS（50个Python文件 AST + diff whitespace 检查） | 不是完整依赖解析或GPU运行验证 |
| CPU 协议/统计/导出 tests | 21项中17项通过，4项skipped | 包含真实 validator 的小规模合成链，不是正式数据；无失败项 |
| GNRI alone/batch/cache 数值一致性 | NOT VERIFIED | 本机 Torch/Diffusers/torchvision 缺失，测试 skipped |
| generation partition 数值一致性 | NOT VERIFIED | 同上；不能以代码结构代替实际 tensor 测试 |
| half 44×44 FFT 与2048候选数值测试 | NOT VERIFIED | 本机缺相关ML依赖；CUDA路径须在4090验证 |
| Ledoit-Wolf/Cholesky algebra | NOT VERIFIED | 本机 Torch/SciPy/sklearn 缺失，测试 skipped |
| clean committed implementation | FAIL | 修改尚未提交；本次不自动 commit/push |
| 真实 COCO metadata/group isolation | NOT VERIFIED | 工作区没有真实 metadata，无法确认字段与可用 group 数 |
| 真实模型本地 fingerprint/环境 | NOT VERIFIED | 没有提供本机可运行模型/完整环境 |
| 真实 GPU smoke | NOT RUN | 本次明确不执行模型实验 |
| G0/G1 全4024 pairing | NOT RUN | 未生成/未提供可验证两池；不猜旧池有效 |
| fit assignment/whitening/calibration 实物 | NOT RUN | 尚无正式模型/阈值；机制测试通过不等于实物通过 |
| Diff 全 test 输入/输出 provenance | NOT RUN | 尚无正式攻击记录 |
| Q0–Q5 完整 Original-12/Rotation raw | NOT RUN | 没有完整真实结果，不能生成正式论文数值 |
| 无 failed/missing、同协议、完整 runtime | NOT RUN | 必须由真实 run_manifest 检查 |
| bootstrap/paper export 来源一致 | 机制测试通过；真实运行未验收 | 最终门禁要求六组、10000 bootstrap、显式非diagnostic export及合法产物哈希 |

**FORMAL EXPERIMENT GATE: FAIL**。
失败原因不是已知要改研究算法，而是上述关键数值、数据、模型和全流程实物证据尚未具备。只有4090上依次完成验收与正式流程、最终门禁全部通过，才能把新结果称为 paper-ready。
