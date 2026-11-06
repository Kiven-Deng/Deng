# 排名中心恢复：广义 Mallows 数据与监督式 Transformer

本项目研究：给定少量带噪声的完整榜单，能否通过监督学习恢复产生这些榜单的潜在中心排名。目标应用是大学排名聚合；当前实验使用合成数据，尚未完成真实大学排名验证。

本文记录截至 **2026-11-06** 的新增文件、改进原因、实现位置、作用、运行方式和实际结果。推荐从 `ranking_recovery_v3` 开始运行。

## 1. 研究目标

给定中心排名，利用广义 Mallows 生成多份观察榜单，用观察榜单作为输入、中心排名作为监督标签。模型最终输出完整排列，以预测与真实中心之间的距离评价恢复能力。

~~~text
独立中心排名 → train / validation / test 划分 → 分别生成带噪观察
带噪观察 → 名次特征 → 共享映射与集合聚合 → Transformer → 指针解码
预测排名与真实中心比较 → Kendall 距离、RBO 距离、完整恢复率
~~~

- 目标是中心恢复，评价不以最大化 Kemeny 求解能力为目标。
- 当前支持同一对象集合上的完整、无并列排名，不支持缺失对象、部分榜单或并列名次。
- 观察榜单视为可交换输入，尚未加入排名机构身份或来源可信度权重。
- 训练损失仍是 teacher forcing 下的交叉熵，没有改成 Kendall 与 RBO 的加权损失。

## 2. 版本关系与原始文件修改情况

| 阶段 | 文件/目录 | 定位 |
|---|---|---|
| 原始实现 | 根目录 `architecture.py`、`mallows.py`、`train.py` 等 | 保留旧模型及旧权重对照 |
| 独立生成器 | `generalized_mallows_generator_v1.py` | 新增分阶段 θ 向量和正确归一化采样 |
| 初始比较脚本 | `compare_rankings_v1.py` | 新增旧模型与传统方法比较 |
| v2（历史版本，源码已移除） | 原 `ranking_recovery_v2/` | 数据、模型、训练、评估与自检实现已完整保留在 v3 中 |
| v3 | `ranking_recovery_v3/` | 独立复制 v2，修复直接运行脚本的相对导入错误 |
| 实验衔接 | `finish_v3_experiment_v1.py` | 训练结束后运行评价、旧权重对照和报告汇总 |
| 本说明 | `README.md` | 新增根目录说明，不参与训练或推理 |

**v2/v3 的模型改进采用新增版本方式，没有在原始源文件上就地修补。**初次编写本文时，原有源文件相对于 Git 提交 `bad09f4` 没有未提交改动；两个独立 v1 脚本已包含在该提交中。这不代表追溯认定更早所有历史改动的作者或时间。

2026-11-06 按用户要求精简上传内容：确认 v3 和辅助脚本不导入 v2 后，删除了根目录 `ranking_recovery_v2/` 源码目录（9 个 Python 文件及包内 README），并更新本文。v3、原始代码、数据、权重和实验结果均保留；`data/ranking_recovery_v2/` 等数据目录不在此次删除范围内。

下表标明原始文件及对应的新实现。旧文件中的问题仍保留在旧版本，新实现并不意味着旧文件已修复。

| 原始文件 | 原相关位置/职责 | 新实现位置 | 本轮就地修改 |
|---|---|---|---|
| [mallows.py](mallows.py) | `_sample_relative_permutations()`，约第 71–101 行：普通几何采样后压到上界 | v1 生成器；v3 `data.py::sample_orders()` | 否 |
| [ranking_data_generator.py](ranking_data_generator.py) | `generate_noisy_rankings()` 调用旧采样器 | v3 `data.py::build_bundle()` | 否 |
| [architecture.py](architecture.py) | `IdentityEmbedding`、`MaskAwareBatchNorm1d`、编码器和解码器 | v3 `model.py` | 否 |
| [train.py](train.py) | 约第 324 行取输入最后一维为嵌入维度；旧训练流程 | v3 `train.py` | 否 |
| [train_config.py](train_config.py) | 旧参数、归一化与位置机制配置 | v3 命令行参数及 `ModelConfig` | 否 |
| [ranking_metrics.py](ranking_metrics.py) | 原 Kendall/RBO 定义 | v3 `metrics.py` 保留评价口径 | 否 |
| [validation.py](validation.py) | 旧权重加载及推理 | v3 `evaluate.py`；衔接脚本复用旧加载函数进行对照 | 否 |
| [.gitignore](.gitignore) | 忽略 Markdown、JSON、`outputs/` 等 | 第 10 节提供显式上传方式 | 否 |

行号是当前版本的定位提示，长期请优先按函数名定位。后续改动继续采用新版本目录或文件，保留现有代码、数据和权重。

## 3. 新增文件及作用

### 3.1 根目录独立脚本

| 文件 | 为什么新增 | 作用 |
|---|---|---|
| [generalized_mallows_generator_v1.py](generalized_mallows_generator_v1.py) | 需要分阶段加噪并消除截尾概率错误 | `build_theta_vector()` 构造 N−1 维 θ；有限支持归一化采样；保存独立数据；支持 `--self-test` |
| [compare_rankings_v1.py](compare_rankings_v1.py) | 原实验缺少统一传统方法比较 | 默认使用旧 checkpoint 的验证划分；比较相同输入；保存汇总和逐样本结果；不训练模型 |
| [finish_v3_experiment_v1.py](finish_v3_experiment_v1.py) | 自动衔接训练完成后的测试与旧权重对照 | 检查 `best.json`、源码指纹、锁与状态；运行评价；生成对照和报告 |

`finish_v3_experiment_v1.py` **不是定时器，也不是模型训练依赖**。调用时没有 `best.json` 就打印等待信息并退出；定时调用由外部监控完成。本次实验已完成，无需再次运行。状态为 `complete` 时返回已有结果；状态为 `error` 时要求检查失败信息，不静默无限重试。

该脚本含实验特定约束：运行根目录下应只有一个训练目录；旧权重固定在 `checkpoints/mallows_transformer.pt`；旧模型导入使用 `Deng.train` 和 `Deng.validation`，依赖仓库目录名为 `Deng`；复用已有评价时检查 500 个测试中心。因此它不是适用于任意目录和任意数据规模的通用工具。

### 3.2 保留的 v3 包内文件

以下完整实现保留在 v3 中，运行时不需要 v2 源码目录。

| 文件 | 主要位置 | 用途 |
|---|---|---|
| [data.py](ranking_recovery_v3/data.py) | `theta_profile()`、`sample_orders()`、`build_bundle()`、`validate_bundle()`、`diagnostics()` | 噪声采样、独立中心划分、不同 m 的嵌套观察与诊断 |
| [model.py](ranking_recovery_v3/model.py) | `ModelConfig`、`features()`、`encode()`、`prefix_logits()` | 固定隐藏维度、保留名次、集合聚合、LayerNorm、指针解码 |
| [train.py](ranking_recovery_v3/train.py) | `train()`、`condition_metrics()` | 混合条件训练、自由解码验证、学习率调度、早停与最佳权重 |
| [evaluate.py](ranking_recovery_v3/evaluate.py) | `evaluate()` | 同输入比较七种方法；保存结果与配对置信区间 |
| [baselines.py](ranking_recovery_v3/baselines.py) | `baseline_predictions()` 及算法函数 | 独立保存 v1 基线实现，避免导入旧训练模块 |
| [metrics.py](ranking_recovery_v3/metrics.py) | `distances()` | 归一化 Kendall 和有限深度归一化 RBO 距离 |
| [tests.py](ranking_recovery_v3/tests.py) | `run_checks()` | 采样、结构性质、因果性、梯度、权重读取和微型拟合 |
| [common.py](ranking_recovery_v3/common.py) | `new_directory()`、`sha256()`、`source_hashes()`、`write_json()` | 唯一目录、文件指纹、来源记录和排他式新文件写入 |
| [__init__.py](ranking_recovery_v3/__init__.py) | 包声明 | 支持包导入与 `python -m` |
| [README_zh.md](ranking_recovery_v3/README_zh.md) | 包内说明 | 版本运行方式与边界；根目录本文补充版本关系和实际结果 |

## 4. 为什么改、改在哪里、有什么作用

### 4.1 噪声生成：正确分布与阶段控制

旧 `mallows.py` 虽有“exact”注释，实际先采无界几何变量，再用 `torch.minimum` 压到上界。新版 `data.py::sample_orders()` 在每个阶段直接采用归一化分布：

~~~text
j = 0, ..., N-2
k = 0, ..., N-j-1
P(V_j=k) = exp(-theta_j*k) / sum_r exp(-theta_j*r)
~~~

代码使用 float64 的 `softmax` 构造概率，`torch.multinomial` 采样，按中心次序在剩余对象中选择第 k+1 个；最后一个阶段只剩一个对象。

| 条件 | N=50 时 θ 设置 | 目的 |
|---|---|---|
| `uniform_low` | 49 个非强制阶段均为 2 | 较低噪声恢复 |
| `uniform_medium` | 所有阶段均为 1 | 较高噪声恢复 |
| `top_stable` | 阶段 1–10：3；11–30：1；31–49：0.35 | 检查前部稳定、后部波动 |
| `no_noise` | 直接复制中心 | 自检及微型拟合，不在本次正式三噪声测试中 |

θ 控制的是选择阶段偏离倾向，不能解释为某个对象最终位移的硬上限；对象可能跨分段边界。参数表达研究假设，尚未通过真实大学排名校准，实际位移见 `diagnostics.json`。

### 4.2 数据隔离：先分中心再生成噪声

位置：`data.py::build_bundle()`、`validate_bundle()`、`main()`。

- 先生成不重复中心，再划分 4000/500/500 个训练、验证和测试中心，然后分别加噪。
- 默认排除旧格式数据中的中心，避免仅更换噪声却重复原始榜单。默认发现逻辑不会自动排除所有历史 v2/v3 bundle。
- 每个中心/噪声条件先生成 10 份观察，m=2/3/5/10 取同一观察组的前缀，减少观察数量比较中的额外重采样干扰。
- 独立 v1 生成器的默认中心种子与旧数据相同，不能只因噪声不同就视为独立测试集。

作用是检验对未见中心的恢复能力，而不是记忆已见原型。

### 4.3 输入表示：榜单数与模型容量分开

位置：`model.py::ModelConfig`、`RecoveryTransformer.features()`、`encode()`。

旧训练流程将榜单数 m 作为嵌入维度。新版隐藏维度固定为 64、4 个注意力头、编码器与解码器各 2 层，不会随输入 m 改变隐藏维度。

每个观察名次转换为归一化名次、平方、倒数、对数四个特征；每个对象额外保留平均名次、标准差、最小/最大名次、平均倒数名次、`log(1+m)` 六个统计量。

作用是同时保留“通常排在哪里”和“各榜单意见是否一致”。没有直接在原始名次向量上做 LayerNorm，避免抹去类似 `[1,2,1,1]` 与 `[21,22,21,21]` 的绝对名次差异。可学习的 `rank_skip` 将统计特征接入最终得分，不是固定 Borda 回退规则。

### 4.4 集合聚合：消除榜单列顺序影响

位置：`model.py::encode()`。

对每份观察使用共享映射，再沿榜单维度求均值，使交换输入榜单顺序不改变表示。模型无对象 ID 嵌入，具有对象重编号等变结构；最终贪心选择若出现精确数值平局，`argmax` 的索引规则可能影响输出。

取舍是聚合可能损失同一来源中不同对象之间的联合关系，当前没有使用机构身份。这是后续可研究的假设，尚未证明是落后于 MC4 的原因。

### 4.5 归一化与解码：减少统计错配并输出合法排列

位置：`model.py::EncoderBlock`、`DecoderBlock`、`prefix_logits()`、`predict()`。

学习到的隐藏表示使用 LayerNorm，去掉旧 BatchNorm 配置对批次及运行统计的依赖。此前旧权重诊断显示统计使用方式会明显影响推理，因此新模型从头训练，没有直接转换旧 BatchNorm 权重。

训练与推理共享前缀得分函数。训练使用真实前缀，推理使用已预测前缀；因果掩码阻止读取未来答案，已选对象掩码避免重复输出。两种前缀来源仍不同，因此并未消除自回归误差累积。

### 4.6 位置机制：可以独立做消融

位置：`model.py::RankAttention`，命令行 `--rarope` 与 `--anlibi`。

保留 rank-log RoPE 和 ANLiBi 风格偏置的独立开关。这些是代码明确写出的实现变体，不宣称完整复现论文实现。**本次正式训练两个开关均关闭**，不能把改善归因于它们。

### 4.7 训练选择：以完整榜单恢复效果选权重

位置：`train.py::train()`、`condition_metrics()`。

- 每轮覆盖训练中心与噪声条件组合，按批次抽取 m。
- 使用交叉熵训练，每轮在 `eval()` 下自由生成完整验证榜单。
- 根据验证各条件宏平均 Kendall 选权重，完全相同时用 RBO 打破平局。
- AdamW 初始学习率 1e-4，梯度裁剪 1；验证停滞则学习率减半，连续 10 轮未改善则早停。
- 本次 `weight_decay=0`，不能将改善解释为增加了权重衰减。
- 学习率调度、早停和权重选择均不使用测试集。

### 4.8 v3 启动修复

位置：v3 `data.py`、`train.py`、`evaluate.py`、`tests.py` 的相对导入之前。

当 `__package__` 为空时，根据文件位置把包父目录加入 `sys.path`，并设置包名，解决 `ImportError: attempted relative import with no known parent package`。VS Code 直接运行文件、绝对路径运行及 `python -m` 均可使用。

**v3 没有在 v2 基础上再次改变模型架构。**其他 Python 文件仅增加版本注释。为保持兼容，数据和权重格式标识仍为 `ranking_recovery_v2`，默认输出路径也可能含 v2；这些是格式标签和输出目录名，不是对已删除 v2 源码的依赖。下面命令显式指定新输出目录。旧错误提示若建议运行 `ranking_recovery_v2.data`，实际请使用 `ranking_recovery_v3.data`；此次未改动 v3 源码，保留与训练记录一致的源码指纹。

## 5. 基线与指标

所有方法使用同一中心对应的相同观察。真实中心用于评价，不参与传统方法候选选择。

| 方法 | 实现 |
|---|---|
| Transformer | 监督式排名恢复模型 |
| Borda | 名次和升序 |
| `Markov_MC4_damped` | 严格多数偏好转移矩阵，阻尼 0.85，稳态容差 1e-12 |
| `KwikSort_32` | 随机多数枢轴排序，32 次重启，按输入分歧选结果 |
| Copeland | 成对多数胜场减负场 |
| Medoid | 选与所有输入总分歧最小的观察榜单 |
| `Kemeny_insertion_local` | 多起点单对象插入局部搜索 |

这些是明确的算法变体，没有完整复现论文的强化学习模型或 DECoR。局部搜索不等于精确 Kemeny 解或广义 Mallows 精确 MLE；当前未单独实现完整 θ 向量估计与精确中心 MLE 基线。

- Kendall 距离：不一致对象对数除以 `N(N-1)/2`，越小越好，不是相关系数。
- RBO 距离：有限深度归一化定义，`p=0.9`，越小越好。若 `A_d` 为两个前 d 集合的交集比例，则距离为 `1-(1-p)*Σ[p^(d-1)*A_d]/(1-p^N)`。
- 同时保存完整恢复率、逐样本预测与置信区间。
- 宏平均配对 bootstrap 先在每个中心内平均各条件，再对中心重采样 2000 次。同一中心的重复条件不作为独立中心。区间只反映中心抽样变化，不包括重新训练的种子差异。

## 6. 运行顺序

在仓库根目录运行以下 PowerShell 命令。v3 核心使用 Python、PyTorch、NumPy；旧模型比较还涉及旧模块的 tqdm 依赖。使用已安装依赖的环境。原实验解释器为 `D:/Anaconda/envs/Pytorch/python.exe`，其他机器请替换。

~~~powershell
$py = 'python'  # 已激活的环境，也可填解释器完整路径
$run = Join-Path (Get-Location) ('outputs/v3_run_' + [guid]::NewGuid().ToString('N'))

# 第一步：自检与无噪声微型拟合。
& $py -B -m ranking_recovery_v3.tests
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }

# 第二步：生成数据。
& $py -B -m ranking_recovery_v3.data --output "$run/data"
if ($LASTEXITCODE -ne 0) { throw 'Data generation failed' }
$datasets = @(Get-ChildItem -Path "$run/data/dataset_*/dataset.pt")
if ($datasets.Count -ne 1) { throw 'Expected one dataset' }
$dataset = $datasets[0].FullName

# 第三步：训练，等待正常结束。
& $py -B -m ranking_recovery_v3.train --dataset $dataset --output "$run/checkpoints"
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
$bestFiles = @(Get-ChildItem -Path "$run/checkpoints/train_*/best.json")
if ($bestFiles.Count -ne 1) { throw 'Expected one completed training run' }
$checkpoint = (Get-Content -Raw $bestFiles[0].FullName | ConvertFrom-Json).checkpoint

# 第四步：全部测试中心及传统方法比较。
& $py -B -m ranking_recovery_v3.evaluate --checkpoint $checkpoint --output "$run/comparison"
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
~~~

需要旧权重对照与自动报告时，在满足第 3.1 节约束、训练结束后，可以用以下命令代替普通第四步；若普通第四步已完成，会尝试复用匹配结果：

~~~powershell
& $py -B finish_v3_experiment_v1.py --run-root $run
~~~

辅助脚本不是轮询服务；没有完成标记时会退出。不要与正在训练的 GPU 工作同时运行评价。需要改变实验逻辑时新增脚本版本。

移动已有权重到另一台机器时注意：checkpoint 元数据中的数据路径通常是绝对路径。普通评价可用 `--dataset` 指定迁移后的同一文件，并核对 SHA-256；辅助脚本依赖更多原绝对路径，不能保证复制仓库后直接复用。

## 7. 生成的数据、权重及报告

| 文件/位置 | 作用 |
|---|---|
| `data/generalized_mallows/generalized_mallows_v1_*.pt` | 独立 v1 生成器输出 |
| `<run>/data/dataset_*/dataset.pt` | v2/v3 中心划分、噪声观察及配置 |
| 数据目录 `manifest.json` | 数据参数、种子、排除数据来源 |
| 数据目录 `diagnostics.json` | 实际噪声距离和位置位移 |
| `<run>/checkpoints/train_*/manifest.json` | 模型/训练参数、数据和源码指纹 |
| 训练目录 `history.jsonl` | 每轮损失、验证指标、学习率及耗时 |
| 训练目录 `best_epoch_*.pt` | 验证改善时保存的模型与训练状态 |
| 训练目录 `best.json` | 训练结束时记录最终最佳权重和完成轮数 |
| 比较目录 `summary.csv` | 各方法、噪声、m 条件的指标汇总 |
| 比较目录 `per_sample.jsonl` | 逐中心真值、预测和距离 |
| 比较目录 `results.json` | 汇总、配置、指纹、索引和配对区间 |
| `legacy_same_test.json` | 相同新测试数据上的旧模型 m=10 对照 |
| `legacy_same_test_per_sample.jsonl` | 旧模型对照逐样本明细 |
| `analysis_report.md` | 衔接脚本生成的英文报告 |
| `结果分析_中文.md` | 实验后补充的中文解读，不是普通评价脚本自动输出 |
| `verified_analysis.json` | 本次额外核对的样本数量、汇总和补充统计 |
| `<run>/auto_completion_v1.json` | 辅助脚本状态及结果路径 |
| `<run>/auto_completion_v1.lock` | 运行期间的锁，正常退出后删除 |

本次正式运行根目录：`outputs/v3_run_ad482c219a0944858edc0efdb7b0f574`。

- 数据：`data/dataset_20261106_121433_096724_9867d22d/dataset.pt`。
- 训练：`checkpoints/train_20261106_121510_562402_633ffbf8`。
- 比较：`comparison_auto/comparison_20261106_123041_882812_a5cc9077`。

以上三项相对于本次运行根目录。只有显式上传后才会出现在 GitHub。较早 `outputs/comparison/v1_20261031_111546_974792_e7925190` 是旧数据验证划分上的比较；`data/ranking_recovery_v2/` 等目录可能包含开发期的小规模流程测试，不能与正式数据混用。

## 8. 验证与正式实验结果

### 8.1 实现检查

已通过小 N 采样规律、中心不重叠、榜单列顺序不变性、对象重编号等变性、批次独立性、train/eval 一致性、因果解码、全部位置开关组合、梯度和权重读写检查。

无噪声微型拟合：24 个 N=6 中心在第 25 步完整恢复率达到 1.0，交叉熵约 0.28841；第 50 步为 0.01249，全部自由解码正确。它证明基本实现可训练，不证明对噪声数据和真实大学排名必然有效。

### 8.2 正式配置与核对

| 项目 | 配置 |
|---|---|
| 对象数与中心划分 | N=50；4000/500/500 |
| 条件 | 3 种噪声 × m=2/3/5/10，共 12 条件 |
| 模型 | hidden=64、heads=4、编码器/解码器各 2 层 |
| 位置机制 | 两项均关闭 |
| 训练 | 种子 42，单次训练；第 24 轮早停 |
| 最佳权重 | `best_epoch_014.pt` |
| 最佳验证宏平均 | Kendall=0.010494966，RBO=0.025528816 |
| 输出核对 | 84 项汇总、42,000 条新比较记录、1,500 条旧模型记录 |

已核对新测试中心与旧数据无重叠、新旧对照中心一致、逐样本指标均值与汇总一致。

### 8.3 12 个条件的宏平均

两种距离都越小越好。

| 方法 | Kendall | RBO |
|---|---:|---:|
| Transformer v3 | 0.010651 | 0.025487 |
| Borda | 0.011201 | 0.027251 |
| MC4（阻尼版） | 0.010307 | 0.024395 |
| KwikSort（32 次） | 0.011178 | 0.026379 |
| Copeland | 0.010303 | 0.024455 |
| Medoid | 0.018843 | 0.048798 |
| Kemeny 插入局部搜索 | 0.011134 | 0.026342 |

相对 Borda，宏平均 Kendall 降低约 4.91%，RBO 降低约 6.47%；11/12 个条件两种距离更好，`uniform_low,m=2` 略差。相对 MC4，仅 1/12 个条件的 Kendall 点估计更好，RBO 无更优条件；宏平均也落后于 Copeland。相对 KwikSort，Kendall 在 3/12 个条件更好、RBO 在 4/12 个条件更好；宏平均优势不代表所有条件均占优。

### 8.4 全部条件下的模型结果

| 噪声 | m | Kendall | RBO |
|---|---:|---:|---:|
| uniform_low | 2 | 0.005071 | 0.032414 |
| uniform_low | 3 | 0.001763 | 0.011433 |
| uniform_low | 5 | 0.000767 | 0.005597 |
| uniform_low | 10 | 0.000106 | 0.001058 |
| uniform_medium | 2 | 0.017365 | 0.092141 |
| uniform_medium | 3 | 0.011558 | 0.062635 |
| uniform_medium | 5 | 0.007533 | 0.045769 |
| uniform_medium | 10 | 0.003394 | 0.021153 |
| top_stable | 2 | 0.030238 | 0.019596 |
| top_stable | 3 | 0.023159 | 0.007528 |
| top_stable | 5 | 0.016645 | 0.004450 |
| top_stable | 10 | 0.010211 | 0.002072 |

三个噪声条件下，更多观察均改善模型的两种距离。`top_stable` 的整体 Kendall 较大而 RBO 较小，与前部稳定、尾部波动及 RBO 更重视前部的评价方式一致。

### 8.5 宏平均配对 95% 区间

差值为“Transformer − 基线”，负值表示模型更好。

| 基线 | Kendall 差值区间 | RBO 差值区间 |
|---|---|---|
| Borda | [-0.000595789, -0.000504486] | [-0.002234316, -0.001282263] |
| MC4 | [0.000301357, 0.000385173] | [0.000745973, 0.001453907] |
| KwikSort | [-0.000602595, -0.000456449] | [-0.001492968, -0.000259557] |
| Copeland | [0.000307884, 0.000385854] | [0.000689337, 0.001407157] |
| Medoid | [-0.008324639, -0.008061449] | [-0.024705395, -0.021979287] |
| Kemeny 局部搜索 | [-0.000556214, -0.000411973] | [-0.001454087, -0.000232596] |

补充分析：`top_stable,m=2` 对 MC4 的 Kendall 差约 -0.000158367，区间 [-0.000334735, 0.000013061] 包含 0，因此点估计领先不能视为已确认优势。

### 8.6 相同新测试输入上的旧/新权重对照

旧模型要求 m=10，因此仅在 m=10 对齐，每种噪声为 500 个中心。

| 噪声 | 旧 Kendall | 新 Kendall | 旧 RBO | 新 RBO |
|---|---:|---:|---:|---:|
| uniform_low | 0.474709 | 0.000106 | 0.781131 | 0.001058 |
| uniform_medium | 0.475056 | 0.003394 | 0.784608 | 0.021153 |
| top_stable | 0.473603 | 0.010211 | 0.788126 | 0.002072 |

这是整套系统的改善：训练数据、表示、归一化与选权重方式同时改变，不能把差异全部归因于某个单独机制。

## 9. 结论边界与后续工作

当前结果支持模型可以学习合成噪声下的中心恢复，本次宏平均优于 Borda、KwikSort、Medoid 和插入局部搜索，但落后于 MC4 与 Copeland；不能声称全面超过现有方法。

后续可在固定数据、种子和预算下做逐项消融、多训练种子检验、来源关系建模及真实大学排名实验。当前测试结果已经被分析；若据此选择新架构或参数，应把它作为开发参考，并另设独立最终测试集，不通过挑选有利条件证明模型更强。

真实大学排名机构可能采用不同目标，不一定共享一个可被无偏恢复的唯一中心。因此合成中心恢复结果不能直接解释为已识别现实中的客观共识榜单。

## 10. 上传 GitHub

现有 `.gitignore` 采用“代码、权重、数据”白名单，显式忽略 `*.md` 和 `outputs/`，JSON 也不在允许名单中。普通 `git add .` 不会自动添加本说明与报告。文档整理与 v2 源码清理没有修改忽略规则，也没有执行提交或推送。

检查内容后，可显式添加说明和源码：

~~~powershell
git add -f README.md ranking_recovery_v3/README_zh.md
git add ranking_recovery_v3 finish_v3_experiment_v1.py
git status --short
git diff --cached --stat
~~~

两个根目录 v1 脚本已在编写本文时被跟踪。如需共享实验结果，逐个选择 `summary.csv`、`results.json`、报告和必要的配置文件，用 `git add -f <具体路径>` 添加，不必加入所有中间权重与开发试验数据。上传前检查配置记录中的本机绝对路径是否适合公开。

本文源码链接使用仓库相对路径，适合 GitHub 阅读；记录了本地结果路径并不代表它们已经上传。
