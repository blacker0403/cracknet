# CrackNet 裂缝分割课程

本仓库提供 ResNet34–UNet 裂缝分割的运行代码。E0 使用基础网络和 BCE、Dice。E8 加入 MDCC、BGGF、边界头、soft-clDice 和 Edge Loss。配套《教程.md》讲解网络、训练过程和实验设计。

## 1. 获取代码与安装环境

以下命令使用 Windows PowerShell。先安装 Git 和 Conda，再从能够运行 `conda activate` 的 PowerShell 进入项目。

```powershell
git clone https://github.com/blacker0403/cracknet.git
Set-Location cracknet
conda create -n cracknet-study python=3.11 -y
conda activate cracknet-study
$env:PYTHONUTF8 = '1'
```

CUDA 训练选用以下安装命令。先确认 NVIDIA 驱动支持对应的 wheel。

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

使用 CPU 跟练时选用这一条。

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

安装项目依赖并检查解释器。CUDA 路线要求最后一项输出为 `True`。每次新开终端都重新激活环境，设置 `PYTHONUTF8` 并进入仓库根目录。

```powershell
python -m pip install -r requirements.txt
python -m pip check
python -c "import sys, torch, torchvision; print(sys.executable); print(torch.__version__, torchvision.__version__); print(torch.version.cuda, torch.cuda.is_available())"
```

版本与安装源见 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v260)。

## 2. 用合成数据跑一次 CPU 全流程

```powershell
python -m scripts.smoke_test --output results/tutorial/smoke_first
Get-Content -Encoding UTF8 results/tutorial/smoke_first/smoke_status.json
```

检查 `passed` 和 `synthetic_only` 都为 `true`。脚本生成带标签的小图，完成 E0 训练和 E8 停训、续训、冻结、测试与推理。在 `E8_seed42/` 看检查点和指标，在 `E8_predictions/` 看推理输出。再次执行时使用新目录，预留数 GB 磁盘空间存放检查点。合成数据的分数用于检查流程。

阅读网络时，可以用随机输入核对奇数尺寸。结构检查使用 CPU 并跳过预训练权重初始化。

```powershell
python tutorial_tools/inspect_model.py --config configs/tutorial/local_e8.yaml --height 65 --width 91
```

## 3. 准备真实数据和编码器权重

可使用课程整理的 CRACK500 图像、标签及三份清单，其划分为 train 1841、val 341、test 1124。自行下载时，作者 README 提供数据下载入口。保留所选数据版本的划分，按下方布局整理图像和标签，再按本节步骤生成清单并登记数据版本。[CRACK500 作者仓库](https://github.com/fyangneil/pavement-crack-detection)

```text
cracknet/
  data/CRACK500/
    train/images/     train/masks/
    val/images/       val/masks/
    test/images/      test/masks/
    manifests/train.jsonl
    manifests/val.jsonl
    manifests/test.jsonl
  weights/resnet34-b627a593.pth
```

JSONL 每行对应一对图像和标签。两个文件路径相对 `data/CRACK500`。掩码为与图像同尺寸的单通道 PNG，像素为 0/1 或 0/255。同一原图、视频或采集场景的相关样本使用统一的 `source_id`，按来源组划入同一个集合。

```json
{"id":"road001_0_0","image":"train/images/road001_0_0.jpg","mask":"train/masks/road001_0_0.png","source_id":"road001"}
```

从已划分的 JPG 和同名 PNG 扫描生成清单时，在 `configs/tutorial/local_e0.yaml` 的 `data` 下写入数据的 `version`，设置 `fixed_manifests: false` 并保留 `workers: 0`。默认文件名规则把 `road001_0_0` 识别成 `road001` 来源组。其他命名规则需要提供准确的 `data.source_id_regex`，也可以手工填写 manifest。E8 配置继承这个数据设置。

```powershell
python -m datasets.inspect_dataset --config configs/tutorial/local_e0.yaml
```

检查 `manifests/audit.json` 和 `manifests/samples/`，确认划分、配对和标签后将 `fixed_manifests` 恢复为 `true`。已有课程清单可直接进入预检。

预检检查本地权重文件。可从 torchvision 的权重地址下载一次。

```powershell
New-Item -ItemType Directory -Force weights | Out-Null
Invoke-WebRequest -UseBasicParsing -Uri 'https://download.pytorch.org/models/resnet34-b627a593.pth' -OutFile 'weights/resnet34-b627a593.pth'
```

## 4. 训练、续训与查看日志

演练使用 E8 的两轮预算，实际遍历准备好的数据。它继承 384×384 裁剪、batch 4、梯度累积 4、FP16、AdamW 和初始学习率 0.0001。Windows 示例使用 `workers: 0`。

```powershell
$cfg = 'configs/tutorial/local_e8_warmup.yaml'
$run = 'results/tutorial/local_E8_2epochs_seed42'
python -m scripts.preflight --config $cfg --paths-only
python -m scripts.preflight --config $cfg --require-cuda
```

第一条检查数据根目录、清单和文件路径。第二条生成数据审计与 `preflight.json`，确认 `ready_for_training` 为 `true` 后再训练。可先用一次合成批次探测显存，输出保存在新的 JSON 文件中。

```powershell
python tutorial_tools/probe_model.py --config $cfg --output results/tutorial/first_gpu_probe.json
```

跑完第 1 轮后退出，再从已经保存的 epoch 恢复。想连续完成两轮时，首次运行可以省略 `--stop-after-epoch 1`。

```powershell
python -m scripts.train --config $cfg --stop-after-epoch 1
Get-Content -Encoding UTF8 "$run/training_status.json"
python -m scripts.train --config "$run/config.yaml" --resume "$run/last.pth"
Get-Content -Encoding UTF8 "$run/training_status.json"
```

| 产物 | 看什么 |
|---|---|
| `config.yaml`、`environment.json` | 实际配置、设备、AMP 和有效 batch |
| `history.csv`、`timing.csv` | 每轮指标、学习率、耗时和显存 |
| `last.pth` | 还原最近保存的权重、优化器、调度器等状态 |
| `best_iou.pth`、`validation_best.json` | 固定阈值下验证 IoU 最佳的检查点和对应指标 |
| `training_status.json` | 第一次暂停后 `complete: false`，两轮结束后 `complete: true` |

在另一个已激活环境的终端打开 TensorBoard，浏览器访问 `http://127.0.0.1:6006`。

```powershell
python -m tensorboard.main --logdir results/tutorial --host 127.0.0.1 --port 6006
```

正式训练起点见 `configs/tutorial/local_e0.yaml` 和 `configs/tutorial/local_e8.yaml`，默认最多 150 轮、早停耐心值 30。变更模型、学习率或数据时，从新结果目录启动。续训使用本次实验保存的配置和 `last.pth`，数据路径、清单与内容应匹配。

从已有 E8 分割权重开始一次微调时，将 `configs/tutorial/local_e8_warmup.yaml` 在同目录另存为 `local_e8_finetune.yaml`，把 `output_dir` 改成 `../../results/tutorial/local_E8_finetune_seed42`，在 `train` 下加一行 `lr: 0.00005`。沿用相同的模型结构，传入可信的、由本工程保存的检查点。

```powershell
python -m scripts.train --config configs/tutorial/local_e8_finetune.yaml --init-checkpoint "$run/best_iou.pth"
```

新目录的 `initialization.json` 记录来源检查点。微调只加载已有模型参数，优化器、调度器和轮数重新初始化。

## 5. 冻结、测试与推理

训练完成后先冻结。冻结在 val 上按 micro Dice 选阈值，并记录 best 检查点、配置和数据的指纹。test 使用这份冻结记录。下列命令继续使用上一节的 `$run`。

```powershell
python -m scripts.evaluate --config "$run/config.yaml" --checkpoint "$run/best_iou.pth" --freeze
python -m scripts.evaluate --config "$run/config.yaml" --checkpoint "$run/best_iou.pth" --split test
Get-Content -Encoding UTF8 "$run/frozen.json"
Get-Content -Encoding UTF8 "$run/test_metrics.json"
```

单图推理以清单中第一张测试图为例。也可以将 `--input` 指向自己的图片或图片目录，`--output` 使用独立的空目录。

```powershell
$sample = Get-Content -Encoding UTF8 data/CRACK500/manifests/test.jsonl -TotalCount 1 | ConvertFrom-Json
$image = Join-Path 'data/CRACK500' $sample.image
python -m scripts.predict --config "$run/config.yaml" --checkpoint "$run/best_iou.pth" --input $image --output results/tutorial/prediction_first
```

同一 run 的 `config.yaml`、`best_iou.pth` 与 `frozen.json` 配套使用。`*.mask.png` 为二值裂缝图，`*.overlay.png` 显示叠加位置，`*.probability.npy` 保存像素分数，`prediction_summary.csv` 汇总图像尺寸、预测与耗时。带标签的质量指标查看 `test_metrics.json`。迁移资产用于推理时，可以在配置副本调整存储路径，模型和评价设置须符合冻结记录。

## 6. 设置并运行模块消融

课堂使用两轮 E0 基底，矩阵为每组覆盖模块开关和损失权重。E0 与 E1 对比 MDCC，E7 与 E8 对比额外的 Edge Loss。四组使用相同划分、预训练编码器、种子、训练预算与验证协议。

```powershell
$matrixArgs = @(
  '--matrix', 'configs/ablation.yaml',
  '--base-config', 'configs/tutorial/local_e0_warmup.yaml',
  '--experiments', 'E0,E1,E7,E8',
  '--output-root', 'results/tutorial/ablation_2epochs'
)
python -m scripts.run_ablation @matrixArgs --dry-run
```

检查 `E0_seed42`、`E1_seed42`、`E7_seed42` 和 `E8_seed42` 的 `planned_config.yaml`。核对开关后运行。同一命令也用于恢复未完成的组。

```powershell
python -m scripts.run_ablation @matrixArgs --resume
python -m scripts.summarize_results --root results/tutorial/ablation_2epochs --output results/tutorial/ablation_2epochs_summary --split val
```

`--dry-run` 只生成方案。正式运行生成训练产物、验证结果和开销记录。查看固定阈值下的指标时读取 `validation_best.json`。`val_metrics.json` 与汇总 CSV 使用各组在验证集选出的阈值，比较时把阈值一并记录。锁定待报告的模型和指标口径后，按第 5 节对相应组冻结并评价测试集。

只有 CPU 时，可以用第 2 节生成的 E0 配置作为共同基底，在同一 checkout 执行 E0 和 E1 两组。

```powershell
python -m scripts.run_ablation --matrix configs/ablation.yaml --base-config results/tutorial/smoke_first/E0_seed42/config.yaml --experiments 'E0,E1' --output-root results/tutorial/ablation_cpu --resume
```

这条路线在相同合成数据和一轮预算下演示对照设置。正式数据实验可用 `--seeds '42,123,3407'` 补充种子，记录每次运行的数据划分、实际轮数、分割与边界指标。

## 运行时常见卡点

| 现象 | 检查项 |
|---|---|
| Python 找不到 `scripts` 或配置文件 | 在含 `scripts/` 和 `configs/` 的仓库根目录运行 |
| CUDA 为 `False` | 核对当前解释器、wheel 与驱动，或先执行 CPU 合成路线 |
| `Dataset root missing` 或 `Missing pair` | 核对解压层级、清单路径、标签文件名及大小写 |
| `pretrained_weights_ready: false` | 将对应编码器权重放入 `weights/` 后重新预检 |
| 训练或预测目录已有产物 | 续训保留原实验目录，新实验或新预测使用新目录 |
| `Finish the configured training budget before freezing` | 查看训练状态并完成预算或早停流程 |

第三方实现与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
