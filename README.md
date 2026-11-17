# CrackNet 裂缝分割课程

本仓库提供 ResNet34–UNet 裂缝分割的运行代码。E0 使用基础网络和 BCE、Dice。E8 加入 MDCC、BGGF、边界头、soft-clDice 和 Edge Loss。配套《教程.md》讲解网络、训练过程和实验设计。

## 1. 获取代码与安装环境

先安装 Git 和 Anaconda。从 Windows 开始菜单打开 **Anaconda Prompt**，进入准备存放项目的目录，再逐行执行下面的命令。

```bat
git clone https://github.com/blacker0403/cracknet.git
cd cracknet
conda create -n cracknet-study python=3.11 -y
conda activate cracknet-study
chcp 65001 >nul
set "PYTHONUTF8=1"
```

CUDA 训练选用以下安装命令。先确认 NVIDIA 驱动支持对应的 wheel。

```bat
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

使用 CPU 跟练时选用这一条。

```bat
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

安装项目依赖并检查解释器。CUDA 路线要求最后一项输出为 `True`。每次新开窗口都重新激活环境，按上面的写法设置 UTF-8，并用 `cd /d` 加仓库路径进入根目录。

```bat
python -m pip install -r requirements.txt
python -m pip check
python -c "import sys, torch, torchvision; print(sys.executable); print(torch.__version__, torchvision.__version__); print(torch.version.cuda, torch.cuda.is_available())"
```

版本与安装源见 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v260)。

## 2. 用合成数据跑一次 CPU 全流程

```bat
python -m scripts.smoke_test --output "results\tutorial\smoke_first"
type "results\tutorial\smoke_first\smoke_status.json"
```

检查 `passed` 和 `synthetic_only` 都为 `true`。脚本生成带标签的小图，完成 E0 训练和 E8 停训、续训、冻结、测试与推理。在 `E8_seed42/` 看检查点和指标，在 `E8_predictions/` 看推理输出。再次执行时使用新目录，预留数 GB 磁盘空间存放检查点。合成数据的分数用于检查流程。

阅读网络时，可以用随机输入核对奇数尺寸。结构检查使用 CPU 并跳过预训练权重初始化。

```bat
python tutorial_tools/inspect_model.py --config configs/tutorial/local_e8.yaml --height 65 --width 91
```

## 3. 准备真实数据和编码器权重

课程整理的 CRACK500 包含 train 1841、val 341、test 1124 对图像和标签。自行下载时，按下方目录整理。组建自己的数据集时，把同一原图的裁块或同一路段的连续画面划入同一集合。[CRACK500 作者仓库](https://github.com/fyangneil/pavement-crack-detection)

```text
cracknet/
  data/CRACK500/
    train/images/     train/masks/
    val/images/       val/masks/
    test/images/      test/masks/
  weights/resnet34-b627a593.pth
```

将 `road001.jpg` 放在对应集合的 `images/`，将 `road001.png` 放在同组的 `masks/`。掩码采用同尺寸的单通道 PNG，像素为 0/1 或 0/255。三个集合放好后即可运行第 4 节的训练。程序按目录取图，读取到缺失、损坏或尺寸不一致的样本时报告文件路径。

自己的数据也可放在 `data/MyCracks`，在 `configs/tutorial/my_cracks_e0.yaml` 保存以下配置。训练时将 `--config` 指向该文件，消融时将 `--base-config` 指向它。默认扫描 JPG 原图。如原图是 PNG，在 `data` 下设置 `image_glob: '*.png'`。

```yaml
extends: local_e0.yaml
experiment: my_cracks_e0
output_dir: ../../results/tutorial/my_cracks_e0_seed42
data:
  root: ../../data/MyCracks
  version: my_cracks_v1
```

默认使用 ImageNet 编码器权重。可提前下载到本地，训练会读取该目录。

```bat
if not exist "weights" mkdir "weights"
python -c "from urllib.request import urlretrieve; urlretrieve('https://download.pytorch.org/models/resnet34-b627a593.pth', 'weights/resnet34-b627a593.pth')"
```

## 4. 训练、续训与查看日志

数据放好后，在已激活环境的 Anaconda Prompt 进入仓库根目录，直接启动 E0 基线。

```bat
python -m scripts.train --config "configs\tutorial\local_e0.yaml"
```

两轮 E8 演练使用 384×384 裁剪、batch 4、梯度累积 4、FP16、AdamW 和初始学习率 0.0001，Windows 示例使用 `workers: 0`。跑完第 1 轮后退出，再从已保存的 epoch 恢复。要连续完成两轮，首次运行可省略 `--stop-after-epoch 1`。

```bat
set "CTN_CFG=configs\tutorial\local_e8_warmup.yaml"
set "CTN_RUN=results\tutorial\local_E8_2epochs_seed42"
python -m scripts.train --config "%CTN_CFG%" --stop-after-epoch 1
type "%CTN_RUN%\training_status.json"
python -m scripts.train --config "%CTN_RUN%\config.yaml" --resume "%CTN_RUN%\last.pth"
type "%CTN_RUN%\training_status.json"
```

| 产物 | 看什么 |
|---|---|
| `config.yaml`、`environment.json` | 实际配置、设备、AMP 和有效 batch |
| `history.csv`、`timing.csv` | 每轮指标、学习率、耗时和显存 |
| `last.pth` | 还原最近保存的权重、优化器、调度器等状态 |
| `best_iou.pth`、`validation_best.json` | 固定阈值下验证 IoU 最佳的检查点和对应指标 |
| `training_status.json` | 第一次暂停后 `complete: false`，两轮结束后 `complete: true` |

在另一个已激活环境的终端打开 TensorBoard，浏览器访问 `http://127.0.0.1:6006`。

```bat
python -m tensorboard.main --logdir results/tutorial --host 127.0.0.1 --port 6006
```

正式训练起点见 `configs/tutorial/local_e0.yaml` 和 `configs/tutorial/local_e8.yaml`，默认最多 150 轮、早停耐心值 30。变更模型、学习率或数据时，从新结果目录启动。续训使用本次实验保存的配置和 `last.pth`，沿用同一组数据文件。目录模式使用文件路径、大小和修改时间识别变化。历史检查点继续使用其保存的配置和对应的原始数据协议。

从已有 E8 分割权重开始一次微调时，将 `configs/tutorial/local_e8_warmup.yaml` 在同目录另存为 `local_e8_finetune.yaml`，把 `output_dir` 改成 `../../results/tutorial/local_E8_finetune_seed42`，在 `train` 下加一行 `lr: 0.00005`。沿用相同的模型结构，传入可信的、由本工程保存的检查点。

```bat
python -m scripts.train --config "configs\tutorial\local_e8_finetune.yaml" --init-checkpoint "%CTN_RUN%\best_iou.pth"
```

新目录的 `initialization.json` 记录来源检查点。微调只加载已有模型参数，优化器、调度器和轮数重新初始化。

## 5. 冻结、测试与推理

训练完成后先冻结。冻结在 val 上按 micro Dice 选阈值，并记录 best 检查点、配置和数据的指纹。test 使用这份冻结记录。下列命令在同一个 Anaconda Prompt 窗口继续使用上一节设置的 `%CTN_RUN%`。

```bat
python -m scripts.evaluate --config "%CTN_RUN%\config.yaml" --checkpoint "%CTN_RUN%\best_iou.pth" --freeze
python -m scripts.evaluate --config "%CTN_RUN%\config.yaml" --checkpoint "%CTN_RUN%\best_iou.pth" --split test
type "%CTN_RUN%\frozen.json"
type "%CTN_RUN%\test_metrics.json"
```

单图推理直接指定图像。下面以课程测试集的一张图为例，可把 `CTN_IMAGE` 的值换成自己的图片路径或图片目录。`--output` 使用独立的空目录。

```bat
set "CTN_IMAGE=data\CRACK500\test\images\20160222_080933_361_1.jpg"
python -m scripts.predict --config "%CTN_RUN%\config.yaml" --checkpoint "%CTN_RUN%\best_iou.pth" --input "%CTN_IMAGE%" --output "results\tutorial\prediction_first"
```

同一 run 的 `config.yaml`、`best_iou.pth` 与 `frozen.json` 配套使用。`*.mask.png` 为二值裂缝图，`*.overlay.png` 显示叠加位置，`*.probability.npy` 保存像素分数，`prediction_summary.csv` 汇总图像尺寸、预测与耗时。带标签的质量指标查看 `test_metrics.json`。迁移资产用于推理时，可以在配置副本调整存储路径，模型和评价设置须符合冻结记录。

## 6. 设置并运行模块消融

### E0–E10 消融矩阵

所有组以 E0 的 ResNet34–UNet 为基底，BCE 和 Dice 权重均为 1。各组使用相同的划分、预训练编码器、种子、训练预算与验证协议。[矩阵配置](configs/ablation.yaml)

模块列中 `√` 表示启用，`-` 表示未启用。

| 实验 | MDCC | BGGF | BoundaryHead | clDice 权重 | Edge Loss 权重 |
|---|---|---|---|---:|---:|
| E0 | - | - | - | 0 | 0 |
| E1 | √ | - | - | 0 | 0 |
| E2 | - | √ | √ | 0 | 0 |
| E3 | - | - | - | 0.2 | 0 |
| E4 | √ | √ | √ | 0 | 0 |
| E5 | √ | - | - | 0.2 | 0 |
| E6 | - | √ | √ | 0.2 | 0 |
| E7 | √ | √ | √ | 0.2 | 0 |
| E8 | √ | √ | √ | 0.2 | 0.1 |
| E9 | - | - | √ | 0 | 0.1 |
| E10 | - | √ | √ | 0 | 0.1 |

E0 与 E1 对比 MDCC，E7 与 E8 对比额外的 Edge Loss，E9 与 E10 对比同样边界监督下的 BGGF。E2、E4、E6、E7 都有供 BGGF 使用的边界预测，Edge Loss 权重为 0。

### 选择运行范围

完成第 3 节的数据与权重准备后，在仓库根目录使用同一个 Anaconda Prompt 窗口。课堂配置给各组两轮预算，默认种子为 42。

```bat
set "CTN_ABLATION_BASE=configs\tutorial\local_e0_warmup.yaml"
set "CTN_ABLATION_ROOT=results\tutorial\ablation_2epochs"
```

按需选择一条命令生成计划。省略 `--experiments` 时默认选择 E0。单组可以将 E0 换为 E1 至 E10 的任意组号。自选多组时，用逗号分隔组号并用双引号包住整串。

```bat
rem 单独跑基线
python -m scripts.run_ablation --matrix "configs\ablation.yaml" --base-config "%CTN_ABLATION_BASE%" --output-root "%CTN_ABLATION_ROOT%" --experiments E0 --dry-run
rem 跑完整的 E0–E10，共 11 组
python -m scripts.run_ablation --matrix "configs\ablation.yaml" --base-config "%CTN_ABLATION_BASE%" --output-root "%CTN_ABLATION_ROOT%" --experiments core --dry-run
rem 自选对照组
python -m scripts.run_ablation --matrix "configs\ablation.yaml" --base-config "%CTN_ABLATION_BASE%" --output-root "%CTN_ABLATION_ROOT%" --experiments "E7,E8" --dry-run
```

检查各组的 `planned_config.yaml`。选 `core` 时应看到 `E0_seed42` 到 `E10_seed42` 共 11 个计划目录。核对配置后执行对应范围的训练命令。

```bat
python -m scripts.run_ablation --matrix "configs\ablation.yaml" --base-config "%CTN_ABLATION_BASE%" --output-root "%CTN_ABLATION_ROOT%" --experiments E0 --resume
python -m scripts.run_ablation --matrix "configs\ablation.yaml" --base-config "%CTN_ABLATION_BASE%" --output-root "%CTN_ABLATION_ROOT%" --experiments core --resume
python -m scripts.run_ablation --matrix "configs\ablation.yaml" --base-config "%CTN_ABLATION_BASE%" --output-root "%CTN_ABLATION_ROOT%" --experiments "E7,E8" --resume
```

`--resume` 启动新组或恢复未完成的组，也会复用配置完全相同的已完成组的最佳权重。可以先跑 E0，再用同一输出目录运行 `core`。正式实验可将 `CTN_ABLATION_BASE` 改为 `configs\tutorial\local_e0.yaml`，用 `CTN_ABLATION_ROOT` 指定新结果目录，并为计划与训练命令同时加上 `--seeds "42,123,3407"`。

### 汇总与评价

训练完成后汇总该目录下已生成的验证结果。只汇总单组时可将 `--root` 指向相应的 run 目录。

```bat
python -m scripts.summarize_results --root "%CTN_ABLATION_ROOT%" --output "%CTN_ABLATION_ROOT%_summary" --split val
```

各组的 `execution.json` 记录完成状态。固定阈值的指标在 `validation_best.json`。`val_metrics.json` 与汇总 CSV 使用各组在验证集选出的阈值，比较时把阈值一并记录。锁定待报告的模型和指标口径后，用 `set "CTN_RUN=%CTN_ABLATION_ROOT%\E8_seed42"` 选中待评价的实验组，再按第 5 节冻结并测试。

只有 CPU 时，可以用第 2 节生成的 E0 配置作为共同基底，在同一 checkout 执行 E0 和 E1 两组。

```bat
python -m scripts.run_ablation --matrix "configs\ablation.yaml" --base-config "results\tutorial\smoke_first\E0_seed42\config.yaml" --experiments "E0,E1" --output-root "results\tutorial\ablation_cpu" --resume
```

这条路线在相同合成数据和一轮预算下演示对照设置。

## 运行时常见卡点

| 现象 | 检查项 |
|---|---|
| Python 找不到 `scripts` 或配置文件 | 在含 `scripts/` 和 `configs/` 的仓库根目录运行 |
| CUDA 为 `False` | 核对当前解释器、wheel 与驱动，或先执行 CPU 合成路线 |
| `No images in` | 核对该集合的 `images/` 目录和配置中的 `data.image_glob` |
| `Missing or unreadable mask` | 按报错路径补齐对应掩码，核对文件名主体及 PNG 后缀 |
| `Image/mask shape mismatch` | 根据报错的两条路径找到原图和掩码，导出与原图同尺寸的标签 |
| `ResNet34 weights failed` | 按第 3 节准备编码器权重 |
| 训练或预测目录已有产物 | 续训保留原实验目录，新实验或新预测使用新目录 |
| `Finish the configured training budget before freezing` | 查看训练状态并完成预算或早停流程 |

第三方实现与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
