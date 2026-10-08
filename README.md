# LDA-YOLO 可运行复现版

本项目根据 `LDA-YOLO.docx` 中的网络图、公式和实验设置实现。程序包含论文提出的四个核心模块：SGConv、IC2f、DCFM 和 DDAM，并把它们接入三尺度、无锚框的 YOLO 风格检测器。项目可训练、评估、推理，也自带不依赖外部数据的合成数据演示。

## 实现范围

- SGConv：`1/2 + 1/4 + 1/4` 通道切分，3x3/5x5/非对称分支，GhostConv、ECA、残差与通道洗牌。
- IC2f：全局通道门控、深度卷积空间门控、ECA、双分支自适应加权、残差与通道洗牌。
- DCFM：高宽双轴自注意力与膨胀/非对称深度卷积分支动态融合。
- DDAM：空间多尺度分支、Haar DWT/IDWT 频域分支和双域自适应融合。
- 检测器：输出步长为 4、8、16 的三个检测尺度，优先保留小目标细节。
- 训练：SGD，动量 0.937，权重衰减 0.0005，初始学习率 0.01，余弦退火到 0.001；默认 200 个 epoch、batch size 8。
- 数据增强：Mosaic、MixUp、随机翻转与随机缩放。
- 评估：Precision、Recall、mAP@0.5 和 mAP@0.5:0.95。

论文没有给出逐层通道数、每层重复次数、检测头及损失函数的全部工程细节，也没有附公开代码。因此本项目对缺失部分采用了清晰、可复现的实现：三尺度无锚框检测头与 IoU/BCE 训练目标。四个创新模块与论文公式一一对应，但本项目不能被视为作者原始代码或论文数值的逐位复现。

## 环境

推荐 Python 3.10 或更高版本。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果需要使用 NVIDIA GPU，请按 PyTorch 官网为本机驱动选择对应 CUDA wheel；代码会自动优先使用 CUDA。

## 立即验证

只检查模型前向、反向、小波重建和解码：

```powershell
python smoke_test.py
```

完整跑一遍“生成数据 -> 训练 -> 评估 -> 推理”：

```powershell
.\run_demo.ps1
```

预测结果保存到 `runs/predict/demo.jpg`。

## 使用真实数据集

标签采用常见 YOLO 格式，每张图片对应一个同名 `.txt`：

```text
class_id center_x center_y width height
```

坐标均归一化到 0 到 1。目录建议如下：

```text
datasets/VisDrone2019/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

项目提供 `configs/visdrone.yaml`、`configs/dior.yaml` 和 `configs/dota.yaml`。修改其中的 `path` 即可。DOTA 原始标注是旋转多边形，本实现使用水平框检测头，需先将多边形转换为外接水平框并输出 YOLO 标签。

按论文设置训练 VisDrone：

```powershell
python train.py `
  --data configs/visdrone.yaml `
  --epochs 200 `
  --batch 8 `
  --img-size 640 `
  --lr 0.01 `
  --final-lr 0.001 `
  --momentum 0.937 `
  --weight-decay 0.0005
```

评估：

```powershell
python evaluate.py `
  --weights runs/train/lda_yolo/best.pt `
  --data configs/visdrone.yaml
```

单图推理：

```powershell
python predict.py `
  --weights runs/train/lda_yolo/best.pt `
  --source path/to/image.jpg `
  --output runs/predict/result.jpg
```

## 主要文件

- `lda_yolo/modules.py`：四个论文模块、SPPF、Haar DWT/IDWT。
- `lda_yolo/model.py`：主干、PAN/FPN 式颈部和三尺度检测头。
- `lda_yolo/loss.py`：目标分配与 IoU/BCE 损失。
- `lda_yolo/data.py`：YOLO 数据读取和论文所用增强。
- `train.py`、`evaluate.py`、`predict.py`：训练、评估和推理入口。
- `tests/test_model.py`：模块形状、小波重建及反向传播测试。

## 调整模型大小

`--base-channels` 控制宽度，必须是 4 的倍数。默认 24；显存有限时可用 16，快速测试可用 8。论文报告的参数量和 GFLOPs 依赖作者未公开的精确通道及重复次数，本参数用于在准确率和开销之间进行工程调整。

