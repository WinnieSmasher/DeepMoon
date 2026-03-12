# DeepMoon

本项目围绕原DeepMoon(https://github.com/silburt/DeepMoon)思路进行了工程化重构：将原始的脚本式流程拆分为清晰的 deepmoon/包结构，统一了配置、训练、推理、后处理和测试入口，并提供 Attention U-Net 与 TransUNet 两种模型实现。

---

## 项目概览

DeepMoon 当前覆盖了完整的实验主链路：

​	1.数据准备：支持读取 HDF5 数据，也支持在缺少数据时自动生成轻量合成样本。

​	2.模型训练：提供 AttentionUNet 与 TransUNet 两种分割模型。

​	3.模型推理：输出概率图并缓存为 HDF5，避免重复计算。

​	4.后处理：基于模板匹配提取陨石坑，并在全局经纬度空间内去重。

​	5.实验复现：支持统一随机种子、确定性实验模式与 pytest 测试。

---

## 目录结构

```text
deepmoon/
├── data/               # 数据集读取、数据增强、合成样本生成、目录处理
├── models/             # AttentionUNet、TransUNet 与共享网络层
├── postprocessing/     # 模板匹配、坐标换算、唯一陨石坑提取
├── training/           # 损失函数、指标、训练器
├── utils/              # 随机种子、预处理等工具函数
└── config.py           # YAML 配置加载与 CLI 覆盖

scripts/
├── train.py                           # 训练入口
├── predict.py                         # 推理与陨石坑提取入口
├── download_training_data.py          # 真实数据下载入口
├── run_claude.py                      # 使用环境变量启动 Claude Code
├── test_api.py                        # x-api-key 鉴权连通性测试
└── test_api_v2.py                     # Bearer 鉴权连通性测试

configs/default.yaml   # 默认训练/推理配置
tests/                 # pytest 测试用例
legacy/                # 原始历史实现，仅作对照参考
catalogues/            # 月球陨石坑目录数据
docs/                  # 补充文档与说明材料
```

---

## 核心设计

### 1. 统一配置驱动

所有训练、推理、后处理参数都通过 configs/default.yaml 管理，并支持使用 --set a.b=value 在命令行覆盖。

例如：

```bash
python "scripts/train.py" \
  --config "configs/default.yaml" \
  --set training.epochs=8 \
  --set data.batch_size=4
```

### 2. 数据读取与回退机制

deepmoon.data.dataset支持两类数据来源：

​	1.真实 HDF5 文件；

​	2.自动生成的轻量合成样本。

当 data.use_synthetic_if_missing=true且指定路径不存在时，项目会自动创建一个最小可运行的数据集，便于快速验证训练与推理链路。

### 3. 两类分割模型

- AttentionUNet：带注意力门控跳连的 U-Net 变体；
- TransUNet：在瓶颈层加入二维 Transformer 编码器，增强全局上下文建模能力。

### 4. 可复现训练

项目支持：

​	1.experiment.seed控制统一随机种子；

​	2.experiment.deterministic=true开启确定性实验模式；

​	3.DataLoader worker随机性同步管理；

​	4.最优检查点自动保存。

---

## 环境安装

项目使用 Python 3.9+。

```bash
python -m venv ".venv"
source ".venv/bin/activate"
pip install -e ".[dev]"
```

主要依赖包括：

- torch
- numpy
- pandas
- h5py
- opencv-python
- scikit-image
- Cartopy
- PyYAML

---

## 默认配置说明

默认配置文件位于configs/default.yaml，其中最常用的字段如下：

- experiment.seed：实验随机种子。
- experiment.deterministic：是否启用确定性模式。
- data.train_path / val_path / test_path：训练、验证、测试 HDF5 路径。
- data.use_synthetic_if_missing：缺少数据时是否自动生成合成样本。
- data.image_size：输入图像边长。
- model.arch：模型类型，可选 attention_unet 或 trans_unet。
- training.epochs：训练轮数。
- training.device：设备选择，默认 auto。
- prediction.model_path：推理时加载的模型权重路径。
- postprocessing.template_thresh / target_thresh：模板匹配阈值。

如果你只想改少量参数，优先使用命令行覆盖，而不是复制整份配置。

---

## 快速开始

### 1. 生成最小可运行数据

```bash
python "scripts/train.py" \
  --config "configs/default.yaml" \
  --seed 2026
```

由于项目文件大小的原因，在默认路径下没有现成数据，如果配置里开启了data.use_synthetic_if_missing=true，脚本会生成用于调试的小样本数据。

### 2. 下载真实训练数据（可选）

如果你希望使用真实 HDF5 数据进行训练，而不是合成小样本，可以先执行：

```bash
python "scripts/download_training_data.py" \
  --output-dir "data/external/zenodo"
```

默认会下载以下文件：

​	1.train_images.hdf5`

​	2.dev_images.hdf5

​	3.test_images.hdf5

下载来源为 Zenodo：https://doi.org/10.5281/zenodo.1133969

### 3. 训练模型

```bash
python "scripts/train.py" \
  --config "configs/default.yaml" \
  --model trans_unet \
  --epochs 2 \
  --seed 2026 \
  --deterministic
```

训练完成后，最佳检查点默认保存到：

```text
outputs/checkpoints/best.pt
```

### 4. 运行推理与陨石坑提取

```bash
python "scripts/predict.py" \
  --config "configs/default.yaml" \
  --model-path "outputs/checkpoints/best.pt"
```

默认会输出：

- 概率图缓存：outputs/predictions/test_preds.hdf5
- 陨石坑结果：outputs/predictions/test_craterdist.npy

---

## 真实数据评估

如果你已经准备好真实留出集与对应检查点，可以运行：

```bash
python "scripts/predict.py" \
  --config "configs/default.yaml" \
  --model-path "outputs/checkpoints/best.pt" \
  --data-path "data/external/zenodo/dev_images.hdf5" \
  --prediction-path "outputs/predictions/dev_preds.hdf5" \
  --result-path "outputs/predictions/dev_craterdist.npy"
```

该脚本会：

1. 计算测试集分割指标；
2. 在验证集上搜索较优的模板匹配阈值；
3. 在测试集上给出最终检测指标；
4. 以 JSON 形式打印结果。

---

## 严格对照实验

仓库内保留了用于公平对比的辅助脚本：

### 验证新旧生成流程等价性

```bash
python "scripts/test_api_v2.py"
```

### 准备固定对照数据集

```bash
python "scripts/test_api.py"
```

---

## 数据与输出格式

### 输入 HDF5

默认约定包含以下数据集：

- input_images
- target_masks

部分后处理流程还会读取以下元数据分组：

- longlat_bounds
- pix_distortion_coefficient

### 推理输出

- predictions：模型输出的概率图；
- *.npy：保存为 N x 3 的陨石坑数组，通常表示经度、纬度和半径/尺度信息。

---

## 测试与质量检查

运行测试：

```bash
pytest "tests/"
```

运行静态检查：

```bash
flake8 "deepmoon" "scripts" "tests"
```

