# AI Task2: 基于 SAM 的遥感综合实验

本项目围绕 PDF《综合实践 - 实验二-sam》实现了一个可运行的 SAM 遥感实验工程，当前仓库主要支持以下能力：

1. Web UI 上传图像并执行 SAM 自动分割
2. 在 UI 中进行类别修正、实例合并、重分割、点提示新增实例
3. 生成规则报告，并支持在网页中配置 DeepSeek 报告
4. 上传 `.shp/.dbf/.shx` 等组件进行 Shapefile 标注预览
5. 针对当前 `dataset/` 下的 DOTA 原始压缩包执行实验版批量测试并生成量化评估报告

## 当前仓库结构

当前仓库中真实存在、建议直接使用的入口如下：

```text
AI-task2/
├── app.py
├── README.md
├── test_dota_dataset.py
├── asset/
│   └── sam_vit_b_01ec64.pth
├── config/
│   └── deepseek_config.example.json
├── data/
│   ├── annotations/
│   ├── archive_samples/
│   ├── exports/
│   ├── reports/
│   ├── shp_previews/
│   └── uploads/
├── dataset/
│   ├── images/
│   ├── labelTxt-v1.0/
│   └── labelTxt-v1.5/
├── sam_experiment/
│   ├── app.py
│   ├── annotation_service.py
│   ├── config.py
│   ├── deepseek_config_service.py
│   ├── deepseek_report_service.py
│   ├── report_service.py
│   ├── sam_service.py
│   ├── shp_annotation_service.py
│   ├── storage.py
│   ├── video_service.py
│   ├── static/
│   └── templates/
└── tests/
```

说明：

- 旧版 README 中提到的 `run_experiment.py`、`test_isaid_dataset.py`、`prepare_dota_dataset.py` 等源码目前不在仓库中，不能直接作为当前版本入口使用。
- 当前这次更新已经补齐新的 `test_dota_dataset.py`，用于适配你现在实际拥有的 `dataset/*.zip` 数据，并输出更像实验提交版的量化评估。

## 环境准备

推荐在 `myenv-3.9` 中运行：

```bash
source ~/miniconda3/bin/activate myenv-3.9
```

如需在网页中预览 `.shp` 标注，请先安装：

```bash
pip install pyshp
```

## 当前数据集结构

你现在的 `dataset/` 不是“已解压目录版”，而是“原始压缩包版”：

```text
dataset/
├── images/
│   ├── part1.zip
│   └── part3.zip
├── labelTxt-v1.0/
│   ├── Train_Task2_gt.zip
│   └── labelTxt.zip
└── labelTxt-v1.5/
    ├── DOTA-v1.5_train.zip
    └── DOTA-v1.5_train_hbb.zip
```

这意味着：

- 当前测试不能再依赖旧的 `dataset/DOTA_v1.0/...` 路径
- 新测试脚本会直接从 zip 中抽样图片
- 标签也直接从 zip 中读取，不要求你先手动解压全量数据

## 启动 Web UI

执行：

```bash
python app.py
```

浏览器访问：

```text
http://127.0.0.1:5000
```

UI 当前可用功能：

- 图像上传自动分割
- 视频上传分析
- DeepSeek 配置
- Shapefile 标注预览
- 历史标注和历史 shp 预览查看

## DOTA 压缩包实验测试脚本

### 脚本作用

`test_dota_dataset.py` 会：

1. 扫描 `dataset/images/*.zip`
2. 扫描指定版本的 `dataset/labelTxt-*/` 标签压缩包
3. 自动找出“图片与标签同时存在”的公共样本
4. 只提取少量测试图片到 `data/archive_samples/`
5. 调用 SAM 做自动分割
6. 计算像素级前景指标，如前景 IoU、Precision、Recall、F1
7. 计算基于 bbox 的粗粒度目标匹配指标，如 object precision / recall / F1
8. 记录单张图像耗时、质量分档、最佳/较弱样本
9. 输出 JSON 和 Markdown 汇总报告

### 默认测试

使用 `v1.5` 标签，默认测试 10 张图：

```bash
python test_dota_dataset.py
```

### 指定标签版本

使用 `v1.0` 标签：

```bash
python test_dota_dataset.py --label-version v1.0
```

### 指定测试样本数

只测试 2 张图：

```bash
python test_dota_dataset.py --max-images 2
```

### 指定具体图像编号

例如只测 `P0010` 和 `P0011`：

```bash
python test_dota_dataset.py --image-stems P0010,P0011
```

### 调整 SAM 参数

```bash
python test_dota_dataset.py --points-per-side 32 --max-images 2
```

### 调整目标匹配阈值

脚本会把 `SAM` 预测实例框与 `DOTA` 标注框做粗粒度 bbox 匹配，默认阈值是 `0.25`：

```bash
python test_dota_dataset.py --max-images 10 --match-iou-threshold 0.3
```

### 控制测试图像缩放

由于当前 `dataset/images/*.zip` 中的原图分辨率很高，脚本默认会先把测试样本缩放到“最长边不超过 1536”后再送入 SAM，以避免内存占用过大。

如果你想手动调整，可以使用：

```bash
python test_dota_dataset.py --max-images 1 --max-long-side 1024
```

## 测试输出位置

脚本会自动生成：

- 抽样图片缓存：

```text
data/archive_samples/<label_version>/
```

其中既可能包含原始抽样图片，也可能包含测试时自动生成的缩放版图片，例如：

```text
data/archive_samples/v1.5/P0010.png
data/archive_samples/v1.5/P0010_resized.png
```

- 标注、实例图、叠加图：

```text
data/annotations/
data/exports/
```

- 汇总报告：

```text
data/reports/dota_archive_test_summary_v1_0.json
data/reports/dota_archive_test_summary_v1_0.md
data/reports/dota_archive_test_summary_v1_5.json
data/reports/dota_archive_test_summary_v1_5.md
```

## 评估指标说明

当前脚本采用的是适合“现有压缩包数据”的实验化近似评估：

- `predicted_instances`：SAM 预测实例数
- `ground_truth_objects`：标签文件中解析出的真值目标数
- `absolute_error`：两者绝对差
- `foreground_iou`：预测前景区域与 DOTA 多边形区域的并集 IoU
- `foreground_precision / recall / f1`：像素级前景指标
- `object_precision / recall / f1`：基于 bbox 粗匹配得到的目标级指标
- `runtime_seconds`：单张图像耗时
- `quality_level / quality_score`：脚本根据前景指标和目标指标得到的综合质量分档

注意：

- 这已经比原来的“只看实例数差异”更接近实验提交版，但仍然是近似评估
- 它不是严格的检测 mAP，也不是标准实例分割 benchmark
- 当前脚本默认会先缩放超大图像，再执行 SAM，以保证在普通机器上也能完成测试
- 如果后续你把 DOTA 全量解压并标准化整理，还可以继续扩展成更严格的评估脚本

## Web UI 与测试脚本的关系

- `test_dota_dataset.py` 适合批量验证当前压缩包数据，并生成实验版量化报告
- Web UI 适合单张图片交互式查看与人工修正

推荐顺序：

1. 先用 `python test_dota_dataset.py --max-images 10` 跑一轮实验版测试
2. 再从 `data/archive_samples/` 中挑一张图片，在 UI 中上传查看可视化结果
3. 抽查 `data/reports/dota_archive_test_summary_v1_5.md` 中的最佳/较弱样本
4. 如有需要，再在 UI 中生成规则报告或 DeepSeek 报告

## 单元测试

仓库中的单元测试可通过以下命令运行：

```bash
python -m unittest discover tests
```

如果你只想快速验证首页和 SHP 功能，也可以执行：

```bash
python -m unittest tests.test_app tests.test_shp_annotation_service
```
