# 基于 YOLO26 的自行车检测系统

支持 `bicycle` 单类别检测，提供 PyQt5 桌面端与 Flask 网页端。桌面端支持单图片、文件夹批量、视频和摄像头/RTSP/HTTP 视频流，并将结果保存到 `output/`。

## 安装

```bash
conda create -n yolo26 python=3.9
conda activate yolo26
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

训练结束后，请将权重复制到 `models/best.pt`，或在桌面端点击“选择模型”加载权重。

## 运行

```bash
python GUI.py                 # PyQt5 桌面端
python WEB_UI/web.py          # Web 端：http://127.0.0.1:5000
python train.py               # 训练
python val.py                 # 测试图片验证
python predict.py             # 命令行推理
```

## VOC 数据处理

原始目录应包含 `VOCData/bicycle_dataset/Annotations` 与 `JPEGImages`。运行：

```bash
python VOCData/ViewCategory.py --annotations VOCData/bicycle_dataset/Annotations
python VOCData/xml2txt.py --input VOCData/bicycle_dataset/Annotations --output VOCData/bicycle_dataset/labels --classes bicycle
python VOCData/SplitDataset.py
```

`SplitDataset.py` 会按 8:1:1 将同名图片与标签复制到 `VOCData/bicycle_dataset/VOdevkit/{train,val,test}/{images,labels}`。训练配置见 `VOCData/mydata.yaml`。

## 项目结构

```text
GUI.py                 PyQt5 桌面端
WEB_UI/web.py          Flask 网页端
models/best.pt         训练后放置的模型权重
VOCData/               XML 转换、类别统计、数据集划分工具
train.py / val.py / predict.py
output/                检测结果保存目录
```
