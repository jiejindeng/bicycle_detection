"""Split matched YOLO images and labels into train/val/test folders."""
import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def split_data(image_dir, label_dir, output_dir, train_rate=0.8, val_rate=0.1, test_rate=0.1, seed=42):
    if round(train_rate + val_rate + test_rate, 6) != 1:
        raise ValueError("训练、验证、测试集比例之和必须为 1。")
    images = {item.stem: item for item in Path(image_dir).iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS}
    labels = {item.stem: item for item in Path(label_dir).glob("*.txt")}
    matched = [(stem, images[stem], labels[stem]) for stem in sorted(images.keys() & labels.keys())]
    if not matched:
        raise FileNotFoundError("未找到同名的图片和 YOLO 标签文件。")
    random.Random(seed).shuffle(matched)
    train_end = int(len(matched) * train_rate)
    val_end = train_end + int(len(matched) * val_rate)
    groups = {"train": matched[:train_end], "val": matched[train_end:val_end], "test": matched[val_end:]}
    output_dir = Path(output_dir)
    for split_name, records in groups.items():
        image_output = output_dir / split_name / "images"
        label_output = output_dir / split_name / "labels"
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        for _, image, label in records:
            shutil.copy2(image, image_output / image.name)
            shutil.copy2(label, label_output / label.name)
        print(f"{split_name}: {len(records)}")
    print("数据集已划分完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按 8:1:1 划分 YOLO 数据集")
    parser.add_argument("--images", type=Path, default=Path("/Users/duncan/Documents/Codex/2026-07-30/https-mp-weixin-qq-com-s/VOCData/bicycle_dataset/Bicycle annotated/images"))
    parser.add_argument("--labels", type=Path, default=Path("/Users/duncan/Documents/Codex/2026-07-30/https-mp-weixin-qq-com-s/VOCData/bicycle_dataset/Bicycle annotated/lables"))
    parser.add_argument("--output", type=Path, default=Path("VOCData/bicycle_dataset/VOdevkit"))
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    split_data(options.images, options.labels, options.output, options.train, options.val, options.test, options.seed)
