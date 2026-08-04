"""Convert Pascal VOC XML annotations to normalized YOLO TXT labels."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_file(xml_path: Path, output_dir: Path, categories: list[str]):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path} 缺少 size 节点")
    width, height = int(size.findtext("width", "0")), int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"{xml_path} 图像尺寸无效")
    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in categories or obj.findtext("difficult", "0") == "1":
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin, ymin = float(box.findtext("xmin", "0")), float(box.findtext("ymin", "0"))
        xmax, ymax = float(box.findtext("xmax", "0")), float(box.findtext("ymax", "0"))
        x_center, y_center = ((xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height)
        box_width, box_height = ((xmax - xmin) / width, (ymax - ymin) / height)
        if box_width > 0 and box_height > 0:
            lines.append(f"{categories.index(name)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{xml_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Annotations XML 目录", default='/Users/duncan/Documents/Codex/2026-07-30/https-mp-weixin-qq-com-s/VOCData/bicycle_dataset/Bicycle annotated/xml')
    # parser.add_argument("--input", type=Path, required=True, help="Annotations XML 目录", default='/Users/duncan/Documents/Codex/2026-07-30/https-mp-weixin-qq-com-s/VOCData/bicycle_dataset/Bicycle annotated/xml')
    parser.add_argument("--output", type=Path, help="YOLO labels 输出目录", default='/Users/duncan/Documents/Codex/2026-07-30/https-mp-weixin-qq-com-s/VOCData/bicycle_dataset/Bicycle annotated/lables')
    # parser.add_argument("--output", type=Path, required=True, help="YOLO labels 输出目录", default='/Users/duncan/Documents/Codex/2026-07-30/https-mp-weixin-qq-com-s/VOCData/bicycle_dataset/Bicycle annotated/lables')
    parser.add_argument("--classes", nargs="+", default=["Bicycle"])
    args = parser.parse_args()
    for xml_file in sorted(args.input.glob("*.xml")):
        convert_file(xml_file, args.output, args.classes)
    print(f"已转换 {len(list(args.input.glob('*.xml')))} 个文件")
