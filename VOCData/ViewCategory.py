"""Count VOC object categories."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


def count_categories(annotation_dir: Path) -> Counter:
    result = Counter()
    for xml_file in annotation_dir.glob("*.xml"):
        root = ET.parse(xml_file).getroot()
        result.update(obj.findtext("name", "unknown") for obj in root.findall("object"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    counts = count_categories(args.annotations)
    if not counts:
        print("未找到标注对象")
    for category, count in counts.most_common():
        print(f"{category}: {count}")
