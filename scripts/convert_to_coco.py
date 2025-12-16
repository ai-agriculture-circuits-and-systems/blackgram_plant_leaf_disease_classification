#!/usr/bin/env python3
"""
Convert Blackgram Plant Leaf Disease Classification dataset annotations to COCO JSON format.
Based on the standardized dataset structure specification.

License: CC BY 4.0 (see LICENSE). This script is distributed alongside the
dataset and follows the same usage terms. Cite the original dataset in publications.

Usage examples:
    python scripts/convert_to_coco.py --root . --out annotations \
        --category blackgrams --splits train val test --combined
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


def _lower_keys(mapping: Dict[str, str]) -> Dict[str, str]:
    """Return a case-insensitive mapping by lowering keys."""
    return {k.lower(): v for k, v in mapping.items()}


def _read_split_list(split_file: Path) -> List[str]:
    """Read image base names (without extension) from a split file."""
    if not split_file.exists():
        return []
    lines = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def _image_size(image_path: Path) -> Tuple[int, int]:
    """Return (width, height) for an image path using PIL."""
    with Image.open(image_path) as img:
        return img.width, img.height


def _parse_csv_boxes(csv_path: Path) -> List[Dict]:
    """Parse a single per-image CSV file and return COCO-style bboxes.
    
    The parser is resilient to header variants by using case-insensitive
    lookups. Supported schemas:
      - Rectangle: x, y, w/h or dx/dy or width/height
      - Circle: x, y, r (converted to rectangle)
    """
    if not csv_path.exists():
        return []
    
    boxes: List[Dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return boxes
        
        header = _lower_keys({k: k for k in reader.fieldnames})
        
        def get(row: Dict[str, str], *keys: str) -> Optional[float]:
            for key in keys:
                if key in row and row[key] not in (None, ""):
                    try:
                        return float(row[key])
                    except ValueError:
                        continue
            return None
        
        for raw_row in reader:
            row = {k.lower(): v for k, v in raw_row.items()}
            x = get(row, "x", "xc", "x_center")
            y = get(row, "y", "yc", "y_center")
            # Circle
            r = get(row, "r", "radius")
            # Rectangle sizes
            w = get(row, "w", "width", "dx")
            h = get(row, "h", "height", "dy")
            label = get(row, "label", "class", "category_id")
            
            if x is None or y is None:
                continue
            
            category_id = int(label) if label is not None else 1
            
            if r is not None:
                # Convert circle to rectangle
                bbox = [x - r, y - r, 2 * r, 2 * r]
                area = (2 * r) * (2 * r)
            elif w is not None and h is not None:
                bbox = [x, y, w, h]
                area = w * h
            else:
                continue
            
            boxes.append({
                "bbox": bbox,
                "area": area,
                "category_id": category_id,
            })
    
    return boxes


def _load_labelmap(labelmap_path: Path) -> Dict[int, str]:
    """Load labelmap.json and return a mapping from category_id to category name."""
    if not labelmap_path.exists():
        return {}
    
    with open(labelmap_path, 'r') as f:
        labelmap = json.load(f)
    
    return {item["object_id"]: item["object_name"] for item in labelmap if item["object_id"] > 0}


def _collect_annotations_for_split(
    category_root: Path,
    subcategory: str,
    split: str,
    labelmap: Dict[int, str],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Collect COCO dictionaries for images, annotations, and categories."""
    subcat_dir = category_root / subcategory
    images_dir = subcat_dir / "images"
    annotations_dir = subcat_dir / "csv"
    sets_dir = subcat_dir / "sets"
    
    split_file = sets_dir / f"{split}.txt"
    image_stems = set(_read_split_list(split_file))
    
    if not image_stems:
        # Fall back to all images if no split file
        image_stems = {p.stem for p in images_dir.glob("*.jpg")}
        image_stems.update({p.stem for p in images_dir.glob("*.png")})
        image_stems.update({p.stem for p in images_dir.glob("*.bmp")})
    
    images: List[Dict] = []
    anns: List[Dict] = []
    
    # Build categories from labelmap
    categories: List[Dict] = []
    for cat_id, cat_name in sorted(labelmap.items()):
        if cat_id > 0:  # Skip background
            categories.append({
                "id": cat_id,
                "name": cat_name,
                "supercategory": "blackgram_leaf_disease"
            })
    
    image_id_counter = 1
    ann_id_counter = 1
    
    for stem in sorted(image_stems):
        img_path = images_dir / f"{stem}.jpg"
        if not img_path.exists():
            # Try PNG fallback
            img_path = images_dir / f"{stem}.png"
            if not img_path.exists():
                # Try BMP fallback
                img_path = images_dir / f"{stem}.bmp"
                if not img_path.exists():
                    continue
        
        width, height = _image_size(img_path)
        images.append({
            "id": image_id_counter,
            "file_name": f"{category_root.name}/{subcategory}/images/{img_path.name}",
            "width": width,
            "height": height,
        })
        
        csv_path = annotations_dir / f"{stem}.csv"
        for box in _parse_csv_boxes(csv_path):
            anns.append({
                "id": ann_id_counter,
                "image_id": image_id_counter,
                "category_id": box["category_id"],
                "bbox": box["bbox"],
                "area": box["area"],
                "iscrowd": 0,
            })
            ann_id_counter += 1
        
        image_id_counter += 1
    
    return images, anns, categories


def _collect_combined_annotations(
    category_root: Path,
    subcategories: List[str],
    split: str,
    labelmap: Dict[int, str],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Collect annotations from all subcategories and combine them."""
    all_images: List[Dict] = []
    all_anns: List[Dict] = []
    categories: List[Dict] = []
    
    # Build categories from labelmap
    for cat_id, cat_name in sorted(labelmap.items()):
        if cat_id > 0:  # Skip background
            categories.append({
                "id": cat_id,
                "name": cat_name,
                "supercategory": "blackgram_leaf_disease"
            })
    
    image_id_counter = 1
    ann_id_counter = 1
    
    for subcat in subcategories:
        subcat_dir = category_root / subcat
        images_dir = subcat_dir / "images"
        annotations_dir = subcat_dir / "csv"
        sets_dir = subcat_dir / "sets"
        
        split_file = sets_dir / f"{split}.txt"
        image_stems = set(_read_split_list(split_file))
        
        if not image_stems:
            image_stems = {p.stem for p in images_dir.glob("*.jpg")}
            image_stems.update({p.stem for p in images_dir.glob("*.png")})
            image_stems.update({p.stem for p in images_dir.glob("*.bmp")})
        
        for stem in sorted(image_stems):
            img_path = images_dir / f"{stem}.jpg"
            if not img_path.exists():
                img_path = images_dir / f"{stem}.png"
                if not img_path.exists():
                    img_path = images_dir / f"{stem}.bmp"
                    if not img_path.exists():
                        continue
            
            width, height = _image_size(img_path)
            all_images.append({
                "id": image_id_counter,
                "file_name": f"{category_root.name}/{subcat}/images/{img_path.name}",
                "width": width,
                "height": height,
            })
            
            csv_path = annotations_dir / f"{stem}.csv"
            for box in _parse_csv_boxes(csv_path):
                all_anns.append({
                    "id": ann_id_counter,
                    "image_id": image_id_counter,
                    "category_id": box["category_id"],
                    "bbox": box["bbox"],
                    "area": box["area"],
                    "iscrowd": 0,
                })
                ann_id_counter += 1
            
            image_id_counter += 1
    
    return all_images, all_anns, categories


def _build_coco_dict(
    images: List[Dict],
    anns: List[Dict],
    categories: List[Dict],
    description: str,
) -> Dict:
    """Build a complete COCO dict from components."""
    return {
        "info": {
            "year": 2025,
            "version": "1.0.0",
            "description": description,
            "url": "",
        },
        "images": images,
        "annotations": anns,
        "categories": categories,
        "licenses": [],
    }


def convert(
    root: Path,
    out_dir: Path,
    category: str,
    subcategories: Optional[List[str]],
    splits: List[str],
    combined: bool = False,
) -> None:
    """Convert selected category and splits to COCO JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    category_root = root / category
    labelmap_path = category_root / "labelmap.json"
    labelmap = _load_labelmap(labelmap_path)
    
    if subcategories is None:
        # Auto-detect subcategories
        subcategories = [d.name for d in category_root.iterdir() 
                        if d.is_dir() and d.name != "sets" and not d.name.startswith(".")]
        subcategories.sort()
    
    if combined:
        # Generate combined COCO files for all subcategories
        for split in splits:
            images, anns, categories = _collect_combined_annotations(
                category_root, subcategories, split, labelmap
            )
            desc = f"Blackgram Plant Leaf Disease Classification {category} {split} split (combined)"
            coco = _build_coco_dict(images, anns, categories, desc)
            out_path = out_dir / f"combined_instances_{split}.json"
            out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
            print(f"Generated {out_path} with {len(images)} images and {len(anns)} annotations")
    else:
        # Generate separate COCO files for each subcategory
        for subcat in subcategories:
            for split in splits:
                images, anns, categories = _collect_annotations_for_split(
                    category_root, subcat, split, labelmap
                )
                desc = f"Blackgram Plant Leaf Disease Classification {category} {subcat} {split} split"
                coco = _build_coco_dict(images, anns, categories, desc)
                out_path = out_dir / f"{subcat}_instances_{split}.json"
                out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
                print(f"Generated {out_path} with {len(images)} images and {len(anns)} annotations")


def main() -> int:
    """Entry point for the converter CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Dataset root containing category subfolders (default: dataset root)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "annotations",
        help="Output directory for COCO JSON files (default: <root>/annotations)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="blackgrams",
        help="Category to convert (default: blackgrams)",
    )
    parser.add_argument(
        "--subcategories",
        nargs="+",
        type=str,
        default=None,
        help="Subcategories to convert (default: auto-detect all)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=str,
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to generate (default: train val test)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate combined COCO files for all subcategories",
    )
    
    args = parser.parse_args()
    
    convert(
        root=Path(args.root),
        out_dir=Path(args.out),
        category=args.category,
        subcategories=args.subcategories,
        splits=args.splits,
        combined=args.combined,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

