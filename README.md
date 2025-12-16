# Blackgram Plant Leaf Disease Classification Dataset

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](#changelog)

A dataset of blackgram (Vigna mungo) plant leaf images for disease classification, collected and organized for computer vision and deep learning research in agricultural applications. This dataset follows the standardized dataset structure specification.

- Project page: ``https://data.mendeley.com/public-files/datasets/zfcv9fmrgv/files/b29e2738-406d-4fd5-a1ef-b3401217cdaa/file_downloaded``

## TL;DR
- Task: classification (with full-image bounding boxes)
- Modality: RGB
- Platform: handheld/field
- Real/Synthetic: real
- Images: 1,007 across 5 disease/health categories
- Resolution: 512×512 pixels
- Annotations: per-image CSV and JSON; COCO format available
- License: CC BY 4.0 (see LICENSE)
- Citation: see below

## Table of contents
- [Download](#download)
- [Dataset structure](#dataset-structure)
- [Sample images](#sample-images)
- [Annotation schema](#annotation-schema)
- [Stats and splits](#stats-and-splits)
- [Quick start](#quick-start)
- [Evaluation and baselines](#evaluation-and-baselines)
- [Datasheet (data card)](#datasheet-data-card)
- [Known issues and caveats](#known-issues-and-caveats)
- [License](#license)
- [Citation](#citation)
- [Changelog](#changelog)
- [Contact](#contact)

## Download
- Original dataset: ``https://data.mendeley.com/public-files/datasets/zfcv9fmrgv/files/b29e2738-406d-4fd5-a1ef-b3401217cdaa/file_downloaded``
- This repo hosts structure and conversion scripts only; place the downloaded folders under this directory.
- Local license file: see `LICENSE` (Creative Commons Attribution 4.0 International).

## Dataset structure

This dataset follows the standardized dataset structure specification:

```
blackgram_plant_leaf_disease_classification/
├── blackgrams/                      # Main category directory
│   ├── healthy/                     # Healthy leaves subcategory
│   │   ├── csv/                     # CSV annotation files (per-image)
│   │   ├── json/                    # JSON annotation files (per-image)
│   │   ├── images/                  # Image files
│   │   └── sets/                    # Dataset split files for healthy
│   │       ├── train.txt
│   │       ├── val.txt
│   │       ├── test.txt
│   │       ├── all.txt
│   │       └── train_val.txt
│   ├── anthracnose/                 # Anthracnose disease subcategory
│   │   ├── csv/                     # CSV annotation files (per-image)
│   │   ├── json/                    # JSON annotation files (per-image)
│   │   ├── images/                  # Image files
│   │   └── sets/                    # Dataset split files for anthracnose
│   │       ├── train.txt
│   │       ├── val.txt
│   │       ├── test.txt
│   │       ├── all.txt
│   │       └── train_val.txt
│   ├── leaf_crinckle/               # Leaf Crinckle disease subcategory
│   │   └── ... (same structure)
│   ├── powdery_mildew/              # Powdery Mildew disease subcategory
│   │   └── ... (same structure)
│   ├── yellow_mosaic/               # Yellow Mosaic disease subcategory
│   │   └── ... (same structure)
│   └── labelmap.json               # Label mapping file
│
├── annotations/                     # COCO format JSON files (generated)
│   ├── combined_instances_train.json
│   ├── combined_instances_val.json
│   └── combined_instances_test.json
│
├── scripts/                         # Utility scripts
│   └── convert_to_coco.py          # Convert CSV to COCO format
│
├── data/                            # Data directory
│   └── origin/                      # Original data (moved here to reduce dataset size)
│       ├── Anthracnose 230/        # Original anthracnose images directory
│       ├── Healthy 220/            # Original healthy images directory
│       ├── Leaf Crinckle 150/      # Original leaf crinckle images directory
│       ├── Powdery Mildew 180/     # Original powdery mildew images directory
│       ├── Yellow Mosaic 220/      # Original yellow mosaic images directory
│       ├── create_sample_structure.py
│       ├── generate_coco_annotations.py
│       ├── generate_coco_annotations_corrected.py
│       ├── generate_coco_annotations_final.py
│       └── generate_coco_annotations_v2.py
│
├── LICENSE                          # License file
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

**Splits**: Splits provided via `blackgrams/{subcategory}/sets/*.txt`. Each subcategory (healthy, anthracnose, leaf_crinckle, powdery_mildew, yellow_mosaic) has its own split files. List image basenames (no extension). If missing, all images are used.

## Sample images

Below are example images from this dataset. Paths are relative to this README location.

<table>
  <tr>
    <th>Category</th>
    <th>Sample</th>
  </tr>
  <tr>
    <td><strong>Healthy</strong></td>
    <td>
      <img src="blackgrams/healthy/images/69h.jpg" alt="Healthy blackgram leaf" width="260"/>
      <div align="center"><code>blackgrams/healthy/images/69h.jpg</code></div>
    </td>
  </tr>
  <tr>
    <td><strong>Anthracnose</strong></td>
    <td>
      <img src="blackgrams/anthracnose/images/1a.jpg" alt="Anthracnose disease" width="260"/>
      <div align="center"><code>blackgrams/anthracnose/images/1a.jpg</code></div>
    </td>
  </tr>
  <tr>
    <td><strong>Leaf Crinckle</strong></td>
    <td>
      <img src="blackgrams/leaf_crinckle/images/1l.jpg" alt="Leaf Crinckle disease" width="260"/>
      <div align="center"><code>blackgrams/leaf_crinckle/images/1l.jpg</code></div>
    </td>
  </tr>
  <tr>
    <td><strong>Powdery Mildew</strong></td>
    <td>
      <img src="blackgrams/powdery_mildew/images/1p.jpg" alt="Powdery Mildew disease" width="260"/>
      <div align="center"><code>blackgrams/powdery_mildew/images/1p.jpg</code></div>
    </td>
  </tr>
  <tr>
    <td><strong>Yellow Mosaic</strong></td>
    <td>
      <img src="blackgrams/yellow_mosaic/images/1y.jpg" alt="Yellow Mosaic disease" width="260"/>
      <div align="center"><code>blackgrams/yellow_mosaic/images/1y.jpg</code></div>
    </td>
  </tr>
</table>

## Annotation schema

### CSV Format

Each image has a corresponding CSV file in `blackgrams/{subcategory}/csv/` with the following format:

```csv
#item,x,y,width,height,label
0,0,0,512,512,1
```

- **Columns**: `#item`, `x`, `y`, `width`, `height`, `label`
- **Coordinates**: Top-left corner (x, y) + width and height in pixels
- **Label**: Category ID (1=healthy, 2=anthracnose, 3=leaf_crinckle, 4=powdery_mildew, 5=yellow_mosaic)
- For classification tasks, each image has a full-image bounding box `[0, 0, image_width, image_height]` with the category ID.

### COCO Format

COCO format JSON files are generated in the `annotations/` directory. Example structure:
```json
{
  "info": {
    "year": 2025,
    "version": "1.0",
    "description": "Blackgram Plant Leaf Disease Classification blackgrams train split (combined)",
    "url": "https://data.mendeley.com/public-files/datasets/zfcv9fmrgv/files/b29e2738-406d-4fd5-a1ef-b3401217cdaa/file_downloaded"
  },
  "images": [
    {
      "id": 2716113204,
      "file_name": "blackgrams/healthy/images/69h.jpg",
      "width": 512,
      "height": 512
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 2716113204,
      "category_id": 1,
      "bbox": [0, 0, 512, 512],
      "area": 262144,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "background",
      "supercategory": "background"
    },
    {
      "id": 1,
      "name": "healthy",
      "supercategory": "blackgram_leaf_disease"
    },
    {
      "id": 2,
      "name": "anthracnose",
      "supercategory": "blackgram_leaf_disease"
    },
    {
      "id": 3,
      "name": "leaf_crinckle",
      "supercategory": "blackgram_leaf_disease"
    },
    {
      "id": 4,
      "name": "powdery_mildew",
      "supercategory": "blackgram_leaf_disease"
    },
    {
      "id": 5,
      "name": "yellow_mosaic",
      "supercategory": "blackgram_leaf_disease"
    }
  ]
}
```

### Label Maps

Label mapping is defined in `blackgrams/labelmap.json`:

```json
[
  {"object_id": 0, "label_id": 0, "keyboard_shortcut": "0", "object_name": "background"},
  {"object_id": 1, "label_id": 1, "keyboard_shortcut": "1", "object_name": "healthy"},
  {"object_id": 2, "label_id": 2, "keyboard_shortcut": "2", "object_name": "anthracnose"},
  {"object_id": 3, "label_id": 3, "keyboard_shortcut": "3", "object_name": "leaf_crinckle"},
  {"object_id": 4, "label_id": 4, "keyboard_shortcut": "4", "object_name": "powdery_mildew"},
  {"object_id": 5, "label_id": 5, "keyboard_shortcut": "5", "object_name": "yellow_mosaic"}
]
```

## Stats and splits

### Dataset Statistics

- **Total images**: 1,007
- **Healthy category**: 221 images
- **Anthracnose category**: 230 images
- **Leaf Crinckle category**: 152 images
- **Powdery Mildew category**: 180 images
- **Yellow Mosaic category**: 224 images
- **Image dimensions**: 512×512 pixels
- **Image format**: JPEG

### Splits

Splits provided via `blackgrams/{subcategory}/sets/*.txt`. Each subcategory has its own split files. You may define your own splits by editing those files.

Default split ratios:
- **Training**: 60%
- **Validation**: 20%
- **Test**: 20%

**Detailed Split Statistics**:
- Training set: 603 images (`combined_instances_train.json`)
  - Healthy: 132 images
  - Anthracnose: 138 images
  - Leaf Crinckle: 91 images
  - Powdery Mildew: 108 images
  - Yellow Mosaic: 134 images
- Validation set: 200 images (`combined_instances_val.json`)
  - Healthy: 44 images
  - Anthracnose: 46 images
  - Leaf Crinckle: 30 images
  - Powdery Mildew: 36 images
  - Yellow Mosaic: 44 images
- Test set: 204 images (`combined_instances_test.json`)
  - Healthy: 45 images
  - Anthracnose: 46 images
  - Leaf Crinckle: 31 images
  - Powdery Mildew: 36 images
  - Yellow Mosaic: 46 images

- **Classes**: 5 (healthy, anthracnose, leaf_crinckle, powdery_mildew, yellow_mosaic)

## Quick start

### 1. Convert to COCO Format

```bash
python scripts/convert_to_coco.py --root . --out annotations \
    --category blackgrams --splits train val test --combined
```

### 2. Load with COCO API

```python
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# Load annotations
coco = COCO('annotations/combined_instances_train.json')

# Get image IDs
img_ids = coco.getImgIds()
print(f"Number of images: {len(img_ids)}")

# Load image info
img_info = coco.loadImgs(img_ids[0])[0]
print(f"Image: {img_info['file_name']}")

# Load annotations for image
ann_ids = coco.getAnnIds(imgIds=img_ids[0])
anns = coco.loadAnns(ann_ids)
print(f"Number of annotations: {len(anns)}")
```

### Dependencies

**Required**:
- Python 3.6+
- Pillow >= 9.5

**Optional** (for COCO API):
- pycocotools >= 2.0.7

Install dependencies:
```bash
pip install -r requirements.txt
```

## Evaluation and baselines

### Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score
- **Detection**: mAP@[.50:.95], mAP@0.50, mAP@0.75

### Baselines

Baseline results will be added as they become available.

## Datasheet (data card)

### Motivation
This dataset was created to support research in automated plant disease detection and classification, specifically for blackgram (Vigna mungo) crops. Early detection and classification of plant diseases can help farmers take timely action to prevent crop losses.

### Composition
- **Image types**: RGB images of blackgram plant leaves
- **Categories**: 5 classes (1 healthy + 4 disease types)
- **Image format**: JPEG
- **Image resolution**: 512×512 pixels (standardized)
- **Annotation format**: Full-image bounding boxes for classification (COCO format compatible)

### Collection Process

- Images collected from publicly available Mendeley Data repositories
- Healthy images: 221 images
- Anthracnose images: 230 images
- Leaf Crinckle images: 152 images
- Powdery Mildew images: 180 images
- Yellow Mosaic images: 224 images
- All images resized to 512×512 pixels
- Annotations generated automatically with full-image bounding boxes

### Preprocessing

- Images standardized to 512×512 pixels
- JPEG format
- Full-image bounding box annotations for classification
- COCO format annotations generated from CSV files

### Distribution

- Available via Mendeley Data
- Standardized structure for easy integration with deep learning frameworks
- COCO format annotations for compatibility with common detection frameworks

### Maintenance

- Dataset structure follows standardized format
- Conversion scripts provided for format compatibility
- Regular updates as new data becomes available

## Known issues and caveats

1. **File naming**: Images use short names (e.g., `69h.jpg`, `1a.jpg`) with suffixes indicating the category (h=healthy, a=anthracnose, l=leaf_crinckle, p=powdery_mildew, y=yellow_mosaic)
2. **Image size**: All images are 512×512 pixels, which is standardized for consistency
3. **Full-image annotations**: Annotations use full-image bounding boxes for classification tasks
4. **Classification Task**: This dataset uses full-image bounding boxes for classification. Each image has a single bounding box covering the entire image `[0, 0, width, height]` with the category ID indicating the disease/health status
5. **Coordinate system**: Bounding boxes use top-left corner (x, y) + width and height format

## License

This dataset is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

Check the original dataset terms and cite appropriately. See `LICENSE` file for full license text.

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{blackgram_leaf_disease_2025,
  title={Blackgram Plant Leaf Disease Classification Dataset},
  author={Dataset Contributors},
  year={2025},
  publisher={Mendeley Data},
  url={https://data.mendeley.com/public-files/datasets/zfcv9fmrgv/files/b29e2738-406d-4fd5-a1ef-b3401217cdaa/file_downloaded},
  license={CC BY 4.0}
}
```

## Changelog

- **V1.0.0** (2025): Initial standardized structure and COCO conversion utility
  - Reorganized dataset into standardized structure following classification task pattern
  - Created labelmap.json with all 5 categories
  - Generated CSV annotations from JSON files
  - Created dataset splits (train/val/test) for each subcategory
  - Added convert_to_coco.py script for COCO format generation
  - Updated README.md following standard format

## Contact

- **Maintainers**: [Your Name/Organization]
- **Original authors**: Mendeley Data Contributors
- **Source**: `https://data.mendeley.com/public-files/datasets/zfcv9fmrgv/files/b29e2738-406d-4fd5-a1ef-b3401217cdaa/file_downloaded`
