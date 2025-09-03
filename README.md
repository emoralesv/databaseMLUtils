# databaseMLUtils

Utilities for dataset IO, annotation conversions, preprocessing transforms, and dataset reporting.

- Convert detection/segmentation datasets into classification crops (object isolation, resize/normalize, dynamic crop).
- Convert polygons ↔ bounding boxes (AABB and oriented OBB).
- Generate dataset reports (class distribution, object size histograms, split summaries).
- CLI: `dbutils convert` and `dbutils report`.

## Install

- Python 3.9+
- Create and activate a virtualenv
- Install:

```
pip install -r requirements.txt
pip install -e .
```

## Quickstart

Convert COCO detection to classification crops with optional resize and dynamic crop:

```python
from databaseMLUtils.io import iter_coco_objects
from databaseMLUtils.transforms import dynamic_crop
from databaseMLUtils.bbox import normalize_bbox

for img, ann in iter_coco_objects("path/to/instances.json", images_dir="path/to/images"):
    x, y, w, h = ann["bbox"]
    crop = dynamic_crop(img, (x, y, w, h), margin=0.1, square=True)
    # save crop per class
```

Generate a class distribution report from a classification folder structure:

```python
from databaseMLUtils.reporting import class_distribution_report

df = class_distribution_report("results/crops", out_csv="results/reports/class_dist.csv")
print(df)
```

## CLI

- Convert detection to classification crops:
```
dbutils convert --task det2cls --ann path/instances.json --images path/images --out results/crops \
  --resize none --dynamic-crop on --size 128
```

- Report dataset stats:
```
dbutils report --task cls --src results/crops --out results/reports
```

## Modules

- `io`: COCO/VOC readers and iterators for annotations.
- `bbox`: bbox normalization, cropping helpers.
- `polygons`: polygon → AABB/OBB conversions (uses shapely if available, falls back to numpy PCA).
- `transforms`: dynamic crop, resize helpers.
- `reporting`: class distribution, object size histogram, basic image stats.

## Notes

- Functions include type hints and small, composable APIs.
- Shapely is optional; fallbacks are provided.
- Deterministic options are available for unit tests.

