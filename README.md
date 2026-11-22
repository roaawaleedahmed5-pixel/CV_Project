# CV Project

This repository contains the dataset handover for the team's Oxford-IIIT Pet breed classification project. It provides fixed dataset splits, a shared RGB loader, and checks for reproducing the handover.

## Dataset splits

| Split | Images | Use |
| --- | ---: | --- |
| TRAIN | 2,944 | Fit models |
| VAL | 736 | Compare methods and tune settings |
| TEST | 3,669 | Final evaluation using the official test split |

Use the saved CSVs in [data/manifests/](data/manifests/) in row order. Each split has a matching ID list. The working label is the official breed class ID minus one, giving labels 0–36. Keep the image IDs and labels in manifest order when saving feature matrices.

## Load an image

From the repository root, use Python 3.11 with NumPy and Pillow. Place the extracted `images/` and `annotations/` folders under `data/raw/`, or set `PET_RAW_ROOT` to the directory that contains them. Each teammate supplies their own dataset files.

~~~python
import csv
import os
from pathlib import Path

from src.data_loader import load_rgb

raw_root = Path(os.environ.get("PET_RAW_ROOT", "data/raw"))
with Path("data/manifests/train.csv").open(encoding="utf-8", newline="") as stream:
    record = next(csv.DictReader(stream))

image = load_rgb(raw_root / record["image_relative_path"])
print(record["image_id"], image.shape, int(record["label"]))
~~~

The loader returns a writable `uint8` RGB array with shape `(height, width, 3)` and the original image dimensions. For RGBA inputs, it checks that every alpha value is fully opaque before dropping the alpha channel.

## Verify the handover

Run the fixture tests from the repository root. They use temporary files and require no local dataset.

~~~sh
python -B -m unittest discover -s tests -v
~~~

See the [dataset handover guide](docs/DATA_HANDOVER.md) for the verification evidence, split algorithm, manifest hashes, and team workflow.
