# Oxford-IIIT Pet dataset handover

## Local evidence

The Oxford-IIIT Pet archives `images.tar.gz` and `annotations.tar.gz` were downloaded to `data/downloads/` and extracted under `data/raw/`. The [local verification summary](../results/dataset_verification/verification_20261122T081553_577669Z.json) records successful gzip and tar checks, local SHA-256 hashes for both archives, and hash matches for all 7,393 archived image-side files and 18,471 archived annotation-side files. Publisher checksum comparison was skipped because no authoritative digest was available. The official lists contained 3,680 trainval IDs and 3,669 test IDs with no duplicates or overlap. All listed JPG paths existed. The 41 unmatched JPGs were left in place and are outside the manifests.

The [trainval inventory summary](../results/dataset_inventory/summary_20261122T084427_808422Z.json) records 3,680 full Pillow decodes across 37 breeds, with zero errors or warnings. It counted 3,678 RGB photographs and two RGBA photographs, `Egyptian_Mau_14` and `Egyptian_Mau_186`. Widths ranged from 114 to 3,264 pixels and heights from 108 to 2,606 pixels. TEST verification covered official membership and file presence. The decoding inventory covered trainval only.

The notebook's **Inspect RGBA images** section reports the two alpha channels and shows display-only composites. Its code cell has no saved output as of this handover. The approved policy allows removing a fully opaque alpha channel because it contributes no transparency and removing it leaves the original RGB values intact. The loader checks every alpha sample before removal. The notebook's shared-loader check includes both RGBA IDs.

The annotation README's printed species-specific breed limits disagree with some official records. The manifests retain the recorded species and species-specific breed IDs.

## Shared image and label representation

- `src.data_loader.load_rgb` returns an independent writable NumPy array with shape `(height, width, 3)`, dtype `uint8`, and values from 0 through 255. Axes are rows, columns, and RGB channels. Dimensions and decoded RGB values are preserved. Conversion happens in memory.
- RGB loads directly. RGBA is accepted after every alpha sample is confirmed fully opaque for its channel dtype. Nonopaque alpha, other modes, metadata filenames beginning with `._`, and missing or undecodable files raise errors.
- The working `label` is `official_class_id - 1`, so labels 0–36 correspond to official class IDs 1–37. `official_class_id` identifies the breed across the whole dataset. `official_species_breed_id` is the separate label in the original annotation's species. `species_id` keeps the official value of 1 for cat or 2 for dog.

## Fixed splits and files

The split starts from official trainval records and targets 80% TRAIN and 20% VAL within each breed. The implementation sorts official class IDs numerically and sorts each class by image ID in case-sensitive Python string order. A single `np.random.default_rng(42)` then selects `floor(0.2 * class_count + 0.5)` VAL indices for each class with `rng.choice(..., replace=False)`. All remaining records enter TRAIN. Per-class rounding determines the actual ratio. Stratification gives each breed representation in both subsets. Official TEST membership is copied unchanged. Every manifest is then sorted by image ID.

The seed makes the generation procedure repeatable with the recorded NumPy version. The fixture tests used NumPy 2.4.6. The exported manifests are the authoritative assignment for both milestones. Teammates should use the same saved files. The producing NumPy version, actual sizes and per-breed counts, source annotation SHA-256 hashes, and hashes of the seven exported payload files are recorded in `split_info.json`. The JSON omits its own digest because including it would change its bytes. Git keeps LF endings for the manifests so their hashes survive Windows and Unix checkouts.

| File under `data/manifests/` | How to use it |
| --- | --- |
| `train.csv`, `val.csv`, `test.csv` | Load the relevant split in CSV row order. Each row gives the image ID, raw-relative image and trimap paths, working and official labels, species fields, `official_split` (`trainval` or `test`), and assigned `split` (`train`, `val`, or `test`). |
| `train_ids.txt`, `val_ids.txt`, `test_ids.txt` | Read one image ID per line in the same order as the matching CSV. |
| `label_mapping.json` | Look up the working label, official class ID, and breed name. |
| `split_info.json` | Confirm the procedure, producing NumPy version, split sizes, counts, checks, and file hashes. |

To create the files, run the notebook's **Export the fixed split manifests** cell once from the project checkout. `src.split_preparation.prepare_splits` also accepts the raw root and manifest directory in ordinary Python. It reuses byte-identical outputs and stops before writing if it finds a conflict. For a completed export, keep all eight files together.

## Read data without the notebook

Run from the project root with the project's NumPy and Pillow available. Set `PET_RAW_ROOT` to the local extracted dataset root containing `images/` and `annotations/`. For an in-project dataset, that root is `data/raw/`. The manifest paths remain relative to that root on each teammate's machine.

~~~python
import csv
import os
from pathlib import Path

from src.data_loader import load_rgb

raw_root = Path(os.environ["PET_RAW_ROOT"]).expanduser()
manifest = Path("data/manifests/train.csv")

with manifest.open(encoding="utf-8", newline="") as stream:
    record = next(csv.DictReader(stream))

rgb = load_rgb(raw_root / record["image_relative_path"])
label = int(record["label"])
print(record["image_id"], rgb.shape, label)
~~~

## Milestone coordination

Original photographs vary in width, height, and aspect ratio. Descriptor pipelines will apply their own explicitly documented preparation. Final resizing and preprocessing choices are still open. Save feature matrices in manifest row order and include matching image IDs and labels so members can verify alignment.

Use TRAIN for fitting and VAL for development decisions. Reserve TEST for final evaluation and keep it out of descriptor, preprocessing, and tuning decisions. Member 5 owns classifier implementation.

## What to share on GitHub

Share `src/`, `tests/`, `data/manifests/` with all eight generated files, this handover, `.gitignore`, `.gitattributes`, and the compact verification and inventory summaries linked above. The detailed per-image evidence and source archives stay local. Each teammate supplies their own extracted dataset root. Run the small fixture tests from the checkout with `python -B -m unittest discover -s tests -v`.
