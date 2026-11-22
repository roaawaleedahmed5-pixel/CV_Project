"""Prepare and export fixed manifests from the official Oxford-IIIT Pet lists."""

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


VAL_RATIO = 0.2
SEED = 42
SPLIT_NAMES = ("train", "val", "test")
MANIFEST_FIELDS = (
    "image_id", "image_relative_path", "trimap_relative_path", "label",
    "official_class_id", "breed_name", "species_id", "official_species_breed_id",
    "official_split", "split",
)
IMAGE_ID = re.compile(r"([A-Za-z][A-Za-z0-9_]*)_[0-9]+")
SPLIT_ALGORITHM = (
    "Sort official class IDs numerically and sort each class's trainval records by "
    "image_id using case-sensitive Python string order. Use one numpy.random.default_rng(42) "
    "and, for each class in that order, select floor(0.2 * class_count + 0.5) VAL "
    "indices with rng.choice(class_count, size=val_count, replace=False). Put the remaining "
    "records in TRAIN. Copy official TEST membership. Sort each final manifest by image_id."
)


@dataclass(frozen=True)
class OfficialRecord:
    image_id: str
    official_class_id: int
    breed_name: str
    species_id: int
    official_species_breed_id: int
    official_split: str

    def manifest_row(self, split: str) -> dict:
        return {
            "image_id": self.image_id,
            "image_relative_path": f"images/{self.image_id}.jpg",
            "trimap_relative_path": f"annotations/trimaps/{self.image_id}.png",
            "label": self.official_class_id - 1,
            "official_class_id": self.official_class_id,
            "breed_name": self.breed_name,
            "species_id": self.species_id,
            "official_species_breed_id": self.official_species_breed_id,
            "official_split": self.official_split,
            "split": split,
        }


@dataclass(frozen=True)
class SplitResult:
    manifests: dict[str, list[dict]]
    label_mapping: list[dict]
    split_info: dict
    export_status: str


def read_official_list(path: Path, content: bytes) -> list[OfficialRecord]:
    records = []
    seen = set()
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            image_id, class_id, species_id, breed_id = text.split()
            match = IMAGE_ID.fullmatch(image_id)
            if match is None:
                raise ValueError(f"invalid image ID {image_id!r}")
            class_id, species_id, breed_id = map(int, (class_id, species_id, breed_id))
            # The publisher README reverses its species-specific limits. Preserve
            # that field as recorded and check the shared numeric bounds.
            if not (1 <= class_id <= 37 and species_id in (1, 2) and 1 <= breed_id <= 25):
                raise ValueError("official class, species, or species breed ID is outside its bounds")
            if image_id in seen:
                raise ValueError(f"duplicate image ID {image_id!r}")
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        seen.add(image_id)
        records.append(OfficialRecord(image_id, class_id, match[1], species_id, breed_id, path.stem))
    if not records:
        raise ValueError(f"No official image records in {path}")
    return records


def assign_train_val(records: list[OfficialRecord]) -> tuple[list[OfficialRecord], list[OfficialRecord]]:
    by_class = defaultdict(list)
    for record in records:
        by_class[record.official_class_id].append(record)

    rng = np.random.default_rng(SEED)
    train, val = [], []
    for class_id in sorted(by_class):
        class_records = sorted(by_class[class_id], key=lambda record: record.image_id)
        val_count = math.floor(VAL_RATIO * len(class_records) + 0.5)
        val_indices = set(rng.choice(len(class_records), size=val_count, replace=False))
        for index, record in enumerate(class_records):
            (val if index in val_indices else train).append(record)
    return train, val


def validate_manifests(manifests: dict, trainval: list[OfficialRecord], test: list[OfficialRecord]) -> tuple[dict, list]:
    ids = {
        split: [row["image_id"] for row in rows]
        for split, rows in manifests.items()
    }
    id_sets = {split: set(split_ids) for split, split_ids in ids.items()}
    rows = [row for split_rows in manifests.values() for row in split_rows]
    official_by_id = {record.image_id: record for record in trainval + test}
    pairs = {(record.official_class_id, record.breed_name) for record in trainval + test}
    classes = {class_id for class_id, _ in pairs}
    breeds = {breed for _, breed in pairs}
    checks = {
        "official_trainval_coverage_once": Counter(ids["train"] + ids["val"]) == Counter(
            record.image_id for record in trainval
        ),
        "train_val_test_disjoint": (
            id_sets["train"].isdisjoint(id_sets["val"])
            and (id_sets["train"] | id_sets["val"]).isdisjoint(id_sets["test"])
        ),
        "official_test_membership_exact": Counter(ids["test"]) == Counter(record.image_id for record in test),
        "every_breed_in_train_and_val": all(
            {(row["official_class_id"], row["breed_name"]) for row in manifests[split]} == pairs
            for split in ("train", "val")
        ),
        "labels_equal_official_class_id_minus_one": all(
            row["label"] == row["official_class_id"] - 1 for row in rows
        ),
        "breed_and_label_mapping_consistent": len(pairs) == len(classes) == len(breeds),
        "official_records_only": all(
            row["image_id"] in official_by_id
            and not row["image_id"].startswith("._")
            and row == official_by_id[row["image_id"]].manifest_row(split)
            for split, split_rows in manifests.items() for row in split_rows
        ),
    }
    failures = [check for check, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"Split validation failed: {', '.join(failures)}")

    mapping = [
        {"label": class_id - 1, "official_class_id": class_id, "breed_name": breed}
        for class_id, breed in sorted(pairs)
    ]
    return checks, mapping


def json_bytes(value: dict | list) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def manifest_payloads(manifests: dict, mapping: list) -> dict[str, bytes]:
    payloads = {}
    for split, rows in manifests.items():
        with io.StringIO(newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            payloads[f"{split}.csv"] = stream.getvalue().encode("utf-8")
        payloads[f"{split}_ids.txt"] = "".join(row["image_id"] + "\n" for row in rows).encode("utf-8")
    payloads["label_mapping.json"] = json_bytes(mapping)
    return payloads


def export_payloads(directory: Path, payloads: dict[str, bytes]) -> str:
    """Reuse identical artifacts, preflight conflicts, and publish metadata last."""
    existing = set()
    conflicts = []
    for name, expected in payloads.items():
        target = directory / name
        if target.is_symlink():
            conflicts.append(name)
        elif target.exists():
            if target.is_file() and target.read_bytes() == expected:
                existing.add(name)
            else:
                conflicts.append(name)
    if conflicts:
        raise FileExistsError(
            f"Existing split files differ in {directory}: {', '.join(conflicts)}. "
            "Use the existing authoritative set or export to a different directory."
        )
    if len(existing) == len(payloads):
        return "reused"

    directory.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        if name in existing:
            continue
        target = directory / name
        stream = target.open("xb")
        try:
            with stream:
                stream.write(content)
        except OSError:
            # Preserve files from previous runs and allow a retry after a failed write.
            target.unlink(missing_ok=True)
            raise
    return "completed" if existing else "created"


def prepare_splits(raw_root: str | Path, manifest_dir: str | Path) -> SplitResult:
    """Validate official records, then export or reuse the authoritative split files."""
    annotation_paths = [Path(raw_root) / "annotations" / f"{split}.txt" for split in ("trainval", "test")]
    sources = [path.read_bytes() for path in annotation_paths]
    trainval, test = (read_official_list(path, content) for path, content in zip(annotation_paths, sources))
    train, val = assign_train_val(trainval)
    manifests = {
        split: [record.manifest_row(split) for record in sorted(records, key=lambda row: row.image_id)]
        for split, records in zip(SPLIT_NAMES, (train, val, test))
    }
    checks, mapping = validate_manifests(manifests, trainval, test)
    payloads = manifest_payloads(manifests, mapping)
    counts_by_split = {
        split: Counter(row["official_class_id"] for row in rows)
        for split, rows in manifests.items()
    }
    info = {
        "trainval_ratios": {"train": 1 - VAL_RATIO, "val": VAL_RATIO},
        "seed": SEED,
        "algorithm": SPLIT_ALGORITHM,
        "numpy_version": np.__version__,
        "official_test_membership_preserved": checks["official_test_membership_exact"],
        "split_sizes": {split: len(rows) for split, rows in manifests.items()},
        "per_breed_counts": [
            {
                **entry,
                **{split: counts[entry["official_class_id"]] for split, counts in counts_by_split.items()},
            }
            for entry in mapping
        ],
        "checks": checks,
        "source_annotation_sha256": {
            f"annotations/{path.name}": hashlib.sha256(content).hexdigest()
            for path, content in zip(annotation_paths, sources)
        },
        "exported_file_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()
        },
    }
    payloads["split_info.json"] = json_bytes(info)
    status = export_payloads(Path(manifest_dir), payloads)
    return SplitResult(manifests, mapping, info, status)
