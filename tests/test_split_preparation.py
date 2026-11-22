import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.split_preparation import MANIFEST_FIELDS, prepare_splits


class SplitPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.output = self.root / "manifests"
        self.annotations = self.raw / "annotations"
        self.annotations.mkdir(parents=True)
        self.trainval = [f"zebra_hound_{i} 2 2 6" for i in range(1, 9)]
        self.trainval += [f"Able_cat_{i} 12 1 3" for i in range(1, 4)]
        self.test = ["zebra_hound_19 2 2 6", "Able_cat_10 12 1 3"]
        self.write_list("trainval", self.trainval)
        self.write_list("test", self.test)
        images = self.raw / "images"
        images.mkdir()
        (images / "unmatched_1.jpg").touch()
        (images / "._zebra_hound_1.jpg").touch()

    def write_list(self, name, rows):
        (self.annotations / f"{name}.txt").write_bytes(("\n".join(rows) + "\n").encode("utf-8"))

    def test_reproducible_assignment_disjoint_coverage_and_preserved_test(self):
        result = prepare_splits(self.raw, self.output)
        self.write_list("trainval", list(reversed(self.trainval)))
        self.write_list("test", list(reversed(self.test)))
        rerun = prepare_splits(self.raw, self.root / "second")
        self.assertEqual(result.manifests, rerun.manifests)

        ids = {split: [row["image_id"] for row in rows] for split, rows in result.manifests.items()}
        self.assertCountEqual(ids["train"] + ids["val"], [row.split()[0] for row in self.trainval])
        self.assertCountEqual(ids["test"], [row.split()[0] for row in self.test])
        self.assertEqual(len(set(sum(ids.values(), []))), sum(map(len, ids.values())))
        trainval_ids = set(ids["train"] + ids["val"])
        self.assertTrue(trainval_ids.isdisjoint(ids["test"]))
        self.assertTrue(all(result.split_info["checks"].values()))

    def test_stratification_and_numeric_labels(self):
        result = prepare_splits(self.raw, self.output)
        self.assertEqual(
            result.label_mapping,
            [
                {"label": 1, "official_class_id": 2, "breed_name": "zebra_hound"},
                {"label": 11, "official_class_id": 12, "breed_name": "Able_cat"},
            ],
        )
        self.assertEqual(
            {row["breed_name"]: (row["train"], row["val"], row["test"])
             for row in result.split_info["per_breed_counts"]},
            {"zebra_hound": (6, 2, 1), "Able_cat": (2, 1, 1)},
        )
        for split, rows in result.manifests.items():
            for row in rows:
                image_id = row["image_id"]
                self.assertEqual(row["label"], row["official_class_id"] - 1)
                self.assertEqual(row["split"], split)
                self.assertEqual(row["official_split"], "test" if split == "test" else "trainval")
                self.assertEqual(row["image_relative_path"], f"images/{image_id}.jpg")
                self.assertEqual(row["trimap_relative_path"], f"annotations/trimaps/{image_id}.png")

    def test_manifest_order_hashes_and_unchanged_reuse(self):
        first = prepare_splits(self.raw, self.output)
        self.assertEqual(first.export_status, "created")
        for split in ("train", "val", "test"):
            with (self.output / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, list(MANIFEST_FIELDS))
                csv_ids = [row["image_id"] for row in reader]
            text_ids = (self.output / f"{split}_ids.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(csv_ids, sorted(csv_ids))
            self.assertEqual(csv_ids, text_ids)

        info = json.loads((self.output / "split_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["numpy_version"], np.__version__)
        self.assertTrue(info["official_test_membership_preserved"])
        for name, digest in info["source_annotation_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((self.raw / name).read_bytes()).hexdigest())
        for name, digest in info["exported_file_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((self.output / name).read_bytes()).hexdigest())
        self.assertEqual(prepare_splits(self.raw, self.output).export_status, "reused")

    def test_completes_a_matching_partial_export(self):
        prepare_splits(self.raw, self.output)
        original = self.output / "train.csv"
        original_mtime = original.stat().st_mtime_ns
        (self.output / "test_ids.txt").unlink()
        (self.output / "split_info.json").unlink()
        self.assertEqual(prepare_splits(self.raw, self.output).export_status, "completed")
        self.assertEqual(original.stat().st_mtime_ns, original_mtime)
        self.assertEqual(prepare_splits(self.raw, self.output).export_status, "reused")

    def test_conflict_stops_before_writing_other_files(self):
        self.output.mkdir()
        existing = self.output / "val.csv"
        existing.write_bytes(b"fixed version from an earlier export\n")
        with self.assertRaisesRegex(FileExistsError, "val.csv"):
            prepare_splits(self.raw, self.output)
        self.assertEqual(existing.read_bytes(), b"fixed version from an earlier export\n")
        self.assertEqual(list(self.output.iterdir()), [existing])

    def test_invalid_official_membership_stops_export(self):
        self.write_list("test", self.test + [self.trainval[0]])
        with self.assertRaisesRegex(ValueError, "train_val_test_disjoint"):
            prepare_splits(self.raw, self.output)
        self.assertFalse(self.output.exists())
