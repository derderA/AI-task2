from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from sam_experiment.annotation_service import AnnotationService
from sam_experiment.storage import Storage
from tests.test_support import TempProject


class AnnotationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.project = TempProject()
        self.settings = self.project.settings()
        self.storage = Storage(self.settings.base_dir)
        self.service = AnnotationService(self.settings, self.storage)

        self.image_path = self.settings.base_dir / "sample.png"
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[1:4, 1:4] = (255, 255, 255)
        image[4:7, 4:7] = (127, 127, 127)
        cv2.imwrite(str(self.image_path), image)

        self.instance_map = np.zeros((8, 8), dtype=np.uint16)
        self.instance_map[1:4, 1:4] = 1
        self.instance_map[4:7, 4:7] = 2
        self.artifacts = self.service.build_artifacts(self.image_path, "unit_case")
        self.service.save_instance_map(self.artifacts.instance_map_path, self.instance_map)
        payload = self.service.relative_payload(
            self.image_path,
            self.artifacts,
            self.instance_map,
            categories={1: "plane", 2: "ship"},
        )
        self.service.save_annotation(self.artifacts.annotation_path, payload)

    def tearDown(self) -> None:
        self.project.cleanup()

    def test_update_and_merge_instances(self) -> None:
        annotation_path = self.storage.to_relpath(self.artifacts.annotation_path)
        updated = self.service.update_category(annotation_path, 1, "airport")
        self.assertEqual(updated["instances"][0]["category"], "airport")

        merged = self.service.merge_instances(annotation_path, [1, 2], "merged_target")
        self.assertEqual(len(merged["instances"]), 1)
        self.assertEqual(merged["instances"][0]["category"], "merged_target")

    def test_add_mask_instance(self) -> None:
        annotation_path = self.storage.to_relpath(self.artifacts.annotation_path)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:2, 5:7] = 1
        updated = self.service.add_mask_instance(annotation_path, mask, "harbor")
        self.assertEqual(len(updated["instances"]), 3)
        self.assertIn("harbor", updated["classes"])


if __name__ == "__main__":
    unittest.main()
