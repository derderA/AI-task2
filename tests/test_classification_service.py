from __future__ import annotations

import unittest

import cv2
import numpy as np

from sam_experiment.classification_service import ClassificationService
from sam_experiment.deepseek_config_service import DeepSeekConfigService
from sam_experiment.storage import Storage
from tests.test_support import TempProject


class ClassificationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.project = TempProject()
        self.settings = self.project.settings()
        self.storage = Storage(self.settings.base_dir)
        self.config_service = DeepSeekConfigService(self.settings)
        self.service = ClassificationService(self.settings, self.storage, self.config_service)

    def tearDown(self) -> None:
        self.project.cleanup()

    def test_rule_classification_builds_sidebar_metadata(self) -> None:
        instance_map_path = self.settings.annotation_dir / "demo_instance_map.png"
        overlay_path = self.settings.annotation_dir / "demo_overlay.png"
        image_stub = self.settings.upload_image_dir / "demo.png"

        instance_map = np.zeros((200, 240), dtype=np.uint16)
        instance_map[20:80, 30:90] = 1
        instance_map[110:180, 120:220] = 2
        cv2.imwrite(str(instance_map_path), instance_map)
        cv2.imwrite(str(overlay_path), np.zeros((200, 240, 3), dtype=np.uint8))
        cv2.imwrite(str(image_stub), np.zeros((200, 240, 3), dtype=np.uint8))

        annotation_path = self.settings.annotation_dir / "demo.json"
        self.storage.write_json(
            annotation_path,
            {
                "annotation_path": self.storage.to_relpath(annotation_path),
                "image_path": self.storage.to_relpath(image_stub),
                "instance_map_path": self.storage.to_relpath(instance_map_path),
                "overlay_path": self.storage.to_relpath(overlay_path),
                "crop_dir": self.storage.to_relpath(self.settings.export_dir / "demo"),
                "classes": ["未分类"],
                "instances": [
                    {"id": 1, "category": "未分类", "area": 3600, "bbox": [30, 20, 60, 60], "center": [60, 50]},
                    {"id": 2, "category": "未分类", "area": 7000, "bbox": [120, 110, 100, 70], "center": [170, 145]},
                ],
            },
        )

        payload = self.service.classify_annotation(annotation_path, provider="rule")
        decorated = self.service.decorate_annotation(payload)

        self.assertIn("classification_summary", payload)
        self.assertTrue(payload["classification_summary"]["clusters"])
        self.assertNotEqual(payload["instances"][0]["category"], "未分类")
        self.assertEqual(len(decorated["category_sidebar"]), len(payload["classes"]))
        self.assertIn("bbox_style", decorated["instances"][0])


if __name__ == "__main__":
    unittest.main()
