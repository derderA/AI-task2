from __future__ import annotations

import unittest

from sam_experiment.report_service import ReportService
from sam_experiment.storage import Storage
from tests.test_support import TempProject


class ReportServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.project = TempProject()
        self.settings = self.project.settings()
        self.storage = Storage(self.settings.base_dir)
        self.service = ReportService(self.settings, self.storage)

    def tearDown(self) -> None:
        self.project.cleanup()

    def test_generate_image_report(self) -> None:
        annotation = {
            "image_path": "dataset/demo.png",
            "instances": [
                {"id": 1, "category": "plane"},
                {"id": 2, "category": "plane"},
                {"id": 3, "category": "helicopter"},
                {"id": 4, "category": "bridge"},
            ],
        }
        summary, report_path = self.service.generate_image_report(annotation)
        self.assertEqual(summary["region_type"], "机场/起降保障区域")
        self.assertTrue(report_path.exists())
        self.assertIn("plane", summary["counts"])


if __name__ == "__main__":
    unittest.main()
