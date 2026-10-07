from __future__ import annotations

import os
import unittest

from sam_experiment.deepseek_config_service import DeepSeekConfigService
from sam_experiment.deepseek_report_service import DeepSeekReportService
from sam_experiment.report_service import ReportService
from sam_experiment.storage import Storage
from tests.test_support import TempProject


class DeepSeekReportServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.project = TempProject()
        self.settings = self.project.settings()
        self.storage = Storage(self.settings.base_dir)
        self.rule_service = ReportService(self.settings, self.storage)
        self.config_service = DeepSeekConfigService(self.settings)
        self.service = DeepSeekReportService(self.settings, self.storage, self.rule_service, self.config_service)

    def tearDown(self) -> None:
        self.project.cleanup()

    def test_build_prompt_contains_evidence(self) -> None:
        annotation = {
            "image_path": "dataset/demo.png",
            "instances": [
                {"id": 1, "category": "plane", "area": 100, "bbox": [1, 2, 3, 4], "center": [5, 6]},
                {"id": 2, "category": "ship", "area": 80, "bbox": [7, 8, 9, 10], "center": [11, 12]},
            ],
        }
        summary = self.rule_service.summarize_image_annotation(annotation)
        prompt = self.service.build_image_prompt(annotation, summary)
        self.assertIn("结构化摘要", prompt)
        self.assertIn("plane", prompt)
        self.assertIn("ship", prompt)

    def test_missing_api_key_raises(self) -> None:
        previous = os.environ.pop(self.settings.deepseek_api_key_env, None)
        try:
            with self.assertRaises(RuntimeError):
                self.service._request_markdown("test", {"api_key": "", "base_url": "", "model": ""})
        finally:
            if previous is not None:
                os.environ[self.settings.deepseek_api_key_env] = previous


if __name__ == "__main__":
    unittest.main()
