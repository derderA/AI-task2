from __future__ import annotations

import unittest

from sam_experiment.deepseek_config_service import DeepSeekConfigService
from tests.test_support import TempProject


class DeepSeekConfigServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.project = TempProject()
        self.settings = self.project.settings()
        self.service = DeepSeekConfigService(self.settings)

    def tearDown(self) -> None:
        self.project.cleanup()

    def test_write_and_read_file_config(self) -> None:
        self.service.write_file_config("sk-test-12345678", "https://api.deepseek.com", "deepseek-chat")
        payload = self.service.read_file_config()
        self.assertEqual(payload["api_key"], "sk-test-12345678")
        self.assertEqual(payload["model"], "deepseek-chat")

    def test_session_credentials_override_file(self) -> None:
        self.service.write_file_config("sk-file-11112222", "https://api.deepseek.com", "deepseek-chat")
        resolved = self.service.resolve_credentials(
            {"api_key": "sk-session-99990000", "base_url": "https://custom.example.com", "model": "deepseek-reasoner"}
        )
        self.assertEqual(resolved["source"], "session")
        self.assertEqual(resolved["api_key"], "sk-session-99990000")
        self.assertEqual(resolved["model"], "deepseek-reasoner")


if __name__ == "__main__":
    unittest.main()
