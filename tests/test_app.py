from __future__ import annotations

import unittest

from app import app


class FlaskAppTest(unittest.TestCase):
    def setUp(self) -> None:
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_index_page(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertIn("SAM 综合实验平台", text)
        self.assertIn("DeepSeek 配置", text)
        self.assertIn("Shp 标注预览", text)
        self.assertIn('type="file"', text)
        self.assertIn("image_file", text)
        self.assertIn("video_file", text)
        self.assertIn("shp_files", text)


if __name__ == "__main__":
    unittest.main()
