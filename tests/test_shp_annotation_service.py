from __future__ import annotations

import unittest

import shapefile

from sam_experiment.shp_annotation_service import ShpAnnotationService
from sam_experiment.storage import Storage
from tests.test_support import TempProject


class ShpAnnotationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.project = TempProject()
        self.settings = self.project.settings()
        self.storage = Storage(self.settings.base_dir)
        self.service = ShpAnnotationService(self.settings.shp_preview_dir, self.storage)

    def tearDown(self) -> None:
        self.project.cleanup()

    def test_build_preview_from_point_shapefile(self) -> None:
        shp_base = self.settings.upload_shp_dir / "sample_points"
        writer = shapefile.Writer(str(shp_base), shapeType=shapefile.POINT)
        writer.field("name", "C")
        writer.point(120.1, 30.2)
        writer.record("A")
        writer.point(120.3, 30.4)
        writer.record("B")
        writer.close()

        preview = self.service.build_preview(shp_base.with_suffix(".shp"))

        self.assertEqual(preview["record_count"], 2)
        self.assertEqual(preview["shape_counts"].get("Point"), 2)
        self.assertEqual(preview["fields"], ["name"])
        self.assertEqual(len(preview["sample_records"]), 2)
        self.assertEqual(len(preview["svg"]["elements"]), 2)
        self.assertTrue(preview["preview_path"].endswith("sample_points.preview.json"))


if __name__ == "__main__":
    unittest.main()
