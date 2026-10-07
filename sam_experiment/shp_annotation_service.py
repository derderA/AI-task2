from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

from .storage import Storage


class ShpAnnotationService:
    def __init__(self, preview_dir: Path, storage: Storage) -> None:
        self.preview_dir = preview_dir
        self.storage = storage
        self.preview_dir.mkdir(parents=True, exist_ok=True)

    def build_preview(self, shp_path: Path) -> dict:
        shapefile = self._load_pyshp()
        reader = shapefile.Reader(str(shp_path))
        fields = [field[0] for field in reader.fields[1:]]

        bounds = self._normalize_bounds(list(reader.bbox) if reader.bbox else None)
        shape_records = list(reader.iterShapeRecords())
        shape_counter: Counter[str] = Counter()
        svg_elements: list[dict] = []
        sample_records: list[dict] = []

        for index, shape_record in enumerate(shape_records):
            shape = shape_record.shape
            shape_name = self._shape_type_name(shape.shapeType)
            shape_counter[shape_name] += 1
            record_values = self._record_to_dict(fields, shape_record.record)
            centroid = self._centroid(shape.points)
            bbox = self._normalize_bounds(list(shape.bbox) if getattr(shape, "bbox", None) else None)
            sample_records.append(
                {
                    "index": index,
                    "shape_type": shape_name,
                    "point_count": len(shape.points),
                    "centroid": [round(centroid[0], 3), round(centroid[1], 3)] if centroid else [],
                    "bbox": [round(value, 3) for value in bbox] if bbox else [],
                    "attributes": record_values,
                }
            )
            if len(svg_elements) < 120:
                svg_elements.extend(self._shape_to_svg_elements(shape, index, bounds))

        preview_payload = {
            "source_name": shp_path.name,
            "source_path": self.storage.to_relpath(shp_path),
            "bundle_dir": self.storage.to_relpath(shp_path.parent),
            "record_count": len(shape_records),
            "shape_counts": dict(shape_counter),
            "fields": fields,
            "bounds": [round(value, 3) for value in bounds] if bounds else [],
            "sample_records": sample_records[:20],
            "svg": {
                "width": 960,
                "height": 540,
                "elements": svg_elements,
            },
        }
        preview_path = self.preview_dir / ("%s.preview.json" % shp_path.stem)
        preview_payload["preview_path"] = self.storage.to_relpath(preview_path)
        self.storage.write_json(preview_path, preview_payload)
        return preview_payload

    def load_preview(self, preview_path) -> dict:
        payload = self.storage.read_json(preview_path)
        payload.setdefault("preview_path", self.storage.to_relpath(self.storage.from_relpath(preview_path)))
        return payload

    def list_recent_previews(self, limit: int = 8) -> list[str]:
        files = sorted(self.preview_dir.glob("*.preview.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]
        return [self.storage.to_relpath(path) for path in files]

    def _load_pyshp(self):
        try:
            import shapefile  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("未安装 pyshp，请先在环境中执行 `pip install pyshp`。") from exc
        return shapefile

    def _record_to_dict(self, fields: list[str], record) -> dict:
        result = {}
        values = list(record) if record is not None else []
        for index, name in enumerate(fields[:8]):
            if index >= len(values):
                break
            result[name] = self._stringify(values[index])
        return result

    def _stringify(self, value):
        if value is None:
            return ""
        if isinstance(value, float):
            return round(value, 4)
        return str(value)

    def _normalize_bounds(self, bbox: Optional[list[float]]) -> Optional[list[float]]:
        if not bbox:
            return None
        x_min, y_min, x_max, y_max = bbox
        if math.isclose(x_min, x_max):
            x_min -= 0.5
            x_max += 0.5
        if math.isclose(y_min, y_max):
            y_min -= 0.5
            y_max += 0.5
        return [x_min, y_min, x_max, y_max]

    def _shape_type_name(self, shape_type: int) -> str:
        mapping = {
            0: "Null",
            1: "Point",
            3: "Polyline",
            5: "Polygon",
            8: "MultiPoint",
            11: "PointZ",
            13: "PolylineZ",
            15: "PolygonZ",
            18: "MultiPointZ",
            21: "PointM",
            23: "PolylineM",
            25: "PolygonM",
            28: "MultiPointM",
            31: "MultiPatch",
        }
        return mapping.get(shape_type, "ShapeType%s" % shape_type)

    def _centroid(self, points: list[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not points:
            return None
        x_total = sum(point[0] for point in points)
        y_total = sum(point[1] for point in points)
        return x_total / len(points), y_total / len(points)

    def _shape_to_svg_elements(self, shape, index: int, bounds: Optional[list[float]]) -> list[dict]:
        if not bounds or not shape.points:
            return []
        shape_name = self._shape_type_name(shape.shapeType)
        points = [self._project_point(point[0], point[1], bounds) for point in shape.points]
        if shape_name.startswith("Point"):
            x, y = points[0]
            return [{"kind": "circle", "record_index": index, "cx": round(x, 2), "cy": round(y, 2), "r": 4}]
        if shape_name.startswith("MultiPoint"):
            return [
                {"kind": "circle", "record_index": index, "cx": round(x, 2), "cy": round(y, 2), "r": 3}
                for x, y in points[:40]
            ]

        part_indices = list(getattr(shape, "parts", []) or [0])
        part_indices.append(len(points))
        elements = []
        for start, end in zip(part_indices[:-1], part_indices[1:]):
            path_points = points[start:end]
            if not path_points:
                continue
            commands = ["M %.2f %.2f" % path_points[0]]
            commands.extend("L %.2f %.2f" % point for point in path_points[1:])
            if shape_name.startswith("Polygon"):
                commands.append("Z")
            elements.append(
                {
                    "kind": "path",
                    "record_index": index,
                    "d": " ".join(commands),
                    "shape_type": shape_name,
                }
            )
        return elements

    def _project_point(self, x: float, y: float, bounds: list[float]) -> Tuple[float, float]:
        x_min, y_min, x_max, y_max = bounds
        width = 960.0
        height = 540.0
        padding = 24.0
        x_ratio = (x - x_min) / (x_max - x_min)
        y_ratio = (y - y_min) / (y_max - y_min)
        svg_x = padding + x_ratio * (width - padding * 2)
        svg_y = height - padding - y_ratio * (height - padding * 2)
        return svg_x, svg_y
