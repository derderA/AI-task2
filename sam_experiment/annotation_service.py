from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import Settings
from .storage import Storage


@dataclass
class AnnotationArtifacts:
    annotation_path: Path
    instance_map_path: Path
    overlay_path: Path
    crop_dir: Path


class AnnotationService:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage
        self.settings.annotation_dir.mkdir(parents=True, exist_ok=True)
        self.settings.export_dir.mkdir(parents=True, exist_ok=True)

    def build_artifacts(self, image_path: Path, run_name: Optional[str] = None) -> AnnotationArtifacts:
        stem = run_name or image_path.stem
        annotation_path = self.settings.annotation_dir / f"{stem}.json"
        instance_map_path = self.settings.annotation_dir / f"{stem}_instance_map.png"
        overlay_path = self.settings.annotation_dir / f"{stem}_overlay.png"
        crop_dir = self.settings.export_dir / stem
        crop_dir.mkdir(parents=True, exist_ok=True)
        return AnnotationArtifacts(
            annotation_path=annotation_path,
            instance_map_path=instance_map_path,
            overlay_path=overlay_path,
            crop_dir=crop_dir,
        )

    def save_instance_map(self, path: Path, instance_map: np.ndarray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), instance_map.astype(np.uint16)):
            raise RuntimeError(f"无法写入实例图: {path}")
        return path

    def load_instance_map(self, path) -> np.ndarray:
        file_path = self.storage.from_relpath(path)
        instance_map = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if instance_map is None:
            raise FileNotFoundError(f"无法读取实例图: {file_path}")
        if instance_map.ndim == 3:
            instance_map = instance_map[:, :, 0]
        return instance_map.astype(np.uint16)

    def save_overlay(self, path: Path, overlay: np.ndarray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), overlay):
            raise RuntimeError(f"无法写入叠加图: {path}")
        return path

    def relative_payload(
        self,
        image_path: Path,
        artifacts: AnnotationArtifacts,
        instance_map: np.ndarray,
        categories: Optional[dict[int, str]] = None,
        source: str = "sam_auto",
    ) -> dict:
        categories = categories or {}
        instances = self._build_instances(instance_map, categories)
        class_names = sorted({item["category"] for item in instances if item["category"]})
        if self.settings.default_category not in class_names:
            class_names.insert(0, self.settings.default_category)
        return {
            "version": "1.0",
            "created_at": self.now(),
            "updated_at": self.now(),
            "source": source,
            "annotation_path": self.storage.to_relpath(artifacts.annotation_path),
            "image_path": self.storage.to_relpath(image_path),
            "instance_map_path": self.storage.to_relpath(artifacts.instance_map_path),
            "overlay_path": self.storage.to_relpath(artifacts.overlay_path),
            "crop_dir": self.storage.to_relpath(artifacts.crop_dir),
            "report_path": "",
            "deepseek_report_path": "",
            "video_report_path": "",
            "image_size": [int(instance_map.shape[1]), int(instance_map.shape[0])],
            "classes": class_names,
            "instances": instances,
        }

    def load_annotation(self, annotation_path) -> dict:
        return self.storage.read_json(annotation_path)

    def save_annotation(self, annotation_path, payload: dict) -> Path:
        payload["updated_at"] = self.now()
        return self.storage.write_json(annotation_path, payload)

    def update_category(self, annotation_path, instance_id: int, category: str) -> dict:
        payload = self.load_annotation(annotation_path)
        normalized = category.strip() or self.settings.default_category
        found = False
        for instance in payload["instances"]:
            if int(instance["id"]) == int(instance_id):
                instance["category"] = normalized
                found = True
                break
        if not found:
            raise KeyError(f"未找到实例 {instance_id}")
        if normalized not in payload["classes"]:
            payload["classes"].append(normalized)
            payload["classes"] = sorted(set(payload["classes"]))
        self.save_annotation(annotation_path, payload)
        return payload

    def rebuild_instances(self, payload: dict, instance_map: np.ndarray) -> dict:
        category_map = {int(item["id"]): item.get("category", self.settings.default_category) for item in payload["instances"]}
        payload["instances"] = self._build_instances(instance_map, category_map)
        payload["classes"] = sorted(
            set([self.settings.default_category] + [item["category"] for item in payload["instances"]])
        )
        return payload

    def merge_instances(
        self,
        annotation_path,
        instance_ids: list[int],
        merged_category: Optional[str] = None,
    ) -> dict:
        if len(instance_ids) < 2:
            raise ValueError("至少选择两个实例才能合并。")
        payload = self.load_annotation(annotation_path)
        instance_map = self.load_instance_map(payload["instance_map_path"])
        keep_id = min(instance_ids)
        union_mask = np.isin(instance_map, instance_ids)
        instance_map[np.isin(instance_map, instance_ids)] = 0
        instance_map[union_mask] = keep_id
        self._reindex_inplace(instance_map)
        payload = self.rebuild_instances(payload, instance_map)
        if merged_category:
            keep_new = self._find_new_instance_id(payload, keep_id)
            for instance in payload["instances"]:
                if int(instance["id"]) == int(keep_new):
                    instance["category"] = merged_category
                    break
            if merged_category not in payload["classes"]:
                payload["classes"].append(merged_category)
        self.save_instance_map(self.storage.from_relpath(payload["instance_map_path"]), instance_map)
        self.save_annotation(annotation_path, payload)
        return payload

    def add_mask_instance(self, annotation_path, binary_mask: np.ndarray, category: str) -> dict:
        payload = self.load_annotation(annotation_path)
        instance_map = self.load_instance_map(payload["instance_map_path"])
        if binary_mask.shape != instance_map.shape:
            raise ValueError("新实例掩码尺寸与实例图不一致。")
        next_id = int(instance_map.max()) + 1
        instance_map[binary_mask.astype(bool)] = next_id
        self.save_instance_map(self.storage.from_relpath(payload["instance_map_path"]), instance_map)
        payload = self.rebuild_instances(payload, instance_map)
        for instance in payload["instances"]:
            if int(instance["id"]) == int(next_id):
                instance["category"] = category
                break
        if category not in payload["classes"]:
            payload["classes"].append(category)
        self.save_annotation(annotation_path, payload)
        return payload

    def replace_instance_with_masks(
        self,
        annotation_path,
        instance_id: int,
        masks: list[np.ndarray],
        category: Optional[str] = None,
    ) -> dict:
        payload = self.load_annotation(annotation_path)
        instance_map = self.load_instance_map(payload["instance_map_path"])
        target_mask = instance_map == int(instance_id)
        if not np.any(target_mask):
            raise KeyError(f"未找到实例 {instance_id}")
        instance_map[target_mask] = 0
        next_id = int(instance_map.max()) + 1
        for mask in masks:
            mask = mask.astype(bool) & target_mask
            if not np.any(mask):
                continue
            instance_map[mask] = next_id
            next_id += 1
        self._reindex_inplace(instance_map)
        self.save_instance_map(self.storage.from_relpath(payload["instance_map_path"]), instance_map)
        payload = self.rebuild_instances(payload, instance_map)
        if category:
            for instance in payload["instances"]:
                if instance["category"] == self.settings.default_category:
                    instance["category"] = category
            if category not in payload["classes"]:
                payload["classes"].append(category)
        self.save_annotation(annotation_path, payload)
        return payload

    def export_crops(self, image: np.ndarray, instance_map: np.ndarray, crop_dir: Path) -> list[dict]:
        crop_dir.mkdir(parents=True, exist_ok=True)
        exported = []
        for instance in self._build_instances(instance_map):
            x, y, w, h = instance["bbox"]
            mask = (instance_map == instance["id"]).astype(np.uint8)[..., None]
            crop = image[y : y + h, x : x + w] * mask[y : y + h, x : x + w]
            file_path = crop_dir / f"instance_{instance['id']:04d}.png"
            cv2.imwrite(str(file_path), crop)
            exported.append({"id": instance["id"], "path": self.storage.to_relpath(file_path)})
        return exported

    def now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _build_instances(self, instance_map: np.ndarray, categories: Optional[dict[int, str]] = None) -> list[dict]:
        categories = categories or {}
        instances = []
        for instance_id in sorted(int(v) for v in np.unique(instance_map) if int(v) != 0):
            ys, xs = np.where(instance_map == instance_id)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            instances.append(
                {
                    "id": instance_id,
                    "category": categories.get(instance_id, self.settings.default_category),
                    "area": int(len(xs)),
                    "bbox": [x_min, y_min, int(x_max - x_min + 1), int(y_max - y_min + 1)],
                    "center": [int(xs.mean()), int(ys.mean())],
                }
            )
        return instances

    def _reindex_inplace(self, instance_map: np.ndarray) -> None:
        unique_ids = [int(v) for v in np.unique(instance_map) if int(v) != 0]
        for new_id, old_id in enumerate(unique_ids, start=1):
            if new_id != old_id:
                instance_map[instance_map == old_id] = new_id

    def _find_new_instance_id(self, payload: dict, desired_id: int) -> int:
        available = [int(item["id"]) for item in payload["instances"]]
        return desired_id if desired_id in available else min(available)
