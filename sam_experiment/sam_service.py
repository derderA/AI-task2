from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from .annotation_service import AnnotationArtifacts, AnnotationService
from .classification_service import category_color_rgb
from .config import Settings
from .storage import Storage


class SamExperimentService:
    def __init__(self, settings: Settings, storage: Storage, annotation_service: AnnotationService) -> None:
        self.settings = settings
        self.storage = storage
        self.annotation_service = annotation_service
        self._sam_model = None
        self._predictor = None

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def auto_segment(
        self,
        image_path,
        run_name: Optional[str] = None,
        points_per_side: int = 24,
    ) -> dict:
        image_file = self.storage.from_relpath(image_path).resolve()
        image = self.read_image(image_file)
        masks = self._get_generator(points_per_side=points_per_side).generate(image)
        instance_map = self._masks_to_instance_map(masks, image.shape[:2])
        artifacts = self.annotation_service.build_artifacts(image_file, run_name=run_name)
        overlay = self.build_overlay(image, instance_map)
        self.annotation_service.save_instance_map(artifacts.instance_map_path, instance_map)
        self.annotation_service.save_overlay(artifacts.overlay_path, overlay)
        payload = self.annotation_service.relative_payload(image_file, artifacts, instance_map)
        payload["sam_settings"] = {
            "device": self.device,
            "model_type": self.settings.default_model_type,
            "points_per_side": points_per_side,
        }
        payload["crop_exports"] = self.annotation_service.export_crops(image, instance_map, artifacts.crop_dir)
        self.annotation_service.save_annotation(artifacts.annotation_path, payload)
        return payload

    def add_prompt_instance(
        self,
        annotation_path,
        x: int,
        y: int,
        category: str,
    ) -> dict:
        payload = self.annotation_service.load_annotation(annotation_path)
        image_path = self.storage.from_relpath(payload["image_path"])
        image = self.read_image(image_path)
        predictor = self._get_predictor()
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best_mask = masks[int(np.argmax(scores))]
        updated = self.annotation_service.add_mask_instance(annotation_path, best_mask, category)
        instance_map = self.annotation_service.load_instance_map(updated["instance_map_path"])
        overlay = self.build_overlay(image, instance_map, updated)
        self.annotation_service.save_overlay(self.storage.from_relpath(updated["overlay_path"]), overlay)
        crop_dir = self.storage.from_relpath(updated["crop_dir"])
        updated["crop_exports"] = self.annotation_service.export_crops(image, instance_map, crop_dir)
        self.annotation_service.save_annotation(annotation_path, updated)
        return updated

    def resegment_instance(self, annotation_path, instance_id: int) -> dict:
        payload = self.annotation_service.load_annotation(annotation_path)
        image_path = self.storage.from_relpath(payload["image_path"])
        image = self.read_image(image_path)
        instance_map = self.annotation_service.load_instance_map(payload["instance_map_path"])
        target_mask = instance_map == int(instance_id)
        if not np.any(target_mask):
            raise KeyError(f"未找到实例 {instance_id}")
        ys, xs = np.where(target_mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        crop = image[y_min : y_max + 1, x_min : x_max + 1]
        crop_masks = self._get_generator(points_per_side=16).generate(crop)
        child_masks = []
        for entry in crop_masks:
            mask = entry["segmentation"].astype(bool)
            full_mask = np.zeros_like(target_mask, dtype=bool)
            full_mask[y_min : y_max + 1, x_min : x_max + 1] = mask
            full_mask &= target_mask
            if np.any(full_mask):
                child_masks.append(full_mask)
        if len(child_masks) <= 1:
            raise ValueError("该实例在当前参数下未分裂出更多子实例，请尝试更复杂目标。")
        updated = self.annotation_service.replace_instance_with_masks(annotation_path, instance_id, child_masks)
        instance_map = self.annotation_service.load_instance_map(updated["instance_map_path"])
        overlay = self.build_overlay(image, instance_map, updated)
        self.annotation_service.save_overlay(self.storage.from_relpath(updated["overlay_path"]), overlay)
        crop_dir = self.storage.from_relpath(updated["crop_dir"])
        updated["crop_exports"] = self.annotation_service.export_crops(image, instance_map, crop_dir)
        self.annotation_service.save_annotation(annotation_path, updated)
        return updated

    def sync_annotation_artifacts(self, annotation_path) -> dict:
        payload = self.annotation_service.load_annotation(annotation_path)
        image = self.read_image(self.storage.from_relpath(payload["image_path"]))
        instance_map = self.annotation_service.load_instance_map(payload["instance_map_path"])
        overlay = self.build_overlay(image, instance_map, payload)
        self.annotation_service.save_overlay(self.storage.from_relpath(payload["overlay_path"]), overlay)
        crop_dir = self.storage.from_relpath(payload["crop_dir"])
        payload["crop_exports"] = self.annotation_service.export_crops(image, instance_map, crop_dir)
        self.annotation_service.save_annotation(annotation_path, payload)
        return payload

    def list_candidate_images(self, limit: int = 30) -> list[str]:
        files: list[str] = []
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        for root in self.settings.preferred_image_roots:
            if not root.exists():
                continue
            matched = []
            for pattern in patterns:
                matched.extend(root.glob(pattern))
            for image_file in sorted(set(matched), key=lambda item: item.name)[:limit]:
                files.append(self.storage.to_relpath(image_file))
            if files:
                break
        return files[:limit]

    def read_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def build_overlay(self, image_rgb: np.ndarray, instance_map: np.ndarray, payload: Optional[dict] = None) -> np.ndarray:
        overlay = image_rgb.copy()
        ids = [int(v) for v in np.unique(instance_map) if int(v) != 0]
        instances = {int(item["id"]): item for item in payload.get("instances", [])} if payload else {}
        for instance_id in ids:
            instance = instances.get(instance_id)
            category = instance.get("category", self.settings.default_category) if instance else self.settings.default_category
            color = category_color_rgb(category)
            mask = instance_map == instance_id
            overlay[mask] = overlay[mask] * 0.45 + np.array(color) * 0.55
        return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _get_generator(self, points_per_side: int = 24) -> SamAutomaticMaskGenerator:
        model = self._get_sam_model()
        return SamAutomaticMaskGenerator(
            model=model,
            points_per_side=points_per_side,
            crop_n_layers=0,
            crop_overlap_ratio=0.2,
            min_mask_region_area=128,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
        )

    def _get_predictor(self) -> SamPredictor:
        if self._predictor is None:
            self._predictor = SamPredictor(self._get_sam_model())
        return self._predictor

    def _get_sam_model(self):
        if self._sam_model is None:
            if not self.settings.checkpoint_path.exists():
                raise FileNotFoundError(f"未找到模型权重: {self.settings.checkpoint_path}")
            self._sam_model = sam_model_registry[self.settings.default_model_type](
                checkpoint=str(self.settings.checkpoint_path)
            )
            self._sam_model.to(device=self.device)
        return self._sam_model

    def _masks_to_instance_map(self, masks: Iterable[dict], shape: tuple[int, int]) -> np.ndarray:
        ordered = sorted(masks, key=lambda item: item["area"], reverse=True)
        instance_map = np.zeros(shape, dtype=np.uint16)
        total_area = shape[0] * shape[1]
        next_id = 1
        for entry in ordered:
            if entry["area"] >= total_area * 0.98:
                continue
            mask = entry["segmentation"].astype(bool)
            if not np.any(mask):
                continue
            instance_map[mask] = next_id
            next_id += 1
        unique_ids = [int(v) for v in np.unique(instance_map) if int(v) != 0]
        for new_id, old_id in enumerate(unique_ids, start=1):
            if new_id != old_id:
                instance_map[instance_map == old_id] = new_id
        return instance_map
