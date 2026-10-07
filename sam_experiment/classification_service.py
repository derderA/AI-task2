from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from openai import OpenAI

from .config import Settings
from .deepseek_config_service import DeepSeekConfigService
from .storage import Storage


def category_color_rgb(category: str) -> Tuple[int, int, int]:
    seed = category.strip() or "未分类"
    base = sum((index + 1) * ord(char) for index, char in enumerate(seed))
    return (
        int((base * 37 + 79) % 205 + 30),
        int((base * 67 + 131) % 205 + 30),
        int((base * 97 + 191) % 205 + 30),
    )


def category_color_hex(category: str) -> str:
    red, green, blue = category_color_rgb(category)
    return "#{:02x}{:02x}{:02x}".format(red, green, blue)


class ClassificationService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        config_service: DeepSeekConfigService,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.config_service = config_service

    def classify_annotation(
        self,
        annotation_path,
        session_values: Optional[dict] = None,
        provider: str = "deepseek",
    ) -> dict:
        annotation = self.storage.read_json(annotation_path)
        instances = list(annotation.get("instances", []))
        if not instances:
            raise ValueError("当前标注中没有可分类的实例。")

        image_width, image_height = self._resolve_image_size(annotation)
        cluster_count = self._suggest_cluster_count(len(instances))
        assignments, cluster_payload = self._cluster_instances(instances, image_width, image_height, cluster_count)

        if provider == "deepseek":
            cluster_names, reasoning, used_provider = self._name_clusters_with_optional_llm(
                annotation=annotation,
                cluster_payload=cluster_payload,
                session_values=session_values,
            )
        else:
            cluster_names = self._fallback_cluster_names(cluster_payload)
            reasoning = []
            used_provider = "rule"

        for instance in instances:
            cluster_id = int(assignments.get(int(instance["id"]), 0))
            category = cluster_names.get(cluster_id) or "编组%s" % (cluster_id + 1)
            instance["cluster_id"] = cluster_id
            instance["category"] = category

        category_counts = Counter(item["category"] for item in instances)
        annotation["instances"] = instances
        annotation["classes"] = sorted(category_counts)
        annotation["image_size"] = [image_width, image_height]
        annotation["classification_summary"] = {
            "provider": used_provider,
            "cluster_count": cluster_count,
            "class_counts": dict(sorted(category_counts.items())),
            "clusters": self._build_cluster_summary(cluster_payload, cluster_names),
            "llm_reasoning": reasoning,
        }
        self.storage.write_json(annotation_path, annotation)
        return annotation

    def decorate_annotation(self, annotation: Optional[dict]) -> Optional[dict]:
        if not annotation:
            return None
        image_width, image_height = self._resolve_image_size(annotation)
        decorated = dict(annotation)
        instances = []
        category_counts = Counter()
        for item in annotation.get("instances", []):
            instance = dict(item)
            category = instance.get("category", self.settings.default_category)
            category_counts[category] += 1
            color = category_color_hex(category)
            bbox = instance.get("bbox", [0, 0, 0, 0])
            left = self._percent(bbox[0], image_width)
            top = self._percent(bbox[1], image_height)
            width = self._percent(bbox[2], image_width)
            height = self._percent(bbox[3], image_height)
            instance["category_color"] = color
            instance["bbox_style"] = {
                "left": "%.4f%%" % left,
                "top": "%.4f%%" % top,
                "width": "%.4f%%" % width,
                "height": "%.4f%%" % height,
            }
            instances.append(instance)

        categories = []
        for name, count in sorted(category_counts.items(), key=lambda pair: (-pair[1], pair[0])):
            categories.append(
                {
                    "name": name,
                    "count": count,
                    "color": category_color_hex(name),
                }
            )

        decorated["instances"] = instances
        decorated["image_size"] = [image_width, image_height]
        decorated["category_sidebar"] = categories
        decorated["classification_summary"] = annotation.get("classification_summary", {})
        return decorated

    def _resolve_image_size(self, annotation: dict) -> Tuple[int, int]:
        image_size = annotation.get("image_size") or []
        if len(image_size) == 2:
            return int(image_size[0]), int(image_size[1])

        instance_map_path = self.storage.from_relpath(annotation["instance_map_path"])
        instance_map = cv2.imread(str(instance_map_path), cv2.IMREAD_UNCHANGED)
        if instance_map is None:
            raise FileNotFoundError("无法读取实例图: %s" % instance_map_path)
        if instance_map.ndim == 3:
            instance_map = instance_map[:, :, 0]
        height, width = instance_map.shape[:2]
        return int(width), int(height)

    def _suggest_cluster_count(self, instance_count: int) -> int:
        if instance_count <= 1:
            return 1
        if instance_count <= 4:
            return 2
        return min(6, max(2, int(round(np.sqrt(float(instance_count))))))

    def _cluster_instances(
        self,
        instances: List[dict],
        image_width: int,
        image_height: int,
        cluster_count: int,
    ) -> Tuple[Dict[int, int], List[dict]]:
        features = []
        for instance in instances:
            bbox = instance.get("bbox", [0, 0, 1, 1])
            width = max(1.0, float(bbox[2]))
            height = max(1.0, float(bbox[3]))
            area = float(instance.get("area", width * height))
            features.append(
                [
                    np.log1p(area / float(max(image_width * image_height, 1))),
                    np.log(width / height),
                    area / float(max(width * height, 1.0)),
                    float(instance.get("center", [0, 0])[0]) / float(max(image_width, 1)),
                    float(instance.get("center", [0, 0])[1]) / float(max(image_height, 1)),
                    width / float(max(image_width, 1)),
                    height / float(max(image_height, 1)),
                ]
            )

        feature_array = np.asarray(features, dtype=np.float32)
        if cluster_count <= 1 or len(instances) <= 1:
            labels = np.zeros((len(instances), 1), dtype=np.int32)
        else:
            means = feature_array.mean(axis=0, keepdims=True)
            stds = feature_array.std(axis=0, keepdims=True)
            stds[stds < 1e-6] = 1.0
            normalized = (feature_array - means) / stds
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
            _compactness, labels, _centers = cv2.kmeans(
                normalized,
                cluster_count,
                None,
                criteria,
                8,
                cv2.KMEANS_PP_CENTERS,
            )

        assignments = {}
        clusters = {}
        for instance, raw_label in zip(instances, labels.flatten().tolist()):
            label = int(raw_label)
            assignments[int(instance["id"])] = label
            clusters.setdefault(label, []).append(instance)

        cluster_payload = []
        for cluster_id in sorted(clusters):
            items = clusters[cluster_id]
            avg_area = float(np.mean([item["area"] for item in items]))
            avg_fill = float(np.mean([self._fill_ratio(item) for item in items]))
            avg_aspect = float(np.mean([self._aspect_ratio(item) for item in items]))
            avg_center_x = float(np.mean([item["center"][0] for item in items]))
            avg_center_y = float(np.mean([item["center"][1] for item in items]))
            cluster_payload.append(
                {
                    "cluster_id": cluster_id,
                    "instance_ids": [int(item["id"]) for item in items],
                    "count": len(items),
                    "avg_area": round(avg_area, 2),
                    "avg_area_ratio": round(avg_area / float(max(image_width * image_height, 1)), 4),
                    "avg_fill_ratio": round(avg_fill, 4),
                    "avg_aspect_ratio": round(avg_aspect, 4),
                    "avg_center": [round(avg_center_x, 1), round(avg_center_y, 1)],
                    "position_hint": self._position_hint(avg_center_x, avg_center_y, image_width, image_height),
                    "shape_hint": self._shape_hint(avg_aspect, avg_fill),
                    "size_hint": self._size_hint(avg_area / float(max(image_width * image_height, 1))),
                }
            )
        return assignments, cluster_payload

    def _name_clusters_with_optional_llm(
        self,
        annotation: dict,
        cluster_payload: List[dict],
        session_values: Optional[dict],
    ) -> Tuple[Dict[int, str], List[str], str]:
        fallback = self._fallback_cluster_names(cluster_payload)
        try:
            credentials = self.config_service.resolve_credentials(session_values)
            api_key = credentials.get("api_key", "").strip()
            if not api_key:
                return fallback, [], "rule"
            prompt = self._build_naming_prompt(annotation, cluster_payload, fallback)
            content = self._request_text(prompt, credentials)
            mapping, reasoning = self._parse_cluster_response(content)
            if not mapping:
                return fallback, reasoning, "rule"
            resolved = {}
            for cluster in cluster_payload:
                cluster_id = int(cluster["cluster_id"])
                resolved[cluster_id] = mapping.get(cluster_id) or fallback[cluster_id]
            return resolved, reasoning, "deepseek"
        except Exception:
            return fallback, [], "rule"

    def _build_naming_prompt(self, annotation: dict, cluster_payload: List[dict], fallback: Dict[int, str]) -> str:
        return (
            "你是一名遥感实例分组命名助手。当前系统已经基于几何特征把多个 SAM 分割主体分成若干组，"
            "请你为每个分组生成一个简洁的中文类别名。\n\n"
            "要求：\n"
            "1. 只能依据提供的几何和分布统计命名，不要臆测具体武器型号或场景细节。\n"
            "2. 如果无法可靠判断语义类别，请使用中性的形态名称，例如“大型紧凑主体”“横向细长主体群”。\n"
            "3. 返回 JSON，不要附加额外解释。\n"
            "4. JSON 格式必须为："
            '{"clusters":[{"cluster_id":0,"category":"类别名","reason":"一句理由"}]}\n\n'
            "图像路径: %s\n"
            "默认候选名称: %s\n"
            "分组统计:\n%s\n"
            % (
                annotation.get("image_path", ""),
                json.dumps(fallback, ensure_ascii=False),
                json.dumps(cluster_payload, ensure_ascii=False, indent=2),
            )
        )

    def _request_text(self, prompt: str, credentials: dict) -> str:
        client = OpenAI(
            api_key=credentials.get("api_key", ""),
            base_url=credentials.get("base_url", self.settings.deepseek_base_url),
        )
        response = client.chat.completions.create(
            model=credentials.get("model", self.settings.deepseek_model),
            messages=[
                {
                    "role": "system",
                    "content": "你是一名严谨的遥感标注辅助助手，只能输出用户要求的 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _parse_cluster_response(self, content: str) -> Tuple[Dict[int, str], List[str]]:
        payload = None
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.S)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = None
        if not isinstance(payload, dict):
            return {}, []

        mapping = {}
        reasoning = []
        for item in payload.get("clusters", []):
            if not isinstance(item, dict):
                continue
            try:
                cluster_id = int(item.get("cluster_id"))
            except (TypeError, ValueError):
                continue
            category = str(item.get("category", "")).strip()
            if not category:
                continue
            mapping[cluster_id] = category
            reason = str(item.get("reason", "")).strip()
            if reason:
                reasoning.append("cluster_%s: %s" % (cluster_id, reason))
        return mapping, reasoning

    def _fallback_cluster_names(self, cluster_payload: List[dict]) -> Dict[int, str]:
        resolved = {}
        used = set()
        for cluster in cluster_payload:
            base = "%s%s" % (cluster["size_hint"], cluster["shape_hint"])
            if cluster["count"] > 1:
                base = "%s主体群" % base
            else:
                base = "%s主体" % base
            name = base
            index = 2
            while name in used:
                name = "%s%s" % (base, index)
                index += 1
            used.add(name)
            resolved[int(cluster["cluster_id"])] = name
        return resolved

    def _build_cluster_summary(self, cluster_payload: List[dict], cluster_names: Dict[int, str]) -> List[dict]:
        summary = []
        for cluster in cluster_payload:
            cluster_id = int(cluster["cluster_id"])
            item = dict(cluster)
            item["category"] = cluster_names.get(cluster_id, "编组%s" % (cluster_id + 1))
            item["color"] = category_color_hex(item["category"])
            summary.append(item)
        return summary

    def _fill_ratio(self, instance: dict) -> float:
        bbox = instance.get("bbox", [0, 0, 1, 1])
        width = max(float(bbox[2]), 1.0)
        height = max(float(bbox[3]), 1.0)
        return float(instance.get("area", width * height)) / float(width * height)

    def _aspect_ratio(self, instance: dict) -> float:
        bbox = instance.get("bbox", [0, 0, 1, 1])
        width = max(float(bbox[2]), 1.0)
        height = max(float(bbox[3]), 1.0)
        return width / height

    def _position_hint(self, center_x: float, center_y: float, image_width: int, image_height: int) -> str:
        if center_x < image_width * 0.33:
            horizontal = "左侧"
        elif center_x > image_width * 0.66:
            horizontal = "右侧"
        else:
            horizontal = "中部"

        if center_y < image_height * 0.33:
            vertical = "上部"
        elif center_y > image_height * 0.66:
            vertical = "下部"
        else:
            vertical = "中部"
        return "%s%s" % (horizontal, vertical)

    def _shape_hint(self, avg_aspect: float, avg_fill: float) -> str:
        if avg_aspect >= 2.2:
            return "横向细长"
        if avg_aspect <= 0.45:
            return "纵向细长"
        if avg_fill >= 0.68:
            return "紧凑"
        return "常规"

    def _size_hint(self, area_ratio: float) -> str:
        if area_ratio >= 0.025:
            return "大型"
        if area_ratio >= 0.008:
            return "中型"
        return "小型"

    def _percent(self, value: float, total: float) -> float:
        if total <= 0:
            return 0.0
        return max(0.0, min(100.0, float(value) * 100.0 / float(total)))
