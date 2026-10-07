#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

import cv2
import numpy as np

from sam_experiment.annotation_service import AnnotationService
from sam_experiment.config import get_settings
from sam_experiment.report_service import ReportService
from sam_experiment.sam_service import SamExperimentService
from sam_experiment.storage import Storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于当前 dataset 压缩包结构，对 DOTA 图像样本执行更完整的 SAM 实验测试并输出汇总报告。"
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="数据集根目录，默认使用项目下的 dataset/。",
    )
    parser.add_argument(
        "--label-version",
        choices=("v1.0", "v1.5"),
        default="v1.5",
        help="选择对照标签版本，默认 v1.5。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="最多测试多少张图像，默认 10。",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=24,
        help="SAM AutomaticMaskGenerator 的 points_per_side 参数，默认 24。",
    )
    parser.add_argument(
        "--image-stems",
        default="",
        help="指定图像编号，逗号分隔，例如 P0010,P0011；为空时自动抽样。",
    )
    parser.add_argument(
        "--max-long-side",
        type=int,
        default=1536,
        help="测试前将图像最长边缩放到该尺寸以内，默认 1536。",
    )
    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=0.25,
        help="基于 bbox 粗匹配时使用的 IoU 阈值，默认 0.25。",
    )
    return parser


def candidate_image_archives(dataset_dir: Path) -> List[Path]:
    image_dir = dataset_dir / "images"
    archives = sorted(image_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError("未在 dataset/images 下找到图像压缩包。")
    return archives


def candidate_label_archives(dataset_dir: Path, label_version: str) -> List[Path]:
    label_dir = dataset_dir / ("labelTxt-%s" % label_version)
    archives = sorted(label_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError("未在 %s 下找到标签压缩包。" % label_dir)
    return archives


def build_image_index(archives: List[Path]) -> Dict[str, Dict[str, str]]:
    index = {}
    for archive_path in archives:
        with ZipFile(str(archive_path), "r") as zf:
            for name in zf.namelist():
                lower_name = name.lower()
                if not lower_name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    continue
                stem = Path(name).stem
                index.setdefault(
                    stem,
                    {
                        "archive_path": str(archive_path),
                        "member_name": name,
                    },
                )
    return index


def build_label_index(archives: List[Path]) -> Dict[str, Dict[str, str]]:
    index = {}
    for archive_path in archives:
        with ZipFile(str(archive_path), "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".txt"):
                    continue
                stem = Path(name).stem
                index.setdefault(
                    stem,
                    {
                        "archive_path": str(archive_path),
                        "member_name": name,
                    },
                )
    return index


def select_image_stems(
    image_index: Dict[str, Dict[str, str]],
    label_index: Dict[str, Dict[str, str]],
    max_images: int,
    explicit_stems: Optional[List[str]] = None,
) -> List[str]:
    common_stems = sorted(set(image_index) & set(label_index))
    if explicit_stems:
        selected = [stem for stem in explicit_stems if stem in image_index and stem in label_index]
        if not selected:
            raise ValueError("指定的 image stems 在当前图片压缩包和标签压缩包中都没有匹配到。")
        return selected[:max_images]
    if not common_stems:
        raise ValueError("当前图片压缩包与标签压缩包之间没有公共样本。")
    return common_stems[:max_images]


def extract_image_member(image_meta: Dict[str, str], output_dir: Path) -> Path:
    archive_path = Path(image_meta["archive_path"])
    member_name = image_meta["member_name"]
    output_path = output_dir / Path(member_name).name
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path
    with ZipFile(str(archive_path), "r") as zf:
        output_path.write_bytes(zf.read(member_name))
    return output_path


def read_image_size(image_path: Path) -> Tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("无法读取图片: %s" % image_path)
    height, width = image.shape[:2]
    return width, height


def resize_for_test(src_path: Path, output_dir: Path, max_long_side: int) -> Path:
    if max_long_side <= 0:
        return src_path
    image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("无法读取抽样图片: %s" % src_path)
    height, width = image.shape[:2]
    long_side = max(width, height)
    if long_side <= max_long_side:
        return src_path

    scale = float(max_long_side) / float(long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    output_dir.mkdir(parents=True, exist_ok=True)
    resized_path = output_dir / ("%s_resized.png" % src_path.stem)
    cv2.imwrite(str(resized_path), resized)
    return resized_path


def read_label_lines(label_meta: Dict[str, str]) -> List[str]:
    archive_path = Path(label_meta["archive_path"])
    member_name = label_meta["member_name"]
    with ZipFile(str(archive_path), "r") as zf:
        content = zf.read(member_name).decode("utf-8", errors="ignore")
    return [line.strip() for line in content.splitlines() if line.strip()]


def parse_dota_objects(label_lines: List[str]) -> List[dict]:
    objects = []
    for line in label_lines:
        lower_line = line.lower()
        if lower_line.startswith("imagesource") or lower_line.startswith("gsd"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            coords = [float(value) for value in parts[:8]]
        except ValueError:
            continue
        category = parts[8]
        difficulty = parts[9] if len(parts) > 9 else ""
        objects.append(
            {
                "coords": coords,
                "category": category,
                "difficulty": difficulty,
            }
        )
    return objects


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def bbox_from_coords(coords: List[float]) -> List[float]:
    xs = coords[0::2]
    ys = coords[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def scale_objects(
    objects: List[dict],
    scale_x: float,
    scale_y: float,
    image_width: int,
    image_height: int,
) -> List[dict]:
    scaled = []
    for item in objects:
        coords = item["coords"]
        scaled_coords = []
        for index, value in enumerate(coords):
            if index % 2 == 0:
                scaled_value = value * scale_x
                scaled_value = min(max(scaled_value, 0.0), float(max(image_width - 1, 0)))
            else:
                scaled_value = value * scale_y
                scaled_value = min(max(scaled_value, 0.0), float(max(image_height - 1, 0)))
            scaled_coords.append(scaled_value)
        scaled.append(
            {
                "coords": scaled_coords,
                "category": item["category"],
                "difficulty": item["difficulty"],
                "bbox": bbox_from_coords(scaled_coords),
            }
        )
    return scaled


def build_gt_union_mask(image_shape: Tuple[int, int], objects: List[dict]) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in objects:
        polygon = np.array(item["coords"], dtype=np.float32).reshape(-1, 2)
        polygon = np.round(polygon).astype(np.int32)
        if polygon.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    return safe_divide(inter_area, union_area)


def pred_bbox_xyxy(instance: dict) -> List[float]:
    x, y, w, h = instance["bbox"]
    return [float(x), float(y), float(x + w), float(y + h)]


def greedy_bbox_match(pred_instances: List[dict], gt_objects: List[dict], threshold: float) -> dict:
    candidate_pairs = []
    for pred_index, instance in enumerate(pred_instances):
        pred_box = pred_bbox_xyxy(instance)
        for gt_index, item in enumerate(gt_objects):
            iou = bbox_iou(pred_box, item["bbox"])
            if iou >= threshold:
                candidate_pairs.append((iou, pred_index, gt_index))
    candidate_pairs.sort(reverse=True)

    matched_pred = set()
    matched_gt = set()
    matched_pairs = []
    for iou, pred_index, gt_index in candidate_pairs:
        if pred_index in matched_pred or gt_index in matched_gt:
            continue
        matched_pred.add(pred_index)
        matched_gt.add(gt_index)
        matched_pairs.append(
            {
                "pred_index": pred_index,
                "gt_index": gt_index,
                "bbox_iou": round(iou, 4),
            }
        )

    matched_count = len(matched_pairs)
    precision = safe_divide(matched_count, len(pred_instances))
    recall = safe_divide(matched_count, len(gt_objects))
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "matched_predictions": matched_count,
        "matched_ground_truth": matched_count,
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": f1,
        "matched_pairs": matched_pairs[:10],
    }


def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    pred_area = int(pred_mask.sum())
    gt_area = int(gt_mask.sum())
    intersection = int(np.logical_and(pred_mask, gt_mask).sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())

    precision = safe_divide(intersection, pred_area)
    recall = safe_divide(intersection, gt_area)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    iou = safe_divide(intersection, union)
    return {
        "pred_area": pred_area,
        "gt_area": gt_area,
        "intersection_area": intersection,
        "union_area": union,
        "foreground_precision": precision,
        "foreground_recall": recall,
        "foreground_f1": f1,
        "foreground_iou": iou,
    }


def quality_score(image_result: dict) -> float:
    return (
        image_result["foreground_iou"] * 0.4
        + image_result["foreground_f1"] * 0.2
        + image_result["object_f1"] * 0.3
        + image_result["object_recall"] * 0.1
    )


def quality_level(image_result: dict) -> str:
    score = quality_score(image_result)
    if score >= 0.5 and image_result["foreground_iou"] >= 0.2:
        return "较好"
    if score >= 0.25:
        return "一般"
    return "较弱"


def build_case_digest(item: dict) -> dict:
    return {
        "image_stem": item["image_stem"],
        "quality_level": item["quality_level"],
        "quality_score": round(item["quality_score"], 4),
        "foreground_iou": item["foreground_iou"],
        "object_f1": item["object_f1"],
        "predicted_instances": item["predicted_instances"],
        "ground_truth_objects": item["ground_truth_objects"],
        "runtime_seconds": item["runtime_seconds"],
        "overlay_path": item["overlay_path"],
        "annotation_path": item["annotation_path"],
    }


def build_markdown(summary: dict) -> str:
    lines = [
        "# DOTA 压缩包实验测试报告",
        "",
        "## 实验配置",
        "",
        "- 标签版本: `%s`" % summary["label_version"],
        "- 图片压缩包: `%s`" % ", ".join(summary["image_archives"]),
        "- 标签压缩包: `%s`" % ", ".join(summary["label_archives"]),
        "- 测试样本数: %s" % summary["tested_images"],
        "- points_per_side: %s" % summary["points_per_side"],
        "- max_long_side: %s" % summary["max_long_side"],
        "- bbox 匹配阈值: %.2f" % summary["match_iou_threshold"],
        "",
        "## 汇总指标",
        "",
        "- 平均预测实例数: %.3f" % summary["average_predicted_instances"],
        "- 平均真值目标数: %.3f" % summary["average_ground_truth_objects"],
        "- 平均绝对误差: %.3f" % summary["average_absolute_error"],
        "- 平均运行时间: %.3f 秒" % summary["average_runtime_seconds"],
        "- 平均前景 IoU: %.4f" % summary["average_foreground_iou"],
        "- 平均前景 Precision: %.4f" % summary["average_foreground_precision"],
        "- 平均前景 Recall: %.4f" % summary["average_foreground_recall"],
        "- 平均前景 F1: %.4f" % summary["average_foreground_f1"],
        "- 平均目标 Precision: %.4f" % summary["average_object_precision"],
        "- 平均目标 Recall: %.4f" % summary["average_object_recall"],
        "- 平均目标 F1: %.4f" % summary["average_object_f1"],
        "- 平均预测掩码覆盖率: %.4f" % summary["average_pred_mask_coverage"],
        "- 平均真值掩码覆盖率: %.4f" % summary["average_gt_mask_coverage"],
        "",
        "## 质量分布",
        "",
    ]
    for name, count in sorted(summary["quality_distribution"].items()):
        lines.append("- %s: %s" % (name, count))
    lines.extend(
        [
            "",
            "## 类别统计",
            "",
        ]
    )
    if summary["category_counts"]:
        for name, count in sorted(summary["category_counts"].items()):
            lines.append("- %s: %s" % (name, count))
    else:
        lines.append("- 未解析到有效类别")

    lines.extend(
        [
            "",
            "## 最佳样本",
            "",
        ]
    )
    for item in summary["top_cases"]:
        lines.append(
            "- `%s` | 质量=%s | score=%.4f | fg IoU=%.4f | obj F1=%.4f | overlay=`%s`"
            % (
                item["image_stem"],
                item["quality_level"],
                item["quality_score"],
                item["foreground_iou"],
                item["object_f1"],
                item["overlay_path"],
            )
        )

    lines.extend(
        [
            "",
            "## 较弱样本",
            "",
        ]
    )
    for item in summary["bottom_cases"]:
        lines.append(
            "- `%s` | 质量=%s | score=%.4f | fg IoU=%.4f | obj F1=%.4f | overlay=`%s`"
            % (
                item["image_stem"],
                item["quality_level"],
                item["quality_score"],
                item["foreground_iou"],
                item["object_f1"],
                item["overlay_path"],
            )
        )

    lines.extend(
        [
            "",
            "## 样本明细",
            "",
            "| 图像 | 质量 | 运行时间(s) | 预测实例数 | 真值目标数 | 前景 IoU | 前景 F1 | 目标 F1 | 绝对误差 | overlay |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in summary["images"]:
        lines.append(
            "| %s | %s | %.3f | %s | %s | %.4f | %.4f | %.4f | %s | `%s` |"
            % (
                item["image_stem"],
                item["quality_level"],
                item["runtime_seconds"],
                item["predicted_instances"],
                item["ground_truth_objects"],
                item["foreground_iou"],
                item["foreground_f1"],
                item["object_f1"],
                item["absolute_error"],
                item["overlay_path"],
            )
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 本脚本直接从 `dataset/*.zip` 中抽样图片，不要求提前解压整包数据。",
            "- 量化评估包含两部分：像素级前景指标，以及基于 bbox 的粗粒度目标匹配指标。",
            "- 由于 DOTA 提供的是旋转框标注，而 SAM 输出的是实例分割结果，这里采用的是近似评估，不等价于正式检测 mAP。",
            "- 当前脚本默认会先缩放超大图像，再执行 SAM，以保证在普通机器上也能完成测试。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    storage = Storage(settings.base_dir)
    annotation_service = AnnotationService(settings, storage)
    sam_service = SamExperimentService(settings, storage, annotation_service)
    report_service = ReportService(settings, storage)

    dataset_dir = (settings.base_dir / args.dataset_dir).resolve()
    image_archives = candidate_image_archives(dataset_dir)
    label_archives = candidate_label_archives(dataset_dir, args.label_version)
    image_index = build_image_index(image_archives)
    label_index = build_label_index(label_archives)

    explicit_stems = [item.strip() for item in args.image_stems.split(",") if item.strip()]
    selected_stems = select_image_stems(
        image_index=image_index,
        label_index=label_index,
        max_images=args.max_images,
        explicit_stems=explicit_stems or None,
    )

    sample_dir = settings.data_dir / "archive_samples" / args.label_version
    category_counter = Counter()
    quality_counter = Counter()
    image_results = []

    for stem in selected_stems:
        image_meta = image_index[stem]
        label_meta = label_index[stem]
        image_path = extract_image_member(image_meta, sample_dir)
        test_image_path = resize_for_test(image_path, sample_dir, args.max_long_side)
        original_width, original_height = read_image_size(image_path)
        test_width, test_height = read_image_size(test_image_path)
        scale_x = safe_divide(test_width, original_width)
        scale_y = safe_divide(test_height, original_height)

        label_lines = read_label_lines(label_meta)
        gt_objects = parse_dota_objects(label_lines)
        scaled_gt_objects = scale_objects(gt_objects, scale_x, scale_y, test_width, test_height)
        for item in scaled_gt_objects:
            category_counter[item["category"]] += 1

        run_name = "archive_%s_%s" % (args.label_version.replace(".", "_"), stem)
        start_time = perf_counter()
        payload = sam_service.auto_segment(
            storage.to_relpath(test_image_path),
            run_name=run_name,
            points_per_side=args.points_per_side,
        )
        runtime_seconds = perf_counter() - start_time
        report_summary, report_path = report_service.generate_image_report(payload)

        instance_map_path = storage.from_relpath(payload["instance_map_path"])
        instance_map = cv2.imread(str(instance_map_path), cv2.IMREAD_UNCHANGED)
        if instance_map is None:
            raise FileNotFoundError("无法读取实例图: %s" % instance_map_path)
        if instance_map.ndim == 3:
            instance_map = instance_map[:, :, 0]
        pred_mask = instance_map > 0
        gt_mask = build_gt_union_mask((test_height, test_width), scaled_gt_objects)

        binary = compute_binary_metrics(pred_mask, gt_mask)
        matched = greedy_bbox_match(payload.get("instances", []), scaled_gt_objects, args.match_iou_threshold)
        predicted_instances = len(payload.get("instances", []))
        ground_truth_objects = len(scaled_gt_objects)
        absolute_error = abs(predicted_instances - ground_truth_objects)
        pred_mask_coverage = safe_divide(binary["pred_area"], test_width * test_height)
        gt_mask_coverage = safe_divide(binary["gt_area"], test_width * test_height)

        image_result = {
            "image_stem": stem,
            "image_archive": image_meta["archive_path"],
            "image_member": image_meta["member_name"],
            "image_path": storage.to_relpath(image_path),
            "test_image_path": storage.to_relpath(test_image_path),
            "label_archive": label_meta["archive_path"],
            "label_member": label_meta["member_name"],
            "original_size": [original_width, original_height],
            "test_size": [test_width, test_height],
            "ground_truth_objects": ground_truth_objects,
            "predicted_instances": predicted_instances,
            "absolute_error": absolute_error,
            "runtime_seconds": round(runtime_seconds, 3),
            "pred_mask_coverage": round(pred_mask_coverage, 4),
            "gt_mask_coverage": round(gt_mask_coverage, 4),
            "foreground_iou": round(binary["foreground_iou"], 4),
            "foreground_precision": round(binary["foreground_precision"], 4),
            "foreground_recall": round(binary["foreground_recall"], 4),
            "foreground_f1": round(binary["foreground_f1"], 4),
            "object_precision": round(matched["object_precision"], 4),
            "object_recall": round(matched["object_recall"], 4),
            "object_f1": round(matched["object_f1"], 4),
            "matched_predictions": matched["matched_predictions"],
            "matched_ground_truth": matched["matched_ground_truth"],
            "matched_pairs_preview": matched["matched_pairs"],
            "annotation_path": payload["annotation_path"],
            "overlay_path": payload["overlay_path"],
            "report_path": storage.to_relpath(report_path),
            "report_summary": report_summary,
        }
        image_result["quality_score"] = round(quality_score(image_result), 4)
        image_result["quality_level"] = quality_level(image_result)
        quality_counter[image_result["quality_level"]] += 1
        image_results.append(image_result)

    tested_images = len(image_results)
    image_results_sorted = sorted(image_results, key=lambda item: item["quality_score"], reverse=True)

    summary = {
        "dataset_dir": str(dataset_dir),
        "label_version": args.label_version,
        "image_archives": [str(path.relative_to(settings.base_dir)) for path in image_archives],
        "label_archives": [str(path.relative_to(settings.base_dir)) for path in label_archives],
        "tested_images": tested_images,
        "points_per_side": args.points_per_side,
        "max_long_side": args.max_long_side,
        "match_iou_threshold": args.match_iou_threshold,
        "average_predicted_instances": round(
            sum(item["predicted_instances"] for item in image_results) / float(tested_images), 3
        ),
        "average_ground_truth_objects": round(
            sum(item["ground_truth_objects"] for item in image_results) / float(tested_images), 3
        ),
        "average_absolute_error": round(
            sum(item["absolute_error"] for item in image_results) / float(tested_images), 3
        ),
        "average_runtime_seconds": round(
            sum(item["runtime_seconds"] for item in image_results) / float(tested_images), 3
        ),
        "average_pred_mask_coverage": round(
            sum(item["pred_mask_coverage"] for item in image_results) / float(tested_images), 4
        ),
        "average_gt_mask_coverage": round(
            sum(item["gt_mask_coverage"] for item in image_results) / float(tested_images), 4
        ),
        "average_foreground_iou": round(
            sum(item["foreground_iou"] for item in image_results) / float(tested_images), 4
        ),
        "average_foreground_precision": round(
            sum(item["foreground_precision"] for item in image_results) / float(tested_images), 4
        ),
        "average_foreground_recall": round(
            sum(item["foreground_recall"] for item in image_results) / float(tested_images), 4
        ),
        "average_foreground_f1": round(
            sum(item["foreground_f1"] for item in image_results) / float(tested_images), 4
        ),
        "average_object_precision": round(
            sum(item["object_precision"] for item in image_results) / float(tested_images), 4
        ),
        "average_object_recall": round(
            sum(item["object_recall"] for item in image_results) / float(tested_images), 4
        ),
        "average_object_f1": round(
            sum(item["object_f1"] for item in image_results) / float(tested_images), 4
        ),
        "quality_distribution": dict(quality_counter),
        "category_counts": dict(category_counter),
        "top_cases": [build_case_digest(item) for item in image_results_sorted[:3]],
        "bottom_cases": [build_case_digest(item) for item in list(reversed(image_results_sorted[-3:]))],
        "images": image_results,
    }

    report_stem = "dota_archive_test_summary_%s" % args.label_version.replace(".", "_")
    json_path = settings.report_dir / ("%s.json" % report_stem)
    md_path = settings.report_dir / ("%s.md" % report_stem)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(summary), encoding="utf-8")

    print("测试完成。")
    print("样本数: %s" % tested_images)
    print("标签版本: %s" % args.label_version)
    print("平均前景 IoU: %.4f" % summary["average_foreground_iou"])
    print("平均目标 F1: %.4f" % summary["average_object_f1"])
    print("JSON 汇总: %s" % storage.to_relpath(json_path))
    print("Markdown 汇总: %s" % storage.to_relpath(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
