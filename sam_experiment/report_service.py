from __future__ import annotations

from collections import Counter
from pathlib import Path

from .config import Settings
from .storage import Storage


class ReportService:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage
        self.settings.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_image_report(self, annotation: dict) -> tuple[dict, Path]:
        summary = self.summarize_image_annotation(annotation)
        report_path = self.settings.report_dir / f"{Path(annotation['image_path']).stem}_report.md"
        report_path.write_text(self._render_report(summary), encoding="utf-8")
        annotation["report_path"] = self.storage.to_relpath(report_path)
        return summary, report_path

    def summarize_image_annotation(self, annotation: dict) -> dict:
        counts = Counter(
            item["category"]
            for item in annotation["instances"]
            if item["category"] and item["category"] != self.settings.default_category
        )
        total_instances = len(annotation["instances"])
        region_type, terrain_desc = self._terrain_inference(counts)
        summary = {
            "image_path": annotation["image_path"],
            "total_instances": total_instances,
            "counts": dict(counts),
            "region_type": region_type,
            "terrain_description": terrain_desc,
            "conclusion": self._build_conclusion(counts, terrain_desc),
            "evidence": self._build_evidence(counts),
        }
        return summary

    def generate_video_report(self, video_name: str, frame_summaries: list[dict]) -> tuple[dict, Path]:
        payload = self.summarize_video_frames(video_name, frame_summaries)
        report_path = self.settings.video_dir / video_name / "video_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self._render_video_report(payload), encoding="utf-8")
        return payload, report_path

    def summarize_video_frames(self, video_name: str, frame_summaries: list[dict]) -> dict:
        total_counter = Counter()
        for summary in frame_summaries:
            total_counter.update(summary["counts"])
        region_type, terrain_desc = self._terrain_inference(total_counter)
        payload = {
            "video_name": video_name,
            "frame_count": len(frame_summaries),
            "counts": dict(total_counter),
            "region_type": region_type,
            "terrain_description": terrain_desc,
            "conclusion": self._build_conclusion(total_counter, terrain_desc),
            "evidence": self._build_evidence(total_counter),
        }
        return payload

    def _terrain_inference(self, counts: Counter) -> tuple[str, str]:
        if not counts:
            return "待人工判读区域", "当前标注主要仍为未分类实例，建议先完成关键目标标注后再生成报告。"

        airport_related = counts["plane"] + counts["helicopter"]
        port_related = counts["ship"] + counts["harbor"]
        road_related = counts["bridge"] + counts["large_vehicle"] + counts["small_vehicle"]
        leisure_related = (
            counts["swimming pool"]
            + counts["tennis court"]
            + counts["basketball court"]
            + counts["soccer ball field"]
            + counts["baseball diamond"]
            + counts["ground track field"]
        )
        industrial_related = counts["storage tank"] + counts["harbor"] + counts["large vehicle"]

        scores = {
            "机场/起降保障区域": airport_related,
            "港口/滨水物流区域": port_related,
            "交通基础设施区域": road_related,
            "城市生活/体育设施区域": leisure_related,
            "工业仓储区域": industrial_related,
        }
        region_type = max(scores, key=scores.get)
        descriptions = {
            "机场/起降保障区域": "图像中出现较多飞机或直升机实例，结合大尺度硬化地表，推测区域接近机场、停机坪或飞行保障设施。",
            "港口/滨水物流区域": "船舶与港口类目标占比明显，说明该区域更可能是码头、沿海装卸区或港航设施集中带。",
            "交通基础设施区域": "桥梁与车辆目标相对集中，说明区域内交通组织特征明显，可能包含主干道、桥梁或运输节点。",
            "城市生活/体育设施区域": "球场、泳池等规则场地较多，更接近城市校园、居住区或综合体育设施分布区域。",
            "工业仓储区域": "储罐、大型车辆和港口设施较活跃，推测这里更接近仓储、能源或工业物流区域。",
        }
        return region_type, descriptions[region_type]

    def _build_conclusion(self, counts: Counter, terrain_desc: str) -> str:
        if not counts:
            return terrain_desc
        top_items = ", ".join(f"{name} {count}个" for name, count in counts.most_common(3))
        return f"自动统计结果显示主要目标包括 {top_items}。{terrain_desc}"

    def _build_evidence(self, counts: Counter) -> list[str]:
        if not counts:
            return ["尚未形成足够证据，建议补充类别标注后重新生成报告。"]
        return [f"{name}: {count} 个实例" for name, count in counts.most_common()]

    def _render_report(self, summary: dict) -> str:
        evidence_lines = "\n".join(f"- {item}" for item in summary["evidence"])
        counts_lines = "\n".join(f"- {name}: {count}" for name, count in summary["counts"].items()) or "- 暂无已分类实例"
        return (
            f"# 图像分析报告\n\n"
            f"- 图像: `{summary['image_path']}`\n"
            f"- 实例总数: {summary['total_instances']}\n"
            f"- 推测区域类型: {summary['region_type']}\n\n"
            f"## 类别统计\n{counts_lines}\n\n"
            f"## 地貌描述\n{summary['terrain_description']}\n\n"
            f"## 结论\n{summary['conclusion']}\n\n"
            f"## 证据\n{evidence_lines}\n"
        )

    def _render_video_report(self, summary: dict) -> str:
        counts_lines = "\n".join(f"- {name}: {count}" for name, count in summary["counts"].items()) or "- 暂无已分类实例"
        evidence_lines = "\n".join(f"- {item}" for item in summary["evidence"])
        return (
            f"# 视频分析报告\n\n"
            f"- 视频名称: `{summary['video_name']}`\n"
            f"- 抽帧数量: {summary['frame_count']}\n"
            f"- 推测区域类型: {summary['region_type']}\n\n"
            f"## 汇总统计\n{counts_lines}\n\n"
            f"## 地貌描述\n{summary['terrain_description']}\n\n"
            f"## 结论\n{summary['conclusion']}\n\n"
            f"## 证据\n{evidence_lines}\n"
        )
