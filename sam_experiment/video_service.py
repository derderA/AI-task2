from __future__ import annotations

from pathlib import Path

import cv2

from .config import Settings
from .report_service import ReportService
from .sam_service import SamExperimentService
from .storage import Storage


class VideoAnalysisService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        sam_service: SamExperimentService,
        report_service: ReportService,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.sam_service = sam_service
        self.report_service = report_service

    def analyze_video(
        self,
        video_path,
        frame_interval: int = 30,
        max_frames: int = 5,
    ) -> dict:
        source = self.storage.from_relpath(video_path)
        if not source.exists():
            raise FileNotFoundError(f"未找到视频文件: {source}")
        run_dir = self.settings.video_dir / source.stem
        frame_dir = run_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频: {source}")

        frame_index = 0
        sampled = 0
        frame_summaries = []
        generated_annotations = []
        while sampled < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(1, frame_interval) == 0:
                frame_path = frame_dir / f"{source.stem}_{frame_index:05d}.png"
                cv2.imwrite(str(frame_path), frame)
                annotation = self.sam_service.auto_segment(frame_path, run_name=f"{source.stem}_{frame_index:05d}")
                generated_annotations.append(annotation["image_path"])
                summary, _ = self.report_service.generate_image_report(annotation)
                frame_summaries.append(summary)
                sampled += 1
            frame_index += 1
        capture.release()

        video_summary, report_path = self.report_service.generate_video_report(source.stem, frame_summaries)
        return {
            "video_path": self.storage.to_relpath(source),
            "frame_interval": frame_interval,
            "sampled_frames": sampled,
            "frame_reports": frame_summaries,
            "generated_annotations": generated_annotations,
            "video_report_path": self.storage.to_relpath(report_path),
            "summary": video_summary,
        }
