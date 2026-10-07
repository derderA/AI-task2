from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .config import Settings
from .deepseek_config_service import DeepSeekConfigService
from .report_service import ReportService
from .storage import Storage


class DeepSeekReportService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        report_service: ReportService,
        config_service: DeepSeekConfigService,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.report_service = report_service
        self.config_service = config_service

    def generate_image_report(self, annotation: dict, session_values: Optional[dict] = None) -> tuple[dict, Path]:
        summary = self.report_service.summarize_image_annotation(annotation)
        prompt = self.build_image_prompt(annotation, summary)
        credentials = self.config_service.resolve_credentials(session_values)
        content = self._request_markdown(prompt, credentials)
        report_path = self.settings.report_dir / f"{Path(annotation['image_path']).stem}_deepseek_report.md"
        report_path.write_text(content, encoding="utf-8")
        payload = {
            "provider": "deepseek",
            "model": credentials["model"],
            "source": credentials["source"],
            "prompt": prompt,
            "structured_summary": summary,
            "report_path": self.storage.to_relpath(report_path),
        }
        annotation["deepseek_report_path"] = payload["report_path"]
        return payload, report_path

    def generate_video_report(
        self,
        video_name: str,
        frame_summaries: list[dict],
        session_values: Optional[dict] = None,
    ) -> tuple[dict, Path]:
        summary = self.report_service.summarize_video_frames(video_name, frame_summaries)
        prompt = self.build_video_prompt(summary, frame_summaries)
        credentials = self.config_service.resolve_credentials(session_values)
        content = self._request_markdown(prompt, credentials)
        report_path = self.settings.video_dir / video_name / "video_deepseek_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(content, encoding="utf-8")
        payload = {
            "provider": "deepseek",
            "model": credentials["model"],
            "source": credentials["source"],
            "prompt": prompt,
            "structured_summary": summary,
            "report_path": self.storage.to_relpath(report_path),
        }
        return payload, report_path

    def build_image_prompt(self, annotation: dict, summary: dict) -> str:
        instance_table = [
            {
                "id": item["id"],
                "category": item["category"],
                "area": item["area"],
                "bbox": item["bbox"],
                "center": item["center"],
            }
            for item in annotation["instances"]
        ]
        return (
            "你是一名遥感图像分析助手。请基于给定的 SAM 实例分割统计结果，"
            "生成一份中文 Markdown 报告。要求结论必须紧扣证据，不得编造图像中未出现的信息。\n\n"
            "输出格式要求：\n"
            "1. 标题为“# DeepSeek 图像分析报告”\n"
            "2. 必须包含以下小节：## 图像概况、## 目标统计、## 场景判读、## 结论、## 证据链、## 局限性\n"
            "3. 在“证据链”中用项目符号明确写出“观察 -> 推断”\n"
            "4. 若未分类实例较多，需要明确提醒人工复核\n\n"
            f"图像路径：{annotation['image_path']}\n"
            f"结构化摘要：\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
            f"实例明细：\n{json.dumps(instance_table, ensure_ascii=False, indent=2)}\n"
        )

    def build_video_prompt(self, summary: dict, frame_summaries: list[dict]) -> str:
        return (
            "你是一名遥感视频分析助手。请根据多帧 SAM 分割统计结果生成中文 Markdown 报告，"
            "报告要体现时序抽帧汇总特征，结论必须引用统计证据。\n\n"
            "输出格式要求：\n"
            "1. 标题为“# DeepSeek 视频分析报告”\n"
            "2. 必须包含以下小节：## 视频概况、## 汇总统计、## 场景判读、## 结论、## 证据链、## 局限性\n"
            "3. 若各帧差异明显，需要指出时序不稳定性\n\n"
            f"汇总摘要：\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
            f"逐帧摘要：\n{json.dumps(frame_summaries, ensure_ascii=False, indent=2)}\n"
        )

    def _request_markdown(self, prompt: str, credentials: dict) -> str:
        api_key = credentials.get("api_key", "").strip()
        if not api_key:
            raise RuntimeError(
                "未检测到 DeepSeek API Key，请先在网页中输入、配置文件中保存，"
                f"或设置环境变量 {self.settings.deepseek_api_key_env}。"
            )
        client = OpenAI(api_key=api_key, base_url=credentials.get("base_url", self.settings.deepseek_base_url))
        response = client.chat.completions.create(
            model=credentials.get("model", self.settings.deepseek_model),
            messages=[
                {
                    "role": "system",
                    "content": "你是一名严谨的遥感图像解译助手，必须依据统计证据生成中文 Markdown 报告。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("DeepSeek 返回为空，未生成报告内容。")
        return content
