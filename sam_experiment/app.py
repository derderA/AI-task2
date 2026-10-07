from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename

from .annotation_service import AnnotationService
from .classification_service import ClassificationService
from .config import get_settings
from .deepseek_config_service import DeepSeekConfigService
from .deepseek_report_service import DeepSeekReportService
from .report_service import ReportService
from .sam_service import SamExperimentService
from .shp_annotation_service import ShpAnnotationService
from .storage import Storage
from .video_service import VideoAnalysisService


def create_app() -> Flask:
    settings = get_settings()
    storage = Storage(settings.base_dir)
    annotation_service = AnnotationService(settings, storage)
    sam_service = SamExperimentService(settings, storage, annotation_service)
    report_service = ReportService(settings, storage)
    deepseek_config_service = DeepSeekConfigService(settings)
    deepseek_report_service = DeepSeekReportService(settings, storage, report_service, deepseek_config_service)
    classification_service = ClassificationService(settings, storage, deepseek_config_service)
    video_service = VideoAnalysisService(settings, storage, sam_service, report_service)
    shp_annotation_service = ShpAnnotationService(settings.shp_preview_dir, storage)

    app = Flask(
        __name__,
        template_folder=str(settings.template_dir),
        static_folder=str(settings.static_dir),
    )
    app.secret_key = "sam-experiment-dev"
    settings.upload_image_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_video_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_shp_dir.mkdir(parents=True, exist_ok=True)

    allowed_image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    allowed_video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
    allowed_shp_suffixes = {".shp", ".dbf", ".shx", ".prj", ".cpg", ".qpj"}

    def load_annotation_or_none(annotation_path: Optional[str]) -> Optional[dict]:
        if not annotation_path:
            return None
        try:
            return classification_service.decorate_annotation(annotation_service.load_annotation(annotation_path))
        except FileNotFoundError:
            return None

    def load_shp_preview_or_none(preview_path: Optional[str]) -> Optional[dict]:
        if not preview_path:
            return None
        try:
            return shp_annotation_service.load_preview(preview_path)
        except FileNotFoundError:
            return None

    def save_uploaded_file(file_storage, target_dir: Path, allowed_suffixes: set[str]) -> Path:
        filename = secure_filename(file_storage.filename or "")
        if not filename:
            raise ValueError("请选择文件后再提交。")
        suffix = Path(filename).suffix.lower()
        if suffix not in allowed_suffixes:
            raise ValueError(f"不支持的文件类型: {suffix or '无扩展名'}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        saved_path = target_dir / f"{timestamp}_{filename}"
        file_storage.save(saved_path)
        return saved_path

    def save_uploaded_shp_bundle(file_storages) -> Path:
        valid_files = []
        for file_storage in file_storages:
            filename = secure_filename(file_storage.filename or "")
            if not filename:
                continue
            suffix = Path(filename).suffix.lower()
            if suffix not in allowed_shp_suffixes:
                raise ValueError(f"不支持的 Shapefile 组件类型: {suffix or '无扩展名'}")
            valid_files.append((file_storage, filename))
        if not valid_files:
            raise ValueError("请至少上传一个 .shp 文件。")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        bundle_dir = settings.upload_shp_dir / timestamp
        bundle_dir.mkdir(parents=True, exist_ok=True)
        shp_path = None
        for file_storage, filename in valid_files:
            saved_path = bundle_dir / filename
            file_storage.save(saved_path)
            if saved_path.suffix.lower() == ".shp":
                shp_path = saved_path
        if shp_path is None:
            raise ValueError("上传 Shapefile 时至少需要包含 .shp 主文件。")
        return shp_path

    @app.get("/")
    def index():
        annotation_path = request.args.get("annotation", "")
        shp_preview_path = request.args.get("shp_preview", "")
        annotation = load_annotation_or_none(annotation_path)
        if annotation is None:
            latest = latest_annotation(annotation_service)
            annotation = classification_service.decorate_annotation(latest)
        shp_preview = load_shp_preview_or_none(shp_preview_path) or latest_shp_preview(shp_annotation_service)
        report_text = ""
        selected_report_type = request.args.get("report_type", "rule")
        report_field = "deepseek_report_path" if selected_report_type == "deepseek" else "report_path"
        if annotation and annotation.get(report_field):
            report_path = storage.from_relpath(annotation[report_field])
            if report_path.exists():
                report_text = report_path.read_text(encoding="utf-8")
        return render_template(
            "index.html",
            annotation=annotation,
            report_text=report_text,
            shp_preview=shp_preview,
            selected_report_type=selected_report_type,
            recent_annotations=list_recent_annotations(settings.annotation_dir, storage),
            recent_shp_previews=shp_annotation_service.list_recent_previews(),
            deepseek_env_name=settings.deepseek_api_key_env,
            deepseek_state=deepseek_config_service.build_view_state(session.get("deepseek_runtime")),
        )

    @app.post("/deepseek/config")
    def save_deepseek_config():
        current_credentials = deepseek_config_service.resolve_credentials(session.get("deepseek_runtime"))
        api_key = request.form.get("api_key", "").strip() or current_credentials.get("api_key", "")
        base_url = request.form.get("base_url", "").strip() or settings.deepseek_base_url
        model = request.form.get("model", "").strip() or settings.deepseek_model
        save_mode = request.form.get("save_mode", "session").strip() or "session"
        try:
            if save_mode == "file":
                deepseek_config_service.write_file_config(api_key, base_url, model)
                flash("DeepSeek 配置已保存到本地配置文件。", "success")
            else:
                session["deepseek_runtime"] = {
                    "api_key": api_key,
                    "base_url": base_url,
                    "model": model,
                }
                flash("DeepSeek 配置已保存到当前网页会话。", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"保存 DeepSeek 配置失败: {exc}", "error")
        return redirect(url_for("index", tab="settings"))

    @app.post("/segment")
    def segment():
        image_file = request.files.get("image_file")
        if image_file is None or not image_file.filename:
            flash("请选择待分割图像文件。", "error")
            return redirect(url_for("index", tab="segment"))
        try:
            saved_image = save_uploaded_file(image_file, settings.upload_image_dir, allowed_image_suffixes)
            annotation = sam_service.auto_segment(storage.to_relpath(saved_image))
            flash("SAM 自动分割完成。", "success")
            return redirect(url_for("index", annotation=annotation["annotation_path"], tab="annotate"))
        except Exception as exc:  # noqa: BLE001
            flash(f"自动分割失败: {exc}", "error")
            return redirect(url_for("index"))

    @app.post("/instance/classify")
    def classify_instances():
        annotation_path = request.form["annotation_path"]
        provider = request.form.get("provider", "deepseek").strip() or "deepseek"
        try:
            classification_service.classify_annotation(
                annotation_path,
                session_values=session.get("deepseek_runtime"),
                provider=provider,
            )
            sam_service.sync_annotation_artifacts(annotation_path)
            if provider == "deepseek":
                flash("实例自动分类完成，已按类别重新着色。", "success")
            else:
                flash("规则分类完成，已按类别重新着色。", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"自动分类失败: {exc}", "error")
        return redirect(url_for("index", annotation=annotation_path, tab="classify"))

    @app.post("/instance/category")
    def update_category():
        annotation_path = request.form["annotation_path"]
        instance_id = int(request.form["instance_id"])
        category = request.form["category"]
        try:
            annotation_service.update_category(annotation_path, instance_id, category)
            sam_service.sync_annotation_artifacts(annotation_path)
            flash("实例类别已更新。", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"更新类别失败: {exc}", "error")
        return redirect(url_for("index", annotation=annotation_path, tab="annotate"))

    @app.post("/instance/merge")
    def merge_instances():
        annotation_path = request.form["annotation_path"]
        instance_ids = [int(item) for item in request.form["instance_ids"].split(",") if item.strip()]
        merged_category = request.form.get("merged_category", "").strip() or None
        try:
            annotation_service.merge_instances(annotation_path, instance_ids, merged_category)
            sam_service.sync_annotation_artifacts(annotation_path)
            flash("实例合并完成。", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"合并失败: {exc}", "error")
        return redirect(url_for("index", annotation=annotation_path, tab="annotate"))

    @app.post("/instance/resegment")
    def resegment():
        annotation_path = request.form["annotation_path"]
        instance_id = int(request.form["instance_id"])
        try:
            sam_service.resegment_instance(annotation_path, instance_id)
            flash("实例重分割完成。", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"重分割失败: {exc}", "error")
        return redirect(url_for("index", annotation=annotation_path, tab="annotate"))

    @app.post("/instance/add-point")
    def add_point_instance():
        annotation_path = request.form["annotation_path"]
        x = int(float(request.form["x"]))
        y = int(float(request.form["y"]))
        category = request.form["point_category"]
        try:
            sam_service.add_prompt_instance(annotation_path, x, y, category)
            flash("基于点提示的新实例已加入。", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"点提示标注失败: {exc}", "error")
        return redirect(url_for("index", annotation=annotation_path, tab="annotate"))

    @app.post("/report")
    def generate_report():
        annotation_path = request.form["annotation_path"]
        provider = request.form.get("provider", "rule").strip() or "rule"
        try:
            annotation = annotation_service.load_annotation(annotation_path)
            if provider == "deepseek":
                _, report_path = deepseek_report_service.generate_image_report(
                    annotation,
                    session_values=session.get("deepseek_runtime"),
                )
                annotation["deepseek_report_path"] = storage.to_relpath(report_path)
                flash("DeepSeek 报告已生成。", "success")
            else:
                _, report_path = report_service.generate_image_report(annotation)
                annotation["report_path"] = storage.to_relpath(report_path)
                flash("规则报告已生成。", "success")
            annotation_service.save_annotation(annotation_path, annotation)
        except Exception as exc:  # noqa: BLE001
            flash(f"报告生成失败: {exc}", "error")
        return redirect(url_for("index", annotation=annotation_path, report_type=provider, tab="report"))

    @app.post("/video")
    def analyze_video():
        video_file = request.files.get("video_file")
        frame_interval = int(request.form.get("frame_interval", 30))
        max_frames = int(request.form.get("max_frames", 5))
        if video_file is None or not video_file.filename:
            flash("请选择待分析视频文件。", "error")
            return redirect(url_for("index", tab="video"))
        try:
            saved_video = save_uploaded_file(video_file, settings.upload_video_dir, allowed_video_suffixes)
            result = video_service.analyze_video(
                storage.to_relpath(saved_video),
                frame_interval=frame_interval,
                max_frames=max_frames,
            )
            flash(f"视频分析完成，抽帧 {result['sampled_frames']} 帧。", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"视频分析失败: {exc}", "error")
        return redirect(url_for("index", tab="video"))

    @app.post("/shp/preview")
    def upload_shp_preview():
        shp_files = request.files.getlist("shp_files")
        if not shp_files:
            flash("请上传 .shp 标注文件及其配套的 .dbf/.shx 文件。", "error")
            return redirect(url_for("index", tab="shp"))
        try:
            shp_path = save_uploaded_shp_bundle(shp_files)
            preview = shp_annotation_service.build_preview(shp_path)
            flash("Shapefile 标注预览已生成。", "success")
            return redirect(url_for("index", shp_preview=preview["preview_path"], tab="shp"))
        except Exception as exc:  # noqa: BLE001
            flash(f"生成 Shapefile 预览失败: {exc}", "error")
            return redirect(url_for("index"))

    @app.get("/artifact/<path:relative_path>")
    def artifact(relative_path: str):
        target = storage.from_relpath(relative_path).resolve()
        if settings.base_dir not in target.parents and target != settings.base_dir:
            raise FileNotFoundError(relative_path)
        return send_file(target)

    return app


def latest_annotation(annotation_service: AnnotationService) -> Optional[dict]:
    files = sorted(annotation_service.settings.annotation_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return annotation_service.load_annotation(files[0])


def list_recent_annotations(annotation_dir: Path, storage: Storage, limit: int = 8) -> list[str]:
    files = sorted(annotation_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return [storage.to_relpath(file_path) for file_path in files]


def latest_shp_preview(shp_annotation_service: ShpAnnotationService) -> Optional[dict]:
    previews = shp_annotation_service.list_recent_previews(limit=1)
    if not previews:
        return None
    return shp_annotation_service.load_preview(previews[0])
