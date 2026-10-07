from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from sam_experiment.config import Settings


class TempProject:
    def __init__(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for rel in [
            "asset",
            "config",
            "dataset",
            "data/annotations",
            "data/reports",
            "data/exports",
            "data/uploads/images",
            "data/uploads/shp",
            "data/uploads/videos",
            "data/shp_previews",
            "data/video_runs",
            "sam_experiment/templates",
            "sam_experiment/static",
        ]:
            (self.root / rel).mkdir(parents=True, exist_ok=True)
        (self.root / "asset" / "sam_vit_b_01ec64.pth").write_bytes(b"placeholder")

    def settings(self) -> Settings:
        return Settings(
            base_dir=self.root,
            asset_dir=self.root / "asset",
            dataset_dir=self.root / "dataset",
            data_dir=self.root / "data",
            config_dir=self.root / "config",
            annotation_dir=self.root / "data" / "annotations",
            report_dir=self.root / "data" / "reports",
            export_dir=self.root / "data" / "exports",
            video_dir=self.root / "data" / "video_runs",
            upload_dir=self.root / "data" / "uploads",
            upload_image_dir=self.root / "data" / "uploads" / "images",
            upload_shp_dir=self.root / "data" / "uploads" / "shp",
            upload_video_dir=self.root / "data" / "uploads" / "videos",
            shp_preview_dir=self.root / "data" / "shp_previews",
            template_dir=self.root / "sam_experiment" / "templates",
            static_dir=self.root / "sam_experiment" / "static",
            checkpoint_path=self.root / "asset" / "sam_vit_b_01ec64.pth",
            deepseek_base_url="https://api.deepseek.com",
            deepseek_model="deepseek-chat",
            deepseek_api_key_env="DEEPSEEK_API_KEY",
            deepseek_config_path=self.root / "config" / "deepseek_config.json",
        )

    def cleanup(self) -> None:
        self._tmp.cleanup()
