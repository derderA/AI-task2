from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    asset_dir: Path
    dataset_dir: Path
    data_dir: Path
    annotation_dir: Path
    report_dir: Path
    export_dir: Path
    video_dir: Path
    upload_dir: Path
    upload_image_dir: Path
    upload_video_dir: Path
    upload_shp_dir: Path
    shp_preview_dir: Path
    template_dir: Path
    static_dir: Path
    checkpoint_path: Path
    config_dir: Path
    default_model_type: str = "vit_b"
    default_category: str = "未分类"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"
    deepseek_config_path: Optional[Path] = None

    @property
    def preferred_image_roots(self) -> List[Path]:
        return [
            self.dataset_dir / "DOTA_v1.0" / "images" / "val",
            self.dataset_dir / "DOTA_v1.0" / "images" / "train",
            self.dataset_dir / "DOTA_v1.0" / "val" / "val" / "images",
            self.dataset_dir / "DOTA_v1.0" / "train" / "train" / "images",
            self.dataset_dir / "iSAID" / "test_images" / "tiles",
            self.dataset_dir / "iSAID" / "test_images" / "originals",
        ]


def get_settings(base_dir: Optional[Union[str, Path]] = None) -> Settings:
    root = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    config_dir = root / "config"
    upload_dir = data_dir / "uploads"
    return Settings(
        base_dir=root,
        asset_dir=root / "asset",
        dataset_dir=root / "dataset",
        data_dir=data_dir,
        config_dir=config_dir,
        annotation_dir=data_dir / "annotations",
        report_dir=data_dir / "reports",
        export_dir=data_dir / "exports",
        video_dir=data_dir / "video_runs",
        upload_dir=upload_dir,
        upload_image_dir=upload_dir / "images",
        upload_video_dir=upload_dir / "videos",
        upload_shp_dir=upload_dir / "shp",
        shp_preview_dir=data_dir / "shp_previews",
        template_dir=root / "sam_experiment" / "templates",
        static_dir=root / "sam_experiment" / "static",
        checkpoint_path=root / "asset" / "sam_vit_b_01ec64.pth",
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_api_key_env=os.getenv("DEEPSEEK_API_KEY_ENV", "DEEPSEEK_API_KEY"),
        deepseek_config_path=config_dir / "deepseek_config.json",
    )
