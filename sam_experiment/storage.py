from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union


class Storage:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()

    def ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def to_relpath(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.base_dir))
        except ValueError:
            return str(path)

    def from_relpath(self, value: Union[str, Path]) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.base_dir / path)

    def read_json(self, path: Union[str, Path]) -> dict[str, Any]:
        file_path = self.from_relpath(path)
        with file_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def write_json(self, path: Union[str, Path], payload: dict[str, Any]) -> Path:
        file_path = self.from_relpath(path)
        self.ensure_parent(file_path)
        with file_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        return file_path
