"""
Lightweight reporting utilities for snapshots (JSON) and streams (JSONL) with day rotation.

Directory layout (under base_dir):
  - snapshots/: small JSON files
  - jsonl/: append-only streams with daily rotation and alias symlink
  - meta/: small metadata files (schema.json, last_update.json)

All timestamps are ISO8601 UTC (Z).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(target_path: Path, data: Dict[str, Any]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, target_path)


@dataclass
class ReportWriter:
    base_dir: Path

    def __post_init__(self) -> None:
        self.snapshots_dir = self.base_dir / "snapshots"
        self.jsonl_dir = self.base_dir / "jsonl"
        self.meta_dir = self.base_dir / "meta"

    def ensure_paths(self) -> None:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def write_schema_once(self, version: str = "1.0", currency: str = "EUR", tz: str = "UTC") -> None:
        schema_path = self.meta_dir / "schema.json"
        if schema_path.exists():
            return
        data = {"version": version, "currency": currency, "tz": tz, "ts": utc_now_iso()}
        atomic_write_json(schema_path, data)

    def update_last_update(self, file_rel_path: str) -> None:
        """Update meta/last_update.json with latest timestamp per file."""
        target = self.meta_dir / "last_update.json"
        content: Dict[str, Any] = {"ts": utc_now_iso(), "files": {}}
        if target.exists():
            try:
                content = json.loads(target.read_text(encoding="utf-8"))
            except Exception:
                content = {"ts": utc_now_iso(), "files": {}}
        content["ts"] = utc_now_iso()
        files = content.get("files", {})
        files[file_rel_path] = utc_now_iso()
        content["files"] = files
        atomic_write_json(target, content)

    # Snapshots
    def write_snapshot(self, name: str, payload: Dict[str, Any]) -> None:
        path = self.snapshots_dir / name
        atomic_write_json(path, payload)
        rel = str(path.relative_to(self.base_dir))
        self.update_last_update(rel)

    # JSONL streams with daily rotation
    def _day_file_for(self, stream: str, day: str) -> Path:
        return self.jsonl_dir / f"{stream}-{day}.jsonl"

    def _alias_path_for(self, stream: str) -> Path:
        return self.jsonl_dir / f"{stream}.jsonl"

    def _ensure_alias(self, stream: str, day_file: Path) -> None:
        alias = self._alias_path_for(stream)
        try:
            if alias.is_symlink() or alias.exists():
                current = alias.resolve(strict=False)
                if current != day_file:
                    alias.unlink(missing_ok=True)
                    alias.symlink_to(day_file.name)
            else:
                alias.symlink_to(day_file.name)
        except Exception:
            # If symlink fails (FS permissions), fall back to copying on first append
            if not alias.exists():
                try:
                    alias.write_text("", encoding="utf-8")
                except Exception:
                    pass

    def append_jsonl(self, stream: str, obj: Dict[str, Any]) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_file = self._day_file_for(stream, day)
        day_file.parent.mkdir(parents=True, exist_ok=True)
        # Ensure alias
        self._ensure_alias(stream, day_file)
        # Append one JSON object per line
        with day_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
        rel = str((self.jsonl_dir / f"{stream}.jsonl").relative_to(self.base_dir))
        self.update_last_update(rel)




