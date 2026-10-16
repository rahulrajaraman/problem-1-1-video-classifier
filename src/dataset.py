"""Load Video-MME split jsonl files."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_splits(split_dir: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, list[dict]]:
    """Return {split_name: rows}. Each row keeps its fields from the jsonl."""
    split_dir = Path(split_dir)
    return {name: read_jsonl(split_dir / f"{name}.jsonl") for name in splits}


def all_rows(splits: dict[str, list[dict]]) -> list[dict]:
    """Flatten splits into one list, tagging each row with `split`."""
    return [{**row, "split": name} for name, rows in splits.items() for row in rows]


def domain_names(rows: list[dict]) -> list[str]:
    return sorted({row["domain"] for row in rows})
