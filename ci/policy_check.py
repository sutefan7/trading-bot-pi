#!/usr/bin/env python3
"""Validate trading-bot-pi requirements against deny-listed train deps."""

from __future__ import annotations

import pathlib
import re
import sys


DENY = re.compile(r"^(scikit-learn|xgboost|lightgbm|torch|jax|numba|tensorflow|cupy)\b", re.IGNORECASE)


def main() -> int:
    bad: list[tuple[str, str]] = []
    for req in pathlib.Path(".").rglob("requirements*.txt"):
        try:
            lines = req.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            continue

        for raw in lines:
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue

            name = (
                line.split("==", 1)[0]
                .split(">=", 1)[0]
                .split("<=", 1)[0]
                .split("[", 1)[0]
                .strip()
            )

            if DENY.match(name):
                bad.append((req.as_posix(), name))

    if bad:
        for file_path, dep in bad:
            print(f"[FAIL] Forbidden on Pi: {dep} found in {file_path}", file=sys.stderr)
        return 1

    print("[OK] Pi requirements are lightweight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

