"""Detect (and optionally repair) UTF-8-read-as-CP1252 mojibake.

    python tools/fix_mojibake.py           # report only
    python tools/fix_mojibake.py --apply   # repair in place
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
apply = "--apply" in sys.argv

for path in sorted(ROOT.rglob("*.py")) + sorted(ROOT.rglob("*.md")):
    if ".venv" in path.parts:
        continue
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        print(f"{path.relative_to(ROOT)}: NOT valid UTF-8")
        continue

    # Windows PowerShell's `Set-Content -Encoding utf8` prepends a BOM. Strip it
    # first: it is both unwanted and it blocks the mojibake reversal below.
    had_bom = text.startswith("﻿")
    text = text.lstrip("﻿")

    # Mojibake reversal: the text was decoded as cp1252 and re-encoded as utf-8.
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        repaired = text

    if had_bom and repaired == text:
        print(f"{path.relative_to(ROOT)}: BOM only")
        if apply:
            path.write_text(text, encoding="utf-8")
            print("   BOM stripped")
        continue

    if repaired != text:
        bad = sorted({c for c in text if ord(c) > 127})
        good = sorted({c for c in repaired if ord(c) > 127})
        print(f"{path.relative_to(ROOT)}: MOJIBAKE  {bad} -> {good}")
        if apply:
            path.write_text(repaired, encoding="utf-8")
            print("   repaired")
    else:
        nonascii = sorted({c for c in text if ord(c) > 127})
        status = f"clean (non-ascii: {nonascii})" if nonascii else "clean (pure ascii)"
        print(f"{path.relative_to(ROOT)}: {status}")
