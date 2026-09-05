#!/usr/bin/env python3
"""Validate the structure and evidence metadata of the literature corpus."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB_PATH = ROOT / "paper" / "references.bib"
README_PATH = ROOT / "README.md"
NOTES_PATH = ROOT / "notes"


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []
    bibliography = BIB_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")
    blocks = re.findall(
        r"(?ms)^@(?P<kind>\w+)\{(?P<key>[^,]+),(?P<body>.*?)(?=^@|\Z)",
        bibliography,
    )

    if not blocks:
        fail("bibliography contains no parseable entries", errors)

    keys = [key.strip() for _, key, _ in blocks]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        fail(f"duplicate BibTeX keys: {', '.join(duplicate_keys)}", errors)

    arxiv_ids: list[str] = []
    non_arxiv = 0
    for _, key, body in blocks:
        eprint = re.search(r"(?m)^\s*eprint\s*=\s*\{([^}]+)\}", body)
        archive = re.search(r"(?mi)^\s*archivePrefix\s*=\s*\{arXiv\}", body)
        if not archive:
            non_arxiv += 1
            continue
        if not eprint:
            fail(f"arXiv entry {key.strip()} has no eprint", errors)
            continue
        arxiv_id = eprint.group(1).strip()
        arxiv_ids.append(arxiv_id)
        expected_url = f"https://arxiv.org/abs/{arxiv_id}"
        if expected_url not in body:
            fail(f"arXiv entry {key.strip()} has no canonical URL", errors)

    duplicate_ids = sorted(
        {arxiv_id for arxiv_id in arxiv_ids if arxiv_ids.count(arxiv_id) > 1}
    )
    if duplicate_ids:
        fail(f"duplicate arXiv identifiers: {', '.join(duplicate_ids)}", errors)

    notes = sorted(path.name for path in NOTES_PATH.glob("*.md"))
    for note in notes:
        if f"`notes/{note}`" not in readme:
            fail(f"README index is missing notes/{note}", errors)

    expected_summary = (
        f"{len(blocks)} BibTeX entries: {len(arxiv_ids)} arXiv records plus "
        f"one\n  non-arXiv web reference"
    )
    if non_arxiv != 1:
        fail(f"expected one non-arXiv reference, found {non_arxiv}", errors)
    if expected_summary not in readme:
        fail("README bibliography summary is stale", errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(
        f"OK: {len(notes)} notes, {len(blocks)} references "
        f"({len(arxiv_ids)} arXiv, {non_arxiv} non-arXiv), no duplicate ids"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
