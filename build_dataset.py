#!/usr/bin/env python3
"""
Build a HuggingFace dataset from the scraped DST articles.

Reads downloads/progress.jsonl and produces a dataset with:
  - id, url, content_type, title, date, series, text, license, source
  - statbank_tables, statbank_data  (if enrich_statbank.py has been run)

Usage:
    uv run build_dataset.py
    uv run build_dataset.py --push dfm/dst-publications
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

PROGRESS_FILE = Path(__file__).parent / "downloads" / "progress.jsonl"
CACHE_FILE = Path(__file__).parent / "downloads" / "statbank_cache.jsonl"

FIELDS = [
    "id",
    "url",
    "content_type",
    "title",
    "date",
    "series",
    "series_url",
    "text",
    "status",
]


def load_records() -> list[dict]:
    if not PROGRESS_FILE.exists():
        raise FileNotFoundError(f"No progress file at {PROGRESS_FILE}. Run scraper.py first.")

    records = []
    for line in PROGRESS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        records.append(r)
    return records


def load_statbank_cache() -> dict[str, dict]:
    """Load statbank_cache.jsonl keyed by article URL. Returns empty dict if file absent."""
    if not CACHE_FILE.exists():
        return {}
    cache: dict[str, dict] = {}
    for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            cache[r["url"]] = r
        except Exception:
            pass
    return cache


def build_dataset(records: list[dict], statbank_cache: dict[str, dict]) -> DatasetDict:
    rows = []
    for i, r in enumerate(records):
        row = {field: r.get(field) for field in FIELDS}
        row["id"] = str(i)

        cached = statbank_cache.get(r.get("url", ""))
        row["statbank_tables"] = cached.get("statbank_tables", []) if cached else []
        row["statbank_data"] = (
            json.dumps(cached["statbank_data"], ensure_ascii=False)
            if cached and cached.get("statbank_data")
            else None
        )

        rows.append(row)

    ds = Dataset.from_list(rows)
    return DatasetDict({"train": ds})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--push",
        metavar="REPO_ID",
        help="Push to HuggingFace Hub (e.g. dfm/dst-publications)",
    )
    parser.add_argument(
        "--out",
        default="dataset",
        help="Local output directory for saved dataset (default: ./dataset)",
    )
    args = parser.parse_args()

    print("Loading records...")
    records = load_records()
    print(f"  {len(records)} articles with status=ok")

    # Stats
    by_type: dict[str, int] = {}
    for r in records:
        t = r.get("content_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")

    print("\nLoading StatBank cache...")
    statbank_cache = load_statbank_cache()
    with_data = sum(1 for r in records if r.get("url") in statbank_cache and statbank_cache[r["url"]].get("statbank_data"))
    print(f"  {len(statbank_cache)} cached, {with_data} with table data")

    print("\nBuilding dataset...")
    dsd = build_dataset(records, statbank_cache)
    print(dsd)

    out_dir = Path(args.out)
    dsd.save_to_disk(str(out_dir))
    print(f"\nSaved to {out_dir}/")

    if args.push:
        print(f"\nPushing to {args.push}...")
        dsd.push_to_hub(
            args.push,
            commit_message="Add DST publications dataset (CC BY 4.0)",
        )
        print("Done!")


if __name__ == "__main__":
    main()
