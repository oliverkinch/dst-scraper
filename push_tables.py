#!/usr/bin/env python3
"""
Build and push the dst-tables HuggingFace dataset.

One row per article, joining:
  - downloads/progress.jsonl          → article metadata + prose text
  - downloads/statbank_cache.jsonl    → StatBank API table data
  - downloads/article_tables_cache.jsonl → inline HTML tables as markdown

Usage:
    uv run push_tables.py
    uv run push_tables.py --push oliverkinch/dst-tables
"""

import argparse
import json
import re
from pathlib import Path

from datasets import Dataset, DatasetDict

PROGRESS_FILE = Path(__file__).parent / "downloads" / "progress.jsonl"
CACHE_FILE = Path(__file__).parent / "downloads" / "statbank_cache_v2.jsonl"
ARTICLE_TABLES_FILE = Path(__file__).parent / "downloads" / "article_tables_cache.jsonl"

DANISH_MONTHS = {
    "januar": "01", "februar": "02", "marts": "03", "april": "04",
    "maj": "05", "juni": "06", "juli": "07", "august": "08",
    "september": "09", "oktober": "10", "november": "11", "december": "12",
}


def parse_danish_date(raw: str | None) -> str | None:
    """Convert '22. april 2026' → '2026-04-22'. Returns None if unparseable."""
    if not raw:
        return None
    m = re.match(r"(\d{1,2})\.\s+(\w+)\s+(\d{4})", raw.strip())
    if not m:
        return None
    day, month_name, year = m.groups()
    month = DANISH_MONTHS.get(month_name.lower())
    if not month:
        return None
    return f"{year}-{month}-{int(day):02d}"


def load_articles() -> dict[str, dict]:
    if not PROGRESS_FILE.exists():
        raise FileNotFoundError(f"No progress file at {PROGRESS_FILE}. Run scraper.py first.")
    articles: dict[str, dict] = {}
    for line in PROGRESS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        articles[r["url"]] = r
    return articles


def load_statbank_cache() -> dict[str, dict]:
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


def load_article_tables_cache() -> dict[str, dict]:
    if not ARTICLE_TABLES_FILE.exists():
        return {}
    cache: dict[str, dict] = {}
    for line in ARTICLE_TABLES_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            cache[r["url"]] = r
        except Exception:
            pass
    return cache


def build_rows(articles: dict[str, dict], statbank_cache: dict[str, dict], article_tables_cache: dict[str, dict]) -> list[dict]:
    rows = []
    article_id = 0
    for url, article in articles.items():
        cached = statbank_cache.get(url)
        if not cached or not cached.get("statbank_data"):
            continue

        at = article_tables_cache.get(url, {})
        inline_tables_md = at.get("inline_tables_md") or ""

        # Only include articles that have at least one inline table
        if not inline_tables_md:
            continue

        date_raw = article.get("date")
        date_iso = parse_danish_date(date_raw)

        # Build one list entry per table, preserving all table fields
        tables = []
        for table_code, table in cached["statbank_data"].items():
            tables.append({
                "table_code": table_code,
                "table_title": table.get("table_title"),
                "unit": table.get("unit"),
                "periods": table.get("periods", []),
                "table_rows": json.dumps(table.get("rows", []), ensure_ascii=False),
            })

        text = article.get("text") or ""
        full_article = (text + "\n\n" + inline_tables_md).strip()

        rows.append({
            "article_id": str(article_id),
            "url": url,
            "content_type": article.get("content_type"),
            "title": article.get("title"),
            "date": date_iso,
            "date_raw": date_raw,
            "series": article.get("series"),
            "series_url": article.get("series_url"),
            "text": text,
            "full_article": full_article,
            "inline_tables_md": at.get("inline_tables_md") or None,
            "noegletal_text": at.get("noegletal_text") or None,
            "license": article.get("license", "CC BY 4.0"),
            "source": article.get("source", "Danmarks Statistik"),
            "tables": tables,
        })
        article_id += 1

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", metavar="REPO_ID", help="Push to HuggingFace Hub")
    parser.add_argument("--out", default="dataset_tables", help="Local output directory")
    args = parser.parse_args()

    print("Loading articles...")
    articles = load_articles()
    print(f"  {len(articles)} articles with status=ok")

    print("Loading StatBank cache...")
    statbank_cache = load_statbank_cache()
    print(f"  {len(statbank_cache)} cached entries")

    print("Loading article tables cache...")
    article_tables_cache = load_article_tables_cache()
    print(f"  {len(article_tables_cache)} entries with inline tables")

    print("Building rows...")
    rows = build_rows(articles, statbank_cache, article_tables_cache)
    print(f"  {len(rows)} rows")

    if not rows:
        print("No rows to build. Exiting.")
        return

    ds = Dataset.from_list(rows)
    dsd = DatasetDict({"train": ds})
    print(dsd)
    print(ds.features)

    out_dir = Path(args.out)
    dsd.save_to_disk(str(out_dir))
    print(f"\nSaved to {out_dir}/")

    if args.push:
        print(f"\nPushing to {args.push}...")
        dsd.push_to_hub(
            args.push,
            commit_message="Add DST tables dataset (CC BY 4.0)",
        )
        print("Done!")


if __name__ == "__main__":
    main()
