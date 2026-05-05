#!/usr/bin/env python3
"""
Build an instruction fine-tuning dataset from DST articles + StatBank tables.

Each example:
  user:      Danish instruction + StatBank tables formatted as markdown
  assistant: The actual published article text

Usage:
    uv run build_finetune.py
    uv run build_finetune.py --push oliverkinch/dst-finetuning
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

PROGRESS_FILE = Path(__file__).parent / "downloads" / "progress.jsonl"
CACHE_FILE = Path(__file__).parent / "downloads" / "statbank_cache.jsonl"

MIN_TEXT_LEN = 100


# ---------------------------------------------------------------------------
# Data loading (shared with push_tables.py)
# ---------------------------------------------------------------------------

def load_articles() -> dict[str, dict]:
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


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _pivot_rows(rows: list[dict]) -> tuple[list[str], list[list[str]]]:
    """
    Pivot StatBank rows (each has dimension cols + TID + INDHOLD) into a
    2-D markdown table.

    Dimension columns become row keys, TID values become column headers.
    Returns (headers, data_rows) where headers = [dim..., period1, period2, ...]
    """
    if not rows:
        return [], []

    dim_cols = [k for k in rows[0] if k not in ("TID", "INDHOLD")]
    periods = sorted({r["TID"] for r in rows})

    # Index: (dim_values_tuple) -> {TID: INDHOLD}
    index: dict[tuple, dict[str, str]] = {}
    for r in rows:
        key = tuple(r.get(d, "") for d in dim_cols)
        index.setdefault(key, {})[r["TID"]] = r.get("INDHOLD", "")

    headers = dim_cols + periods
    data_rows = [
        list(key) + [vals.get(p, "") for p in periods]
        for key, vals in index.items()
    ]
    return headers, data_rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""
    col_widths = [
        max(len(h), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    data_lines = [
        "| " + " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)) + " |"
        for row in rows
    ]
    return "\n".join([header_row, sep] + data_lines)


def format_table(table_code: str, table: dict) -> str:
    title = table.get("table_title") or table_code
    unit = table.get("unit", "")
    rows = table.get("rows", [])

    heading = f"## {table_code} – {title}"
    if unit:
        heading += f" ({unit})"

    headers, pivoted = _pivot_rows(rows)
    if not headers:
        return heading

    return heading + "\n\n" + _markdown_table(headers, pivoted)


def build_user_prompt(article: dict, statbank_data: dict) -> str:
    series = article.get("series")
    if series:
        instruction = (
            f"Du er statistiker hos Danmarks Statistik. "
            f"Skriv en nyhedsartikel til serien '{series}' "
            f"baseret på følgende statistiske data:"
        )
    else:
        instruction = (
            "Du er statistiker hos Danmarks Statistik. "
            "Skriv en nyhedsartikel baseret på følgende statistiske data:"
        )

    table_blocks = [
        format_table(code, tbl)
        for code, tbl in statbank_data.items()
    ]
    return instruction + "\n\n" + "\n\n".join(table_blocks)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_rows(articles: dict[str, dict], statbank_cache: dict[str, dict]) -> list[dict]:
    rows = []
    skipped_no_tables = 0
    skipped_short_text = 0

    for url, article in articles.items():
        cached = statbank_cache.get(url)
        if not cached or not cached.get("statbank_data"):
            skipped_no_tables += 1
            continue

        text = article.get("text", "")
        if len(text) < MIN_TEXT_LEN:
            skipped_short_text += 1
            continue

        user_content = build_user_prompt(article, cached["statbank_data"])
        rows.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": text},
            ]
        })

    print(f"  Skipped (no tables): {skipped_no_tables}")
    print(f"  Skipped (text too short): {skipped_short_text}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", metavar="REPO_ID", help="Push to HuggingFace Hub")
    parser.add_argument("--out", default="dataset_finetune", help="Local output directory")
    args = parser.parse_args()

    print("Loading articles...")
    articles = load_articles()
    print(f"  {len(articles)} articles with status=ok")

    print("Loading StatBank cache...")
    statbank_cache = load_statbank_cache()
    print(f"  {len(statbank_cache)} cached entries")

    print("Building fine-tuning rows...")
    rows = build_rows(articles, statbank_cache)
    print(f"  {len(rows)} examples")

    if not rows:
        print("No rows produced. Exiting.")
        return

    # Show a sample
    print("\n--- Sample user prompt (first 800 chars) ---")
    print(rows[0]["messages"][0]["content"][:800])
    print("\n--- Sample assistant response (first 300 chars) ---")
    print(rows[0]["messages"][1]["content"][:300])
    print("---\n")

    ds = Dataset.from_list(rows)
    dsd = DatasetDict({"train": ds})
    print(dsd)

    out_dir = Path(args.out)
    dsd.save_to_disk(str(out_dir))
    print(f"Saved to {out_dir}/")

    if args.push:
        print(f"\nPushing to {args.push}...")
        dsd.push_to_hub(
            args.push,
            commit_message="Add DST instruction fine-tuning dataset (CC BY 4.0)",
        )
        print("Done!")


if __name__ == "__main__":
    main()
