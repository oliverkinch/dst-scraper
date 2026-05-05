#!/usr/bin/env python3
"""
Re-fetch DST article pages and extract inline statistical tables as markdown.

Reads downloads/progress.jsonl for article URLs, fetches each page,
extracts <table class="TabelSmal"> elements (skipping footnote tables),
and stores results in downloads/article_tables_cache.jsonl.

The script is resumable: URLs already in the cache are skipped.

Usage:
    uv run extract_article_tables.py
    uv run extract_article_tables.py --limit 10
"""

import argparse
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

DOWNLOADS_DIR = Path(__file__).parent / "downloads"
PROGRESS_FILE = DOWNLOADS_DIR / "progress.jsonl"
CACHE_FILE = DOWNLOADS_DIR / "article_tables_cache.jsonl"

REQUEST_DELAY = 0.5

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "Mozilla/5.0 (research bot; Danish Foundation Models; "
    "https://github.com/centre-for-humanities-computing)"
)
SESSION.headers["Accept-Language"] = "da,en;q=0.9"

NOEGLETAL_SELECTORS = [
    ".noegle_boks_ydre_container",
    ".nyt_noegletalsboks_container",
    "#NoegletalsBoksVenstre",
    "#NoegletalsBoksHoejre",
]


# ---------------------------------------------------------------------------
# HTML table → markdown
# ---------------------------------------------------------------------------

def _cell_text(cell: Tag) -> str:
    return " ".join(cell.get_text(separator=" ").split())


def _expand_row(row: Tag) -> list[tuple[str, int]]:
    """Return list of (cell_text, colspan) for each cell in a row."""
    cells = []
    for td in row.find_all(["td", "th"]):
        try:
            span = int(td.get("colspan", 1))
        except (ValueError, TypeError):
            span = 1
        cells.append((_cell_text(td), span))
    return cells


def html_table_to_markdown(table: Tag) -> str:
    """Convert a <table> element to a markdown table string.

    Handles multi-row headers with colspan by merging them:
    a year header spanning 3 months becomes "2025 jan" / "2025 feb" / "2025 mar".
    """
    rows = table.find_all("tr")
    if not rows:
        return ""

    # Separate header rows (TabelTop / TabelAdskiller) from data rows
    header_rows = []
    data_rows = []
    in_header = True
    for row in rows:
        cls = " ".join(row.get("class", []))
        if in_header and ("TabelTop" in cls or "TabelAdskiller" in cls or not row.find_all(["td", "th"])):
            header_rows.append(row)
        else:
            in_header = False
            if row.find_all(["td", "th"]):
                data_rows.append(row)

    # Build flat column headers by expanding colspan across header rows
    # Strategy: for each header row, expand cells left-to-right using colspan,
    # then combine multi-row headers by joining non-empty strings with " ".
    if header_rows:
        # Find max columns
        max_cols = max(
            sum(span for _, span in _expand_row(r))
            for r in header_rows
            if _expand_row(r)
        ) if header_rows else 0

        col_labels: list[list[str]] = [[] for _ in range(max_cols)]
        for row in header_rows:
            expanded = _expand_row(row)
            col_idx = 0
            for text, span in expanded:
                for offset in range(span):
                    if col_idx + offset < max_cols:
                        if text:
                            col_labels[col_idx + offset].append(text)
                col_idx += span

        headers = [" ".join(parts) if parts else "" for parts in col_labels]
    else:
        # Infer column count from first data row
        if not data_rows:
            return ""
        n = sum(span for _, span in _expand_row(data_rows[0]))
        headers = [""] * n

    if not headers:
        return ""

    # Build markdown
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    header_line = "| " + " | ".join(h.replace("|", "/") for h in headers) + " |"

    data_lines = []
    for row in data_rows:
        cells_raw = _expand_row(row)
        cells: list[str] = []
        for text, span in cells_raw:
            cells.append(text.replace("|", "/"))
            for _ in range(span - 1):
                cells.append("")
        # Pad or trim to header width
        while len(cells) < len(headers):
            cells.append("")
        cells = cells[: len(headers)]
        data_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_line, sep] + data_lines)


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------

def extract_from_page(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # Key-figure boxes
    noegletal_parts = []
    for sel in NOEGLETAL_SELECTORS:
        for el in soup.select(sel):
            t = " ".join(el.get_text(separator=" ").split())
            if t:
                noegletal_parts.append(t)
    noegletal_text = "\n".join(noegletal_parts)

    # Inline tables: TabelSmal but not Notetabel
    table_mds = []
    for tbl in soup.find_all("table", class_="TabelSmal"):
        classes = tbl.get("class", [])
        if "Notetabel" in classes:
            continue
        md = html_table_to_markdown(tbl)
        if md:
            table_mds.append(md)

    return {
        "inline_tables_md": "\n\n".join(table_mds),
        "noegletal_text": noegletal_text,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_urls_from_progress() -> list[str]:
    if not PROGRESS_FILE.exists():
        return []
    urls = []
    for line in PROGRESS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "ok":
            urls.append(r["url"])
    return urls


def load_cached_urls() -> set[str]:
    seen: set[str] = set()
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    seen.add(json.loads(line)["url"])
                except Exception:
                    pass
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max articles to process")
    args = parser.parse_args()

    urls = load_urls_from_progress()
    print(f"Total articles in progress: {len(urls)}")

    cached = load_cached_urls()
    todo = [u for u in urls if u not in cached]
    print(f"Already cached: {len(cached)}, remaining: {len(todo)}")

    if args.limit:
        todo = todo[: args.limit]
        print(f"Limited to {args.limit}")

    CACHE_FILE.parent.mkdir(exist_ok=True)
    ok = err = 0

    with CACHE_FILE.open("a", encoding="utf-8") as f:
        for i, url in enumerate(todo):
            try:
                resp = SESSION.get(url, timeout=15)
                resp.raise_for_status()
                data = extract_from_page(resp.text)
                data["url"] = url
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
                ok += 1
                if (i + 1) % 100 == 0 or args.limit:
                    print(f"  [{i+1}/{len(todo)}] {url}")
            except Exception as e:
                print(f"  ERROR {url}: {e}")
                err += 1
            time.sleep(REQUEST_DELAY)

    print(f"\nDone. ok={ok}, errors={err}, total cached={len(cached)+ok}")


if __name__ == "__main__":
    main()
