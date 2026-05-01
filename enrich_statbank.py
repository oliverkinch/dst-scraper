#!/usr/bin/env python3
"""
Enrich scraped DST articles with StatBank table data.

For each 'nyt' article in downloads/progress.jsonl:
  1. Extract StatBank table codes from article text
  2. Fetch data from api.statbank.dk for those tables
  3. Cache results in downloads/statbank_cache.jsonl

Usage:
    uv run enrich_statbank.py
    uv run enrich_statbank.py --limit 10
    uv run enrich_statbank.py --content-types nyt,pub
    uv run enrich_statbank.py --max-periods 5
"""

import argparse
import csv
import io
import json
import re
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROGRESS_FILE = Path(__file__).parent / "downloads" / "progress.jsonl"
CACHE_FILE = Path(__file__).parent / "downloads" / "statbank_cache.jsonl"
BASE_URL = "https://api.statbank.dk/v1"
REQUEST_DELAY = 0.5  # seconds between API calls
MAX_ROWS = 2000  # skip tables that return more rows than this

STATBANK_HREF_RE = re.compile(r"statistikbanken\.dk/([A-Za-z0-9]+)", re.IGNORECASE)

DA_MONTHS = {
    "januar": 1, "februar": 2, "marts": 3, "april": 4,
    "maj": 5, "juni": 6, "juli": 7, "august": 8,
    "september": 9, "oktober": 10, "november": 11, "december": 12,
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (research bot; Danish Foundation Models; "
        "https://github.com/centre-for-humanities-computing)"
    ),
    "Accept-Language": "da,en;q=0.9",
})


# --- Date helpers ---

def parse_article_date(date_str: str | None) -> date | None:
    """Parse Danish date string like '22. april 2026' to a date."""
    if not date_str:
        return None
    m = re.search(r"(\d{1,2})\.\s+(\w+)\s+(\d{4})", date_str.lower())
    if not m:
        return None
    day, month_name, year = m.group(1), m.group(2), m.group(3)
    month = DA_MONTHS.get(month_name)
    if month is None:
        return None
    try:
        return date(int(year), month, int(day))
    except ValueError:
        return None


def parse_statbank_period(period_id: str) -> date | None:
    """Parse StatBank period ID to a comparable date."""
    # Monthly: 2024M01
    if m := re.match(r"(\d{4})M(\d{2})$", period_id):
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            return None
    # Quarterly: 2024Q1, 2024K1
    if m := re.match(r"(\d{4})[QK](\d)$", period_id):
        month = (int(m.group(2)) - 1) * 3 + 1
        return date(int(m.group(1)), month, 1)
    # Annual: 2024
    if m := re.match(r"^(\d{4})$", period_id):
        return date(int(m.group(1)), 1, 1)
    return None


def select_time_periods(values: list[dict], article_date: date | None, max_periods: int) -> list[str]:
    """Select up to max_periods time periods up to and including the article date."""
    candidates: list[tuple[date, str]] = []
    for v in values:
        pid = v.get("id", "")
        d = parse_statbank_period(pid)
        if d is None:
            continue
        if article_date is None or d <= article_date:
            candidates.append((d, pid))

    candidates.sort(reverse=True)
    return [pid for _, pid in candidates[:max_periods]]


# --- StatBank API ---

def _api_get(path: str, **kwargs) -> requests.Response | None:
    try:
        resp = SESSION.get(f"{BASE_URL}/{path}", timeout=15, **kwargs)
        return resp
    except requests.RequestException:
        return None
    finally:
        time.sleep(REQUEST_DELAY)


def _api_post(path: str, body: dict, **kwargs) -> requests.Response | None:
    try:
        resp = SESSION.post(f"{BASE_URL}/{path}", json=body, timeout=30, **kwargs)
        return resp
    except requests.RequestException:
        return None
    finally:
        time.sleep(REQUEST_DELAY)


def get_tableinfo(table_code: str) -> dict | None:
    resp = _api_get(f"tableinfo/{table_code}")
    if resp is None or resp.status_code == 404:
        return None
    if not resp.ok:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def get_table_data(table_code: str, variables: list[dict]) -> list[dict] | None:
    """Fetch CSV data rows from StatBank. Returns None on failure or if too many rows."""
    body = {
        "table": table_code,
        "format": "CSV",
        "valuePresentation": "Default",
        "variables": variables,
    }
    resp = _api_post("data", body)
    if resp is None or resp.status_code in (400, 404):
        return None
    if not resp.ok:
        return None
    try:
        resp.encoding = "utf-8-sig"  # strips BOM if present
        reader = csv.DictReader(io.StringIO(resp.text), delimiter=";")
        rows = list(reader)
        if len(rows) > MAX_ROWS:
            return None
        return rows
    except Exception:
        return None


def fetch_table_data(table_code: str, article_date: date | None, max_periods: int) -> dict | None:
    """Fetch and structure data for a single StatBank table."""
    info = get_tableinfo(table_code)
    if info is None:
        return None

    variables_meta = info.get("variables", [])
    if not variables_meta:
        return None

    variables: list[dict] = []
    selected_periods: list[str] = []

    for var in variables_meta:
        var_id = var.get("id", "")
        var_values = var.get("values", [])

        if var_id.upper() == "TID":
            periods = select_time_periods(var_values, article_date, max_periods)
            if not periods:
                return None
            selected_periods = periods
            variables.append({"code": var_id, "values": periods})
        else:
            variables.append({"code": var_id, "values": ["*"]})

    if not variables:
        return None

    rows = get_table_data(table_code, variables)
    if rows is None:
        return None

    return {
        "table_code": table_code,
        "table_title": info.get("text", ""),
        "unit": info.get("unit", ""),
        "periods": selected_periods,
        "rows": rows,
    }


# --- Table code extraction ---

def _is_valid_code(code: str) -> bool:
    """Reject purely numeric strings (subject IDs) and enforce length bounds."""
    return 2 <= len(code) <= 20 and not code.isdigit()


def extract_table_codes_from_url(url: str) -> list[str]:
    """
    Fetch the article HTML and extract StatBank table codes from
    <a href="statistikbanken.dk/CODE"> links.
    Falls back to scanning raw HTML for statistikbanken.dk/CODE patterns.
    Returns unique uppercased codes.
    """
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        html = resp.text
    except requests.RequestException:
        return []
    finally:
        time.sleep(REQUEST_DELAY)

    seen: set[str] = set()
    result: list[str] = []

    # Primary: extract from href attributes (catches linked codes the text scraper drops)
    try:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            m = STATBANK_HREF_RE.search(a["href"])
            if m:
                code = m.group(1).upper()
                if _is_valid_code(code) and code not in seen:
                    seen.add(code)
                    result.append(code)
    except Exception:
        pass

    # Fallback: scan raw HTML text (catches codes written out in prose)
    for code in STATBANK_HREF_RE.findall(html):
        upper = code.upper()
        if _is_valid_code(upper) and upper not in seen:
            seen.add(upper)
            result.append(upper)

    return result


# --- Cache & progress loading ---

def load_cache() -> set[str]:
    """Return the set of article URLs already in the cache file."""
    if not CACHE_FILE.exists():
        return set()
    seen: set[str] = set()
    for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                seen.add(json.loads(line)["url"])
            except Exception:
                pass
    return seen


def load_articles(content_types: list[str]) -> list[dict]:
    """Load status=ok articles of the given content types from progress.jsonl."""
    if not PROGRESS_FILE.exists():
        raise FileNotFoundError(f"No progress file at {PROGRESS_FILE}. Run scraper.py first.")
    articles: list[dict] = []
    for line in PROGRESS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") == "ok" and r.get("content_type") in content_types:
            articles.append(r)
    return articles


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Max articles to process")
    parser.add_argument(
        "--content-types",
        default="nyt",
        metavar="TYPES",
        help="Comma-separated content types to process (default: nyt)",
    )
    parser.add_argument(
        "--max-periods",
        type=int,
        default=3,
        metavar="N",
        help="Max time periods to fetch per table (default: 3)",
    )
    args = parser.parse_args()

    content_types = [t.strip() for t in args.content_types.split(",")]

    print(f"Loading articles (content_types={content_types})...")
    articles = load_articles(content_types)
    print(f"  {len(articles)} articles found")

    print("Loading cache...")
    cached_urls = load_cache()
    print(f"  {len(cached_urls)} already cached")

    to_process = [a for a in articles if a["url"] not in cached_urls]
    if args.limit is not None:
        to_process = to_process[: args.limit]
    print(f"  {len(to_process)} to process\n")

    if not to_process:
        print("Nothing to do.")
        return

    stats = {"no_codes": 0, "tables_ok": 0, "tables_fail": 0, "articles_done": 0}

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("a", encoding="utf-8") as f:
        for i, article in enumerate(to_process):
            url = article["url"]
            text = article.get("text", "")
            article_date = parse_article_date(article.get("date"))

            codes = extract_table_codes_from_url(url)

            if not codes:
                stats["no_codes"] += 1
                record = {"url": url, "statbank_tables": [], "statbank_data": None}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                table_data: dict[str, dict] = {}
                for code in codes:
                    data = fetch_table_data(code, article_date, max_periods=args.max_periods)
                    if data is not None:
                        table_data[code] = data
                        stats["tables_ok"] += 1
                    else:
                        stats["tables_fail"] += 1

                record = {
                    "url": url,
                    "statbank_tables": codes,
                    "statbank_data": table_data if table_data else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats["articles_done"] += 1

            if (i + 1) % 10 == 0 or i + 1 == len(to_process):
                print(
                    f"  [{i+1}/{len(to_process)}] "
                    f"no_codes={stats['no_codes']}  "
                    f"tables_ok={stats['tables_ok']}  "
                    f"tables_fail={stats['tables_fail']}"
                )

    total = stats["articles_done"]
    with_data = total - stats["no_codes"]
    print(f"\nDone! Processed {total} articles → {CACHE_FILE}")
    print(f"  with table data: {with_data} ({100*with_data//total if total else 0}%)")
    print(f"  no table codes:  {stats['no_codes']}")
    print(f"  tables fetched:  {stats['tables_ok']}")
    print(f"  api failures:    {stats['tables_fail']}")


if __name__ == "__main__":
    main()
