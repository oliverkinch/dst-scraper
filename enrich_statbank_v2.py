#!/usr/bin/env python3
"""
Enrich DST articles with StatBank table data — improved version.

Improvements over enrich_statbank.py:
  1. Discovers tables from the official "Tabeller fra Statistikbanken" section
     (<ul data-expandable-id="tables">) instead of all links in the article.
  2. Adapts max_periods per table based on time-range frequency:
       annual  → 5 periods   (e.g. 2007-2025)
       quarterly → 8 periods (e.g. 2007K1-2025K4)
       monthly → 12 periods  (e.g. 2007M01-2025M12)
  3. Collapses large multi-dimensional tables by requesting only "I alt" for
     non-TID dimensions that have it — prevents MAX_ROWS silent failures.
  4. Raises MAX_ROWS to 10 000 and logs skipped tables explicitly.

Output: downloads/statbank_cache_v2.jsonl (same schema as statbank_cache.jsonl)

Usage:
    uv run enrich_statbank_v2.py
    uv run enrich_statbank_v2.py --limit 10
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
ARTICLE_TABLES_FILE = Path(__file__).parent / "downloads" / "article_tables_cache.jsonl"
CACHE_FILE = Path(__file__).parent / "downloads" / "statbank_cache_v2.jsonl"
BASE_URL = "https://api.statbank.dk/v1"
REQUEST_DELAY = 0.5
MAX_ROWS = 10_000

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


# ---------------------------------------------------------------------------
# Date helpers (unchanged from enrich_statbank.py)
# ---------------------------------------------------------------------------

def parse_article_date(date_str: str | None) -> date | None:
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
    if m := re.match(r"(\d{4})M(\d{2})$", period_id):
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            return None
    if m := re.match(r"(\d{4})[QK](\d)$", period_id):
        month = (int(m.group(2)) - 1) * 3 + 1
        return date(int(m.group(1)), month, 1)
    if m := re.match(r"^(\d{4})$", period_id):
        return date(int(m.group(1)), 1, 1)
    return None


def select_time_periods(values: list[dict], article_date: date | None, max_periods: int) -> list[str]:
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


# ---------------------------------------------------------------------------
# Frequency detection from time-range string
# ---------------------------------------------------------------------------

def detect_frequency(time_range: str) -> str:
    """
    Detect frequency from a StatBank time-range string like:
      '2007-2025'        → 'annual'
      '2007K1-2025K4'    → 'quarterly'
      '2007M01-2025M12'  → 'monthly'
    """
    if re.search(r"\d{4}[KQ]\d", time_range, re.IGNORECASE):
        return "quarterly"
    if re.search(r"\d{4}M\d{2}", time_range, re.IGNORECASE):
        return "monthly"
    return "annual"


def max_periods_for_frequency(freq: str) -> int:
    return {"annual": 5, "quarterly": 8, "monthly": 12}.get(freq, 5)


# ---------------------------------------------------------------------------
# StatBank API (unchanged from enrich_statbank.py)
# ---------------------------------------------------------------------------

def _api_get(path: str, **kwargs) -> requests.Response | None:
    for attempt in range(2):
        try:
            resp = SESSION.get(f"{BASE_URL}/{path}", timeout=15, **kwargs)
            if resp.ok or resp.status_code in (400, 404):
                return resp
            # Transient error — retry after a longer sleep
            time.sleep(2.0)
        except requests.RequestException:
            if attempt == 0:
                time.sleep(2.0)
        finally:
            if attempt == 0:
                time.sleep(REQUEST_DELAY)
    time.sleep(REQUEST_DELAY)
    return None


def _api_post(path: str, body: dict, **kwargs) -> requests.Response | None:
    for attempt in range(2):
        try:
            resp = SESSION.post(f"{BASE_URL}/{path}", json=body, timeout=30, **kwargs)
            if resp.ok or resp.status_code in (400, 404):
                return resp
            time.sleep(2.0)
        except requests.RequestException:
            if attempt == 0:
                time.sleep(2.0)
        finally:
            if attempt == 0:
                time.sleep(REQUEST_DELAY)
    time.sleep(REQUEST_DELAY)
    return None


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


def get_table_data(table_code: str, variables: list[dict]) -> tuple[list[dict] | None, str | None]:
    """Returns (rows, failure_reason). Exactly one is None on success."""
    body = {
        "table": table_code,
        "format": "CSV",
        "valuePresentation": "Default",
        "variables": variables,
    }
    resp = _api_post("data", body)
    if resp is None:
        return None, "request failed"
    if resp.status_code in (400, 404):
        return None, f"HTTP {resp.status_code}"
    if not resp.ok:
        return None, f"HTTP {resp.status_code}"
    try:
        resp.encoding = "utf-8-sig"
        reader = csv.DictReader(io.StringIO(resp.text), delimiter=";")
        rows = list(reader)
        if len(rows) > MAX_ROWS:
            return None, f"too many rows ({len(rows)} > {MAX_ROWS})"
        return rows, None
    except Exception as e:
        return None, f"parse error: {e}"


# ---------------------------------------------------------------------------
# Dimension collapsing — request "I alt" for large non-TID dimensions
# ---------------------------------------------------------------------------

_TOTAL_LABELS = {"i alt", "total", "in total", "alle", "all", "hele landet"}
_TOTAL_SUBSTRINGS = (" i alt", "i alt ")  # catches "Alder i alt", "I alt ekskl."


def _has_total(values: list[dict]) -> str | None:
    """Return the value id for 'I alt' / 'Total' if present, else None.

    Handles exact matches ('I alt') and prefixed variants ('Alder i alt',
    'Hele landet') common in Danish StatBank tables.
    """
    for v in values:
        label = (v.get("text") or v.get("id") or "").lower().strip()
        if label in _TOTAL_LABELS:
            return v["id"]
        if any(sub in label for sub in _TOTAL_SUBSTRINGS):
            return v["id"]
        if label.startswith("i alt"):
            return v["id"]
    return None


def build_variables(variables_meta: list[dict], article_date: date | None, max_periods: int) -> tuple[list[dict], list[str]]:
    """
    Build the variables list for the StatBank data request.
    - TID: select up to max_periods periods ≤ article_date
    - Other dims: use 'I alt' if available, else '*'
    Returns (variables, selected_periods).
    """
    variables: list[dict] = []
    selected_periods: list[str] = []

    for var in variables_meta:
        var_id = var.get("id", "")
        var_values = var.get("values", [])

        if var_id.upper() == "TID":
            periods = select_time_periods(var_values, article_date, max_periods)
            if not periods:
                return [], []
            selected_periods = periods
            variables.append({"code": var_id, "values": periods})
        else:
            total_id = _has_total(var_values)
            if total_id is not None:
                variables.append({"code": var_id, "values": [total_id]})
            else:
                variables.append({"code": var_id, "values": ["*"]})

    return variables, selected_periods


def fetch_table_data(
    table_code: str,
    article_date: date | None,
    max_periods: int,
) -> tuple[dict | None, str | None]:
    """
    Fetch and structure data for a single StatBank table.
    Returns (result_dict, failure_reason) — exactly one is None.
    """
    info = get_tableinfo(table_code)
    if info is None:
        return None, "tableinfo fetch failed"

    variables_meta = info.get("variables", [])
    if not variables_meta:
        return None, "no variables in tableinfo"

    variables, selected_periods = build_variables(variables_meta, article_date, max_periods)
    if not variables:
        return None, "no valid TID periods"

    rows, reason = get_table_data(table_code, variables)
    if rows is None:
        return None, reason

    return {
        "table_code": table_code,
        "table_title": info.get("text", ""),
        "unit": info.get("unit", ""),
        "periods": selected_periods,
        "rows": rows,
    }, None


# ---------------------------------------------------------------------------
# Table discovery — "Tabeller fra Statistikbanken" section
# ---------------------------------------------------------------------------

def _is_valid_code(code: str) -> bool:
    return 2 <= len(code) <= 20 and not code.isdigit()


def extract_tables_from_statbank_section(html: str) -> list[dict]:
    """
    Parse the "Tabeller fra Statistikbanken" section:
      <ul data-expandable-id="tables"> inside <div class="nyt_mere_info">

    Returns list of {code, time_range, frequency, max_periods}.
    Falls back to all statistikbanken.dk links if the section is not found.
    """
    soup = BeautifulSoup(html, "html.parser")
    result: list[dict] = []
    seen: set[str] = set()

    table_list = soup.find("ul", attrs={"data-expandable-id": "tables"})

    if table_list:
        for a in table_list.find_all("a", href=STATBANK_HREF_RE):
            m = STATBANK_HREF_RE.search(str(a.get("href", "")))
            if not m:
                continue
            code = m.group(1).upper()
            if not _is_valid_code(code) or code in seen:
                continue
            seen.add(code)

            # Extract time range from <span class="text-muted">
            muted = a.find("span", class_="text-muted")
            time_range_text = muted.get_text(strip=True) if muted else ""
            freq = detect_frequency(time_range_text)

            result.append({
                "code": code,
                "time_range": time_range_text,
                "frequency": freq,
                "max_periods": max_periods_for_frequency(freq),
            })
    else:
        # Fallback: all statistikbanken links on page
        for a in soup.find_all("a", href=True):
            m = STATBANK_HREF_RE.search(str(a["href"]))
            if m:
                code = m.group(1).upper()
                if _is_valid_code(code) and code not in seen:
                    seen.add(code)
                    result.append({
                        "code": code,
                        "time_range": "",
                        "frequency": "annual",
                        "max_periods": 5,
                    })

    return result


def fetch_article_tables(url: str, article_date: date | None) -> tuple[list[dict], dict]:
    """
    Fetch article HTML, find table codes, fetch data for each.
    Returns (table_infos, statbank_data_dict).
    """
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        html = resp.text
    except requests.RequestException:
        return [], {}
    finally:
        time.sleep(REQUEST_DELAY)

    table_infos = extract_tables_from_statbank_section(html)
    statbank_data: dict[str, dict] = {}

    for info in table_infos:
        code = info["code"]
        result, reason = fetch_table_data(code, article_date, info["max_periods"])
        if result is not None:
            statbank_data[code] = result
        else:
            print(f"    SKIP {code}: {reason}")

    return table_infos, statbank_data


# ---------------------------------------------------------------------------
# Cache & progress loading
# ---------------------------------------------------------------------------

def load_cache() -> set[str]:
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


def load_articles_with_tables() -> list[dict]:
    """Load nyt articles. If article_tables_cache.jsonl exists, limit to those
    with at least one inline table; otherwise process all nyt articles."""
    if not PROGRESS_FILE.exists():
        raise FileNotFoundError(f"No progress file at {PROGRESS_FILE}")

    # Build set of URLs that have inline tables (optional filter)
    urls_with_tables: set[str] = set()
    if ARTICLE_TABLES_FILE.exists():
        for line in ARTICLE_TABLES_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("inline_tables_md"):
                    urls_with_tables.add(r["url"])
            except Exception:
                pass
        print(f"  (filtering to {len(urls_with_tables)} articles with inline tables)")

    articles: list[dict] = []
    for line in PROGRESS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") != "ok" or r.get("content_type") != "nyt":
            continue
        if urls_with_tables and r["url"] not in urls_with_tables:
            continue
        articles.append(r)
    return articles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    args = parser.parse_args()

    print("Loading articles with inline tables...")
    articles = load_articles_with_tables()
    print(f"  {len(articles)} nyt articles with inline tables")

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
            article_date = parse_article_date(article.get("date"))

            if args.limit:
                print(f"[{i+1}/{len(to_process)}] {url}")

            table_infos, statbank_data = fetch_article_tables(url, article_date)
            codes = [t["code"] for t in table_infos]

            if not codes:
                stats["no_codes"] += 1
            else:
                stats["tables_ok"] += len(statbank_data)
                stats["tables_fail"] += len(codes) - len(statbank_data)

            record = {
                "url": url,
                "statbank_tables": codes,
                "statbank_data": statbank_data if statbank_data else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            stats["articles_done"] += 1

            if (i + 1) % 50 == 0:
                print(
                    f"  [{i+1}/{len(to_process)}] "
                    f"no_codes={stats['no_codes']}  "
                    f"ok={stats['tables_ok']}  "
                    f"fail={stats['tables_fail']}"
                )

    total = stats["articles_done"]
    with_data = total - stats["no_codes"]
    print(f"\nDone. {total} articles → {CACHE_FILE}")
    print(f"  with table data: {with_data} ({100*with_data//total if total else 0}%)")
    print(f"  no table codes:  {stats['no_codes']}")
    print(f"  tables ok:       {stats['tables_ok']}")
    print(f"  tables failed:   {stats['tables_fail']}")


if __name__ == "__main__":
    main()
