#!/usr/bin/env python3
"""
Scrape CC BY publications from dst.dk/da/Statistik/udgivelser.

Four content types (16.498 total):
  - /nyt/{id}                              – Nyt fra Danmarks Statistik (13.898)
  - /analyser/{id}-slug                    – Analyser (184)
  - /pubomtale/{id}                        – Publikationer (1.724)
  - /da/Statistik/udgivelser/bagtal/{slug} – Bag tallene (692)

Progress is saved to downloads/progress.jsonl so runs are resumable.

Usage:
    uv run scraper.py
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "https://www.dst.dk"
LISTING_URL = f"{BASE_URL}/da/Statistik/udgivelser"
DOWNLOADS_DIR = Path(__file__).parent / "downloads"
PROGRESS_FILE = DOWNLOADS_DIR / "progress.jsonl"

REQUEST_DELAY = 0.5  # seconds between requests

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "Mozilla/5.0 (research bot; Danish Foundation Models; "
    "https://github.com/centre-for-humanities-computing)"
)
SESSION.headers["Accept-Language"] = "da,en;q=0.9"


# ── Progress tracking ─────────────────────────────────────────────────────────


def load_progress() -> set[str]:
    seen: set[str] = set()
    if PROGRESS_FILE.exists():
        for line in PROGRESS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line)["url"])
    return seen


def log_progress(record: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def get(url: str) -> requests.Response | None:
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=30, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == 2:
                print(f"    GET failed {url}: {e}")
            else:
                time.sleep(2 ** attempt)
    return None


def soup(url: str) -> BeautifulSoup | None:
    r = get(url)
    if r is None:
        return None
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(r.text, "html.parser")


# ── Listing scraper ───────────────────────────────────────────────────────────

# Publication types: (url_param, url_pattern_regex)
PUB_TYPES = [
    ("nyt",  re.compile(r"^/nyt/\d+")),
    ("aly",  re.compile(r"^/analyser/")),
    ("pub",  re.compile(r"^/pubomtale/\d+")),
    ("bag",  re.compile(r"^/da/Statistik/udgivelser/bagtal/")),
]


def get_listing_page(pub_type: str, page: int) -> list[str]:
    """Return list of article URLs from one listing page for a given pub type."""
    url = f"{LISTING_URL}?pub={pub_type}&page={page}"
    page_soup = soup(url)
    if page_soup is None:
        return []

    pattern = next(p for t, p in PUB_TYPES if t == pub_type)
    seen: set[str] = set()
    urls = []
    for a in page_soup.select(".flash-container a[href], .flash-link a[href]"):
        href = a["href"]
        if pattern.match(href) and href not in seen:
            seen.add(href)
            urls.append(BASE_URL + href if not href.startswith("http") else href)
    return urls


def get_type_pages(pub_type: str) -> int:
    """Return number of listing pages for a given pub type.

    Known counts (as of April 2026) used as fallback if the request fails.
    """
    _KNOWN = {"nyt": 695, "aly": 10, "pub": 87, "bag": 35}
    page_soup = soup(f"{LISTING_URL}?pub={pub_type}")
    if page_soup is None:
        return _KNOWN.get(pub_type, 100)
    last = 1
    for a in page_soup.select("a[href*='page=']"):
        m = re.search(r"page=(\d+)", a["href"])
        if m:
            last = max(last, int(m.group(1)))
    return last if last > 1 else _KNOWN.get(pub_type, 1)


# ── Article scrapers ──────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"(\d{1,2}\.\s+\w+ \d{4})")


def _clean_text(el: Tag) -> str:
    """Get clean text from a BS4 element, preserving paragraph breaks."""
    for br in el.find_all("br"):
        br.replace_with("\n")
    for tag in el.find_all(["p", "h2", "h3", "h4", "h5", "li"]):
        tag.append("\n\n")
    return re.sub(r"\n{3,}", "\n\n", el.get_text()).strip()


def _extract_date(el: Tag) -> str | None:
    """Find the first Danish date (e.g. '22. april 2026') inside el."""
    for p in el.find_all(["p", "span", "div"]):
        txt = p.get_text(strip=True)
        m = _DATE_RE.search(txt)
        if m:
            return m.group(1)
    return None


def scrape_nyt(url: str, page_soup: BeautifulSoup) -> dict:
    """Scrape a /nyt/{id} page."""
    record: dict = {"url": url, "content_type": "nyt"}

    # Title
    h1 = page_soup.select_one("h1")
    record["title"] = h1.get_text(strip=True) if h1 else None

    # Series (e.g. "Alle udgivelser i serien: Forbrugerforventninger")
    series_el = page_soup.select_one("a[href^='/nytserie/']")
    if series_el:
        raw = series_el.get_text(strip=True)
        # Strip prefix "Alle udgivelser i serien: "
        record["series"] = re.sub(r"^Alle udgivelser i serien:\s*", "", raw)
        record["series_url"] = BASE_URL + series_el["href"]
    else:
        record["series"] = None
        record["series_url"] = None

    # Main content is inside #webnyt (inside .cludoContent)
    content_el = page_soup.select_one("#webnyt") or page_soup.select_one(".cludoContent")

    if content_el:
        # Date lives inside content as a plain <p>
        record["date"] = _extract_date(content_el)

        # Remove noise: key-figures boxes, images, nav, breadcrumbs, scripts
        for tag in content_el.select(
            "script, style, img, .noegle_boks_ydre_container, "
            ".nyt_noegletalsboks_container, #NoegletalsBoksVenstre, "
            "#NoegletalsBoksHoejre, nav, .breadcrumb"
        ):
            tag.decompose()

        record["text"] = _clean_text(content_el)
    else:
        record["date"] = None
        record["text"] = None

    record["period"] = None
    return record


def scrape_analyse(url: str, page_soup: BeautifulSoup) -> dict:
    """Scrape a /analyser/{id}-{slug} page.

    Note: the analyse pages have malformed HTML where a large base64 <img> is not
    self-closed, so the parser nests the entire article content inside the <img>.
    We target .aly_container (the actual article div) to avoid this.
    """
    record: dict = {"url": url, "content_type": "analyse"}

    # Title
    h1 = page_soup.select_one("h1.aly_header, h1")
    record["title"] = h1.get_text(strip=True) if h1 else None

    record["series"] = None
    record["series_url"] = None

    # The article content lives in .aly_container (inside the malformed <img>)
    content_el = page_soup.select_one(".aly_container") or page_soup.select_one(".alymainarea")

    if content_el:
        # Date: look in .aly_metatop or .aly_dato or search whole content
        date_el = content_el.select_one(".aly_dato, .aly_metatop")
        if date_el:
            record["date"] = _extract_date(date_el) or _extract_date(page_soup)
        else:
            record["date"] = _extract_date(content_el) or _extract_date(page_soup)

        for tag in content_el.select(
            "script, style, nav, .breadcrumb, .breadcrumbMobile, "
            "#alysticky__scroller__base, #alystickymobile__box, .alysticky__box"
        ):
            tag.decompose()
        record["text"] = _clean_text(content_el)
    else:
        record["date"] = None
        record["text"] = None

    record["period"] = None
    return record


def scrape_cludocontent(url: str, page_soup: BeautifulSoup, content_type: str) -> dict:
    """Generic scraper for pages that use .cludoContent (pubomtale, bagtal)."""
    record: dict = {"url": url, "content_type": content_type}

    h1 = page_soup.select_one("h1")
    record["title"] = h1.get_text(strip=True) if h1 else None

    record["series"] = None
    record["series_url"] = None

    # Content is in .cludoContent inside the main content column
    content_el = page_soup.select_one(".cludoContent")

    if content_el:
        record["date"] = _extract_date(content_el) or _extract_date(page_soup)

        for tag in content_el.select(
            "script, style, nav, .breadcrumb, .breadcrumbMobile, "
            "#alysticky__scroller__base, #alystickymobile__box"
        ):
            tag.decompose()
        record["text"] = _clean_text(content_el)
    else:
        record["date"] = None
        record["text"] = None

    record["period"] = None
    return record


def scrape_article(url: str) -> dict:
    page = soup(url)
    if page is None:
        return {"url": url, "status": "fetch_failed"}

    if "/analyser/" in url:
        record = scrape_analyse(url, page)
    elif "/pubomtale/" in url:
        record = scrape_cludocontent(url, page, "pub")
    elif "/bagtal/" in url:
        record = scrape_cludocontent(url, page, "bag")
    else:
        record = scrape_nyt(url, page)

    record["status"] = "ok" if record.get("text") else "no_text"
    record["license"] = "CC BY 4.0"
    record["source"] = "Danmarks Statistik"

    return record


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    seen = load_progress()
    print(f"Resuming — {len(seen)} articles already processed\n")

    total_new = 0
    total_ok = 0

    for pub_type, _ in PUB_TYPES:
        print(f"\n{'='*60}")
        print(f"Type: {pub_type.upper()}")
        total_pages = get_type_pages(pub_type)
        print(f"Pages: {total_pages}")
        print(f"{'='*60}")

        consecutive_empty = 0
        MAX_CONSECUTIVE_EMPTY = 10

        for page_num in range(1, total_pages + 1):
            urls = get_listing_page(pub_type, page_num)

            if not urls:
                consecutive_empty += 1
                if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                    print(f"  Page {page_num}: {consecutive_empty} empty pages — stopping type.")
                    break
                continue
            consecutive_empty = 0

            new_urls = [u for u in urls if u not in seen]
            if not new_urls:
                if page_num % 50 == 0:
                    print(f"  Page {page_num}/{total_pages} — all seen")
                continue

            print(f"  Page {page_num}/{total_pages} — {len(new_urls)} new")

            for url in new_urls:
                record = scrape_article(url)
                log_progress(record)
                seen.add(url)
                total_new += 1
                if record.get("status") == "ok":
                    total_ok += 1
                time.sleep(REQUEST_DELAY)

    # Summary
    all_records = []
    if PROGRESS_FILE.exists():
        for line in PROGRESS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_records.append(json.loads(line))

    counts: dict[str, int] = {}
    for r in all_records:
        s = r.get("status", "unknown")
        counts[s] = counts.get(s, 0) + 1

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for status, count in sorted(counts.items()):
        print(f"  {status:25s}: {count}")
    print(f"\n  Articles scraped this run: {total_new}")
    print(f"  Successfully extracted:    {total_ok}")
    print(f"\n  Total in progress file:    {len(all_records)}")


if __name__ == "__main__":
    main()
