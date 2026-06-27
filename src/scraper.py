"""
scraper.py

Scrapes Fiddler AI's public documentation from https://docs.fiddler.ai and
saves each page as a .txt file under data/, organized into subdirectories by
URL section (e.g. data/getting-started/, data/observability/, etc.).

The site is GitBook-powered and exposes a clean Markdown alternate for every
page at <url>.md, which we fetch instead of parsing JavaScript-rendered HTML.

Sitemap strategy:
  1. Fetch the sitemap index at /sitemap.xml to discover all sub-sitemaps.
  2. Parse each sub-sitemap for <loc> URLs.
  3. Fall back to a breadth-first crawl from the root if the sitemap is absent.

Usage:
    python src/scraper.py
"""

import re
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://docs.fiddler.ai"
SITEMAP_INDEX_URL = f"{BASE_URL}/sitemap.xml"
DATA_DIR = Path(__file__).parent.parent / "data"

CRAWL_DELAY_SECONDS = 1.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; fiddler-rag-scraper/1.0)"
}

# URL path prefixes that are clearly not reference documentation.
SKIP_PREFIXES = {
    "/blog",
    "/pricing",
    "/login",
    "/signup",
    "/terms",
    "/privacy",
    "/~gitbook",
}

# GitBook appends an "Agent Instructions" block to every .md response.
# It's boilerplate — strip everything from this header onward.
AGENT_INSTRUCTIONS_MARKER = "# Agent Instructions"


# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------


def fetch_sitemap_urls(session: requests.Session) -> list[str]:
    """Fetch all page URLs by parsing the sitemap index and its sub-sitemaps.

    Returns a list of absolute URLs, or an empty list if the sitemap is
    unavailable.
    """
    print(f"Fetching sitemap index: {SITEMAP_INDEX_URL}")
    try:
        resp = session.get(SITEMAP_INDEX_URL, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  Warning: could not fetch sitemap index ({exc})")
        return []

    index_soup = BeautifulSoup(resp.text, "lxml-xml")
    sub_sitemap_urls = [loc.get_text(strip=True) for loc in index_soup.find_all("loc")]
    print(f"  Found {len(sub_sitemap_urls)} sub-sitemaps")

    all_urls: list[str] = []
    for sub_url in sub_sitemap_urls:
        print(f"  Parsing sub-sitemap: {sub_url}")
        try:
            sub_resp = session.get(sub_url, timeout=15)
            sub_resp.raise_for_status()
            sub_soup = BeautifulSoup(sub_resp.text, "lxml-xml")
            locs = [loc.get_text(strip=True) for loc in sub_soup.find_all("loc")]
            print(f"    → {len(locs)} URLs")
            all_urls.extend(locs)
        except requests.RequestException as exc:
            print(f"    Warning: skipping sub-sitemap ({exc})")

    return all_urls


# ---------------------------------------------------------------------------
# Fallback crawler
# ---------------------------------------------------------------------------


def crawl_from_root(session: requests.Session) -> list[str]:
    """Breadth-first crawl starting from BASE_URL.

    Used only when the sitemap is unavailable.  Returns discovered doc URLs.
    """
    print(f"Sitemap unavailable — crawling from root: {BASE_URL}")
    visited: set[str] = set()
    queue: list[str] = [BASE_URL]
    found: list[str] = []

    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException:
            continue

        found.append(url)
        soup = BeautifulSoup(resp.text, "lxml")
        for anchor in soup.find_all("a", href=True):
            href: str = anchor["href"]
            abs_url = urljoin(url, href).split("#")[0].rstrip("/")
            parsed = urlparse(abs_url)
            if parsed.netloc == urlparse(BASE_URL).netloc and abs_url not in visited:
                queue.append(abs_url)

        time.sleep(CRAWL_DELAY_SECONDS)

    return found


# ---------------------------------------------------------------------------
# URL filtering and section detection
# ---------------------------------------------------------------------------


def is_doc_url(url: str) -> bool:
    """Return True if this URL looks like a documentation page to scrape."""
    parsed = urlparse(url)
    if parsed.netloc != urlparse(BASE_URL).netloc:
        return False
    path = parsed.path.rstrip("/")
    if not path or path == "":
        return False
    for prefix in SKIP_PREFIXES:
        if path.startswith(prefix):
            return False
    return True


def detect_section(url: str) -> str:
    """Extract the top-level section name from the URL path.

    Examples:
        https://docs.fiddler.ai/getting-started/ml-observability → getting-started
        https://docs.fiddler.ai/observability/platform           → observability
        https://docs.fiddler.ai/api/python-client               → api
    """
    path = urlparse(url).path.strip("/")
    if not path:
        return "root"
    parts = path.split("/")
    return parts[0] if parts[0] else "root"


def url_to_filename(url: str) -> str:
    """Derive a safe .txt filename from the last URL path segment.

    Falls back to the full sanitized path when the last segment is empty.
    """
    path = urlparse(url).path.strip("/")
    if not path:
        return "index.txt"
    slug = path.split("/")[-1]
    slug = re.sub(r"[^\w\-]", "_", slug)
    return f"{slug}.txt"


# ---------------------------------------------------------------------------
# Content fetching and cleaning
# ---------------------------------------------------------------------------


def fetch_page_markdown(session: requests.Session, url: str) -> str | None:
    """Fetch the GitBook Markdown alternate for a page.

    Appending .md to a GitBook URL returns clean Markdown without any
    navigation chrome, which is ideal for RAG ingestion.

    Returns the cleaned text, or None on failure.
    """
    md_url = url.rstrip("/") + ".md"
    try:
        resp = session.get(md_url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"    Warning: failed to fetch {md_url} ({exc})")
        return None

    text = resp.text

    # Strip the GitBook "Agent Instructions" boilerplate appended to every page.
    marker_idx = text.find(AGENT_INSTRUCTIONS_MARKER)
    if marker_idx != -1:
        text = text[:marker_idx].rstrip()

    return text.strip() if text.strip() else None


# ---------------------------------------------------------------------------
# File saving
# ---------------------------------------------------------------------------


def save_page(section: str, filename: str, content: str) -> Path:
    """Write page content to data/<section>/<filename>.

    Creates the section subdirectory if it does not exist.
    Returns the path of the saved file.
    """
    section_dir = DATA_DIR / section
    section_dir.mkdir(parents=True, exist_ok=True)
    out_path = section_dir / filename
    out_path.write_text(content, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def scrape() -> None:
    """Run the full scrape: discover URLs, fetch content, save to data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    # --- Discover URLs ---
    raw_urls = fetch_sitemap_urls(session)
    if not raw_urls:
        raw_urls = crawl_from_root(session)

    # Deduplicate, normalise, and filter.
    seen: set[str] = set()
    urls: list[str] = []
    for url in raw_urls:
        normalised = url.rstrip("/")
        if normalised not in seen and is_doc_url(normalised):
            seen.add(normalised)
            urls.append(normalised)

    print(f"\nTotal URLs to scrape: {len(urls)}\n{'─' * 50}")

    # --- Scrape each page ---
    section_counts: dict[str, int] = defaultdict(int)
    skipped = 0

    for i, url in enumerate(urls, start=1):
        section = detect_section(url)
        filename = url_to_filename(url)

        print(f"[{i:>3}/{len(urls)}] {url}")
        print(f"         section={section}  file={filename}")

        content = fetch_page_markdown(session, url)
        if not content:
            print(f"         Skipped (empty or fetch error)")
            skipped += 1
            time.sleep(CRAWL_DELAY_SECONDS)
            continue

        out_path = save_page(section, filename, content)
        print(f"         Saved → {out_path.relative_to(DATA_DIR.parent)}")
        section_counts[section] += 1

        time.sleep(CRAWL_DELAY_SECONDS)

    # --- Summary ---
    total_saved = sum(section_counts.values())
    print(f"\n{'═' * 50}")
    print(f"Scrape complete.")
    print(f"  Pages saved : {total_saved}")
    print(f"  Pages skipped: {skipped}")
    print(f"\nBreakdown by section:")
    for section, count in sorted(section_counts.items()):
        print(f"  {section:<30} {count:>4} pages")
    print(f"{'═' * 50}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scrape()
