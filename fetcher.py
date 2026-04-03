from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import logging
import yaml
import feedparser
import httpx
import concurrent.futures
from typing import List, Tuple, Set, Optional
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
MAX_CONTENT_CHARS = 6000  # ~1500 tokens
TITLE_SIMILARITY_THRESHOLD = 80
TIMEOUT_SECONDS = 10.0
MAX_WORKERS = 10
USER_AGENT = "LuyinDailyDigest/1.0 (Contact: tech@example.com)"

DOMAIN_QUOTAS: dict[str, int] = {
    "edutech": 4,
    "ai": 2,
    "business": 1,
}
QUICK_LINKS_MAX = 5


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    domain: str


@dataclass(frozen=True)
class FeedItem:
    title: str
    url: str
    content: str
    source_name: str
    domain: str
    published: datetime


def load_sources(path: str) -> List[Source]:
    """Load sources from a YAML file with basic error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "sources" not in data:
            logger.error(f"Invalid sources file format at {path}")
            return []
        return [
            Source(name=s["name"], url=s["url"], domain=s["domain"])
            for s in data["sources"]
        ]
    except FileNotFoundError:
        logger.error(f"Sources file not found: {path}")
        return []
    except Exception as exc:
        logger.error(f"Failed to load sources from {path}: {exc}")
        return []


def _parse_published(entry: feedparser.FeedParserDict) -> datetime:
    struct = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if struct:
        return datetime(*struct[:6], tzinfo=timezone.utc)
    raise ValueError(
        f"Entry has no published or updated date: {getattr(entry, 'title', 'unknown')!r}"
    )


def _extract_content(entry: feedparser.FeedParserDict) -> str:
    content = ""
    if hasattr(entry, "content") and entry.content:
        content = entry.content[0].get("value", "")
    if not content:
        content = getattr(entry, "summary", "") or getattr(entry, "description", "")
    return content[:MAX_CONTENT_CHARS]


def fetch_feed(source: Source, client: httpx.Client) -> List[FeedItem]:
    """Fetch and parse a single feed using a shared httpx client."""
    try:
        response = client.get(source.url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        
        parsed = feedparser.parse(response.content)
        if parsed.bozo:
            logger.debug(f"Non-fatal parse error for {source.name!r}: {parsed.bozo_exception}")
            
    except httpx.HTTPError as exc:
        logger.warning(f"Network error for {source.name!r} ({source.url}): {exc}")
        return []
    except Exception as exc:
        logger.error(f"Unexpected error fetching {source.name!r}: {exc}")
        return []

    items: List[FeedItem] = []
    for entry in parsed.entries:
        url = getattr(entry, "link", "")
        title = getattr(entry, "title", "").strip()
        if not url or not title:
            continue
        try:
            items.append(
                FeedItem(
                    title=title,
                    url=url,
                    content=_extract_content(entry),
                    source_name=source.name,
                    domain=source.domain,
                    published=_parse_published(entry),
                )
            )
        except ValueError as exc:
            logger.warning(
                "Skipping entry",
                extra={"source": source.name, "title": title, "error": str(exc)},
            )
    return items


def fetch_all_sources(sources: List[Source]) -> List[FeedItem]:
    """Fetch all sources in parallel using a ThreadPool and connection pooling."""
    all_items: List[FeedItem] = []
    
    # Use httpx.Client for connection pooling and ThreadPoolExecutor for concurrency
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_source = {executor.submit(fetch_feed, s, client): s for s in sources}
            
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    all_items.extend(future.result())
                except Exception as exc:
                    logger.error(f"Parallel task failed for {source.name!r}: {exc}")
                    
    return all_items


def filter_recent(
    items: List[FeedItem],
    now: datetime,
    max_age_hours: int,
) -> List[FeedItem]:
    cutoff = now - timedelta(hours=max_age_hours)
    return [item for item in items if item.published >= cutoff]


def deduplicate(items: List[FeedItem]) -> List[FeedItem]:
    """Remove duplicates based on URL and fuzzy title matching."""
    seen_urls: Set[str] = set()
    seen_titles: List[str] = []
    result: List[FeedItem] = []

    for item in items:
        if item.url in seen_urls:
            continue

        title_lower = item.title.lower()
        
        # Exact title match fast-path
        if title_lower in seen_titles:
            seen_urls.add(item.url)
            continue

        # Fuzzy title match using rapidfuzz.process for efficiency
        if seen_titles:
            best_match = process.extractOne(
                title_lower, 
                seen_titles, 
                scorer=fuzz.ratio, 
                score_cutoff=TITLE_SIMILARITY_THRESHOLD
            )
            if best_match:
                seen_urls.add(item.url)
                continue

        seen_urls.add(item.url)
        seen_titles.append(title_lower)
        result.append(item)

    return result


def quota_select(
    items: List[FeedItem],
) -> Tuple[List[FeedItem], List[FeedItem]]:
    """Split items into (top_stories, quick_links) using domain quotas."""
    buckets: dict[str, List[FeedItem]] = {domain: [] for domain in DOMAIN_QUOTAS}
    for item in items:
        if item.domain in buckets:
            buckets[item.domain].append(item)

    for domain in buckets:
        buckets[domain].sort(key=lambda i: i.published, reverse=True)

    top_stories: List[FeedItem] = []
    for domain, quota in DOMAIN_QUOTAS.items():
        top_stories.extend(buckets[domain][:quota])

    top_urls = {item.url for item in top_stories}
    remainder = [item for item in items if item.url not in top_urls]
    remainder.sort(key=lambda i: i.published, reverse=True)
    quick_links = remainder[:QUICK_LINKS_MAX]

    return top_stories, quick_links
