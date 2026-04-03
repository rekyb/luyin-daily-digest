from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import feedparser
import yaml
from rapidfuzz import fuzz


MAX_CONTENT_CHARS = 6000  # ~1500 tokens
TITLE_SIMILARITY_THRESHOLD = 80  # percent, for fuzzy dedup

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


def load_sources(path: str) -> list[Source]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        Source(name=s["name"], url=s["url"], domain=s["domain"])
        for s in data["sources"]
    ]


def _parse_published(entry: feedparser.FeedParserDict) -> datetime:
    struct = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if struct:
        return datetime(*struct[:6], tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc)


def _extract_content(entry: feedparser.FeedParserDict) -> str:
    content = ""
    if hasattr(entry, "content") and entry.content:
        content = entry.content[0].get("value", "")
    if not content:
        content = getattr(entry, "summary", "") or getattr(entry, "description", "")
    return content[:MAX_CONTENT_CHARS]


def fetch_feed(source: Source) -> list[FeedItem]:
    parsed = feedparser.parse(source.url)
    items: list[FeedItem] = []
    for entry in parsed.entries:
        url = getattr(entry, "link", "")
        title = getattr(entry, "title", "").strip()
        if not url or not title:
            continue
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
    return items


def fetch_all_sources(sources: list[Source]) -> list[FeedItem]:
    all_items: list[FeedItem] = []
    for source in sources:
        try:
            all_items.extend(fetch_feed(source))
        except Exception as exc:
            print(f"WARNING: failed to fetch {source.name}: {exc}")
    return all_items


def filter_recent(
    items: list[FeedItem],
    now: datetime,
    max_age_hours: int,
) -> list[FeedItem]:
    cutoff = now - timedelta(hours=max_age_hours)
    return [item for item in items if item.published >= cutoff]


def deduplicate(items: list[FeedItem]) -> list[FeedItem]:
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    result: list[FeedItem] = []

    for item in items:
        if item.url in seen_urls:
            continue

        is_duplicate = any(
            fuzz.ratio(item.title.lower(), seen.lower()) >= TITLE_SIMILARITY_THRESHOLD
            for seen in seen_titles
        )
        if is_duplicate:
            continue

        seen_urls.add(item.url)
        seen_titles.append(item.title)
        result.append(item)

    return result


def quota_select(
    items: list[FeedItem],
) -> tuple[list[FeedItem], list[FeedItem]]:
    """
    Split items into (top_stories, quick_links) using domain quotas.
    Items are sorted by recency within each domain bucket.
    """
    buckets: dict[str, list[FeedItem]] = {domain: [] for domain in DOMAIN_QUOTAS}
    for item in items:
        if item.domain in buckets:
            buckets[item.domain].append(item)

    for domain in buckets:
        buckets[domain].sort(key=lambda i: i.published, reverse=True)

    top_stories: list[FeedItem] = []
    for domain, quota in DOMAIN_QUOTAS.items():
        top_stories.extend(buckets[domain][:quota])

    top_urls = {item.url for item in top_stories}
    remainder = [item for item in items if item.url not in top_urls]
    remainder.sort(key=lambda i: i.published, reverse=True)
    quick_links = remainder[:QUICK_LINKS_MAX]

    return top_stories, quick_links
