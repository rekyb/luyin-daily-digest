from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fetcher import FeedItem
from summarizer import SummarizedItem


WIB_TZ = timezone(timedelta(hours=7))


@dataclass(frozen=True)
class DigestContent:
    top_stories: list[SummarizedItem]
    quick_links: list[FeedItem]
    insight: str
    generated_at: datetime  # UTC


def _divider() -> dict:
    return {"type": "divider"}


def _section(text: str) -> dict:
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": text},
    }


def _header(text: str) -> dict:
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": text, "emoji": True},
    }


def _context(text: str) -> dict:
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": text}],
    }


def _format_date(dt: datetime) -> str:
    return dt.astimezone(WIB_TZ).strftime("%A, %d %B %Y")


def _format_time(dt: datetime) -> str:
    return dt.astimezone(WIB_TZ).strftime("%H:%M")


def _format_top_story(index: int, item: SummarizedItem) -> dict:
    summary = item.summary[:2800] if len(item.summary) > 2800 else item.summary
    text = (
        f"*{index}. {item.title}*\n"
        f"{summary}\n"
        f"🔗 <{item.url}|{item.source_name}> · _{item.domain}_"
    )
    return _section(text)


def _format_quick_links(items: list[FeedItem]) -> dict:
    assert items, "_format_quick_links must not be called with an empty list"
    lines = [f"• <{item.url}|{item.title}> · _{item.source_name}_" for item in items]
    return _section("🔗 *EXTRA INSIGHTS*\n" + "\n".join(lines))


def build_slack_message(digest: DigestContent) -> list[dict]:
    date_str = _format_date(digest.generated_at)
    time_str = _format_time(digest.generated_at)

    blocks: list[dict] = []

    # Header
    blocks.append(_header(f"📰 Product and Technology Daily Digest — {date_str}"))
    blocks.append(_divider())

    # Top Stories
    blocks.append(_section("🔥 *TOP STORIES*"))
    blocks.append(_divider())
    for i, story in enumerate(digest.top_stories, start=1):
        blocks.append(_format_top_story(i, story))

    blocks.append(_divider())

    # Insight & Advisory
    blocks.append(_section("💡 *INSIGHT & ADVISORY*"))
    # Split insight into paragraphs to respect Slack's 3000-char block limit
    paragraphs = [p.strip() for p in digest.insight.split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        blocks.append(_section(paragraph))

    blocks.append(_divider())

    # Quick Links
    if digest.quick_links:
        blocks.append(_format_quick_links(digest.quick_links))
        blocks.append(_divider())

    # Footer
    blocks.append(
        _context(f"_This digest created at {time_str} WIB, {date_str}_")
    )

    return blocks
