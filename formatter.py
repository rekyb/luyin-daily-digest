from dataclasses import dataclass
from datetime import datetime, timedelta

from fetcher import FeedItem
from summarizer import SummarizedItem


WIB_OFFSET = timedelta(hours=7)


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
    wib_dt = dt + WIB_OFFSET
    return wib_dt.strftime("%A, %d %B %Y")


def _format_time(dt: datetime) -> str:
    wib_dt = dt + WIB_OFFSET
    return wib_dt.strftime("%H:%M")


def _format_top_story(index: int, item: SummarizedItem) -> dict:
    text = (
        f"*{index}. {item.title}*\n"
        f"{item.summary}\n"
        f"🔗 <{item.url}|{item.source_name}> · _{item.domain}_"
    )
    return _section(text)


def _format_quick_links(items: list[FeedItem]) -> dict:
    lines = [f"• <{item.url}|{item.title}> · _{item.source_name}_" for item in items]
    return _section("🔗 *QUICK LINKS*\n" + "\n".join(lines))


def build_slack_message(digest: DigestContent) -> list[dict]:
    date_str = _format_date(digest.generated_at)
    time_str = _format_time(digest.generated_at)

    blocks: list[dict] = []

    # Header
    blocks.append(_header(f"📰 Luyin's Daily Digest — {date_str}"))
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
        _context(f"_This digest created by Lú yīn at {time_str} WIB, {date_str}_")
    )

    return blocks
