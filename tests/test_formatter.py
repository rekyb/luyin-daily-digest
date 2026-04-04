from datetime import datetime, timezone
import pytest

from fetcher import FeedItem
from summarizer import SummarizedItem
from formatter import DigestContent, build_slack_message


def make_summarized(title: str, domain: str, url: str = "https://ex.com") -> SummarizedItem:
    return SummarizedItem(
        title=title,
        url=url,
        summary="A short summary sentence. Another sentence here.",
        source_name="EdSurge",
        domain=domain,
    )


def make_feed_item(title: str, url: str = "https://ex.com/q") -> FeedItem:
    return FeedItem(
        title=title,
        url=url,
        content="",
        source_name="Arxiv",
        domain="ai",
        published=datetime(2026, 4, 3, 6, 0, tzinfo=timezone.utc),
    )


FIXED_DATE = datetime(2026, 4, 3, 1, 0, 0, tzinfo=timezone.utc)  # 08:00 WIB


def make_digest() -> DigestContent:
    return DigestContent(
        top_stories=[make_summarized("Top Story 1", "edutech", "https://a.com/1")],
        quick_links=[make_feed_item("Quick Link 1", "https://b.com/1")],
        insight="This is the insight text.",
        generated_at=FIXED_DATE,
    )


def test_build_slack_message_returns_list_of_blocks():
    digest = make_digest()
    blocks = build_slack_message(digest)
    assert isinstance(blocks, list)
    assert len(blocks) > 0


def test_build_slack_message_contains_header():
    digest = make_digest()
    blocks = build_slack_message(digest)
    header_texts = [
        b.get("text", {}).get("text", "")
        for b in blocks
        if b.get("type") == "header"
    ]
    assert any("Daily Digest" in t for t in header_texts)
    assert any("2026" in t for t in header_texts)


def test_build_slack_message_contains_top_story_title():
    digest = make_digest()
    blocks = build_slack_message(digest)
    all_text = " ".join(str(b) for b in blocks)
    assert "Top Story 1" in all_text


def test_build_slack_message_contains_insight():
    digest = make_digest()
    blocks = build_slack_message(digest)
    all_text = " ".join(str(b) for b in blocks)
    assert "This is the insight text." in all_text


def test_build_slack_message_contains_quick_link():
    digest = make_digest()
    blocks = build_slack_message(digest)
    all_text = " ".join(str(b) for b in blocks)
    assert "Quick Link 1" in all_text


def test_build_slack_message_contains_footer_with_wib():
    digest = make_digest()
    blocks = build_slack_message(digest)
    all_text = " ".join(str(b) for b in blocks)
    assert "WIB" in all_text


def test_build_slack_message_contains_story_url():
    digest = make_digest()
    blocks = build_slack_message(digest)
    all_text = " ".join(str(b) for b in blocks)
    assert "https://a.com/1" in all_text


def test_build_slack_message_block_count_under_50():
    top_stories = [make_summarized(f"Story {i}", "edutech", f"https://a.com/{i}") for i in range(7)]
    quick_links = [make_feed_item(f"Quick {i}", f"https://b.com/{i}") for i in range(5)]
    digest = DigestContent(
        top_stories=top_stories,
        quick_links=quick_links,
        insight="Insight text.",
        generated_at=FIXED_DATE,
    )
    blocks = build_slack_message(digest)
    assert len(blocks) <= 50
