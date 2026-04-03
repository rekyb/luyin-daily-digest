from unittest.mock import MagicMock
from datetime import datetime, timezone
import pytest

from fetcher import FeedItem
from summarizer import (
    SummarizedItem,
    build_summarization_prompt,
    build_insight_prompt,
    summarize_item,
    generate_insight,
    TONE_RULES,
)


def make_feed_item(title: str = "Test Title", domain: str = "edutech") -> FeedItem:
    return FeedItem(
        title=title,
        url="https://example.com/article",
        content="Schools in Southeast Asia are adopting AI tutoring tools faster than expected.",
        source_name="EdSurge",
        domain=domain,
        published=datetime(2026, 4, 3, 6, 0, tzinfo=timezone.utc),
    )


def make_summarized_item() -> SummarizedItem:
    return SummarizedItem(
        title="Test Title",
        url="https://example.com/article",
        summary="A 2-3 sentence summary of the article.",
        source_name="EdSurge",
        domain="edutech",
    )


def test_build_summarization_prompt_contains_title_and_content():
    item = make_feed_item()
    prompt = build_summarization_prompt(item)
    assert item.title in prompt
    assert item.content in prompt


def test_build_summarization_prompt_includes_tone_rules():
    item = make_feed_item()
    prompt = build_summarization_prompt(item)
    assert TONE_RULES in prompt


def test_build_insight_prompt_includes_all_summaries():
    items = [make_summarized_item(), make_summarized_item()]
    items[1] = SummarizedItem(
        title="Second Item",
        url="https://example.com/2",
        summary="Another summary.",
        source_name="TechCrunch",
        domain="ai",
    )
    prompt = build_insight_prompt(items)
    assert "A 2-3 sentence summary" in prompt
    assert "Another summary" in prompt
    assert TONE_RULES in prompt


def test_summarize_item_returns_summarized_item():
    item = make_feed_item()
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(
        text="AI tutoring is spreading fast in Southeast Asia."
    )

    result = summarize_item(item=item, model=mock_model)

    assert isinstance(result, SummarizedItem)
    assert result.title == item.title
    assert result.url == item.url
    assert result.summary == "AI tutoring is spreading fast in Southeast Asia."
    assert result.source_name == item.source_name
    assert result.domain == item.domain


def test_generate_insight_returns_string():
    items = [make_summarized_item()]
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(
        text="Edtech teams should pay attention to regional AI adoption curves."
    )

    result = generate_insight(items=items, model=mock_model)
    assert isinstance(result, str)
    assert len(result) > 0


def test_summarize_item_raises_on_api_failure():
    item = make_feed_item()
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("API error")

    with pytest.raises(Exception, match="API error"):
        summarize_item(item=item, model=mock_model)
