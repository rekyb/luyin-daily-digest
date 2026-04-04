from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import pytest

from main import run_digest, MINIMUM_ITEMS_THRESHOLD


def make_feed_items(n: int, domain: str = "edutech"):
    from fetcher import FeedItem
    return [
        FeedItem(
            title=f"Story {i}",
            url=f"https://example.com/{i}",
            content="Content.",
            source_name="EdSurge",
            domain=domain,
            published=datetime(2026, 4, 3, 6, 0, tzinfo=timezone.utc),
        )
        for i in range(n)
    ]


def make_summarized(n: int):
    from summarizer import SummarizedItem
    return [
        SummarizedItem(
            title=f"Story {i}",
            url=f"https://example.com/{i}",
            summary="Summary.",
            source_name="EdSurge",
            domain="edutech",
        )
        for i in range(n)
    ]


@patch("main.post_to_slack")
@patch("main.build_slack_message")
@patch("main.generate_insight")
@patch("main.summarize_all_items")
@patch("main.quota_select")
@patch("main.deduplicate")
@patch("main.filter_recent")
@patch("main.fetch_all_sources")
@patch("main.load_sources")
@patch("main.make_gemini_model")
@patch("main.load_config")
def test_run_digest_full_pipeline(
    mock_config, mock_model, mock_sources, mock_fetch,
    mock_filter, mock_dedup, mock_quota, mock_summarize,
    mock_insight, mock_format, mock_post,
):
    from config import Config
    mock_config.return_value = Config(
        gemini_api_key="fake-key",
        slack_webhook_url="https://hooks.slack.com/fake",
    )
    mock_model.return_value = MagicMock()
    mock_sources.return_value = []
    mock_fetch.return_value = make_feed_items(5)
    mock_filter.return_value = make_feed_items(5)
    mock_dedup.return_value = make_feed_items(5)
    mock_quota.return_value = (make_feed_items(4), make_feed_items(1))
    mock_summarize.return_value = make_summarized(4)
    mock_insight.return_value = "Insight text."
    mock_format.return_value = [{"type": "section"}]

    run_digest()

    mock_post.assert_called_once()
    # Verify summarize_all_items was called
    mock_summarize.assert_called_once()
    # Verify insight was generated from the summaries
    mock_insight.assert_called_once()
    # Verify the digest was formatted
    mock_format.assert_called_once()


@patch("main.post_to_slack")
@patch("main.fetch_all_sources")
@patch("main.filter_recent")
@patch("main.deduplicate")
@patch("main.quota_select")
@patch("main.load_sources")
@patch("main.make_gemini_model")
@patch("main.load_config")
def test_run_digest_skips_post_when_too_few_items(
    mock_config, mock_model, mock_sources, mock_quota,
    mock_dedup, mock_filter, mock_fetch, mock_post,
):
    from config import Config
    mock_config.return_value = Config(
        gemini_api_key="fake-key",
        slack_webhook_url="https://hooks.slack.com/fake",
    )
    mock_model.return_value = MagicMock()
    mock_sources.return_value = []
    mock_fetch.return_value = make_feed_items(2)
    mock_filter.return_value = make_feed_items(2)
    mock_dedup.return_value = make_feed_items(2)
    mock_quota.return_value = (make_feed_items(2), [])

    run_digest()

    mock_post.assert_not_called()


@patch("main.post_to_slack")
@patch("main.build_slack_message")
@patch("main.generate_insight")
@patch("main.summarize_all_items")
@patch("main.quota_select")
@patch("main.deduplicate")
@patch("main.filter_recent")
@patch("main.fetch_all_sources")
@patch("main.load_sources")
@patch("main.make_gemini_model")
@patch("main.load_config")
def test_run_digest_posts_fallback_on_gemini_failure(
    mock_config, mock_model, mock_sources, mock_fetch,
    mock_filter, mock_dedup, mock_quota, mock_summarize,
    mock_insight, mock_format, mock_post,
):
    from config import Config
    mock_config.return_value = Config(
        gemini_api_key="fake-key",
        slack_webhook_url="https://hooks.slack.com/fake",
    )
    mock_model.return_value = MagicMock()
    mock_sources.return_value = []
    mock_fetch.return_value = make_feed_items(5)
    mock_filter.return_value = make_feed_items(5)
    mock_dedup.return_value = make_feed_items(5)
    mock_quota.return_value = (make_feed_items(4), make_feed_items(1))
    mock_summarize.side_effect = Exception("Gemini API down")
    mock_insight.side_effect = Exception("Gemini API down also")
    mock_format.return_value = [{"type": "section"}]

    run_digest()  # Should not raise

    mock_post.assert_called_once()
    # Fallback digest insight should mention summaries unavailable
    digest_arg = mock_format.call_args[0][0]
    assert "unavailable" in digest_arg.insight.lower()


@patch("main.run_digest")
def test_handler_returns_200(mock_run_digest):
    from main import handler
    result = handler(request=MagicMock())
    mock_run_digest.assert_called_once()
    assert result == ("Digest posted", 200)
