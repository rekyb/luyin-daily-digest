from datetime import datetime
from unittest.mock import MagicMock, mock_open, patch, call
import httpx
import pytest

from fetcher import Source
from audit_sources import (
    check_feed_health,
    send_slack_notification,
    ask_gemini_for_fix,
    truncate_log_file,
)


def make_source(
    name: str = "Test Feed",
    url: str = "https://example.com/feed",
    domain: str = "edutech",
) -> Source:
    return Source(name=name, url=url, domain=domain)


# ---------------------------------------------------------------------------
# check_feed_health
# ---------------------------------------------------------------------------

def test_check_feed_health_returns_true_for_healthy_feed():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"<rss/>"

    mock_parsed = MagicMock()
    mock_parsed.bozo = False
    mock_parsed.entries = [MagicMock()]

    with patch("audit_sources.httpx.Client") as mock_client_cls, \
         patch("audit_sources.feedparser.parse", return_value=mock_parsed):
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_response
        healthy, error = check_feed_health(make_source())

    assert healthy is True
    assert error == ""


def test_check_feed_health_returns_false_on_http_error():
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.content = b""

    with patch("audit_sources.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_response
        healthy, error = check_feed_health(make_source())

    assert healthy is False
    assert "404" in error


def test_check_feed_health_returns_false_on_bozo_with_no_entries():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"<bad xml"

    mock_parsed = MagicMock()
    mock_parsed.bozo = True
    mock_parsed.bozo_exception = Exception("XML parse error")
    mock_parsed.entries = []

    with patch("audit_sources.httpx.Client") as mock_client_cls, \
         patch("audit_sources.feedparser.parse", return_value=mock_parsed):
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_response
        healthy, error = check_feed_health(make_source())

    assert healthy is False
    assert "Parse error" in error


def test_check_feed_health_returns_true_for_bozo_feed_with_entries():
    """Bozo flag but feed has entries — treat as healthy (some feeds are technically malformed but work)."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"<rss/>"

    mock_parsed = MagicMock()
    mock_parsed.bozo = True
    mock_parsed.bozo_exception = Exception("minor encoding issue")
    mock_parsed.entries = [MagicMock()]

    with patch("audit_sources.httpx.Client") as mock_client_cls, \
         patch("audit_sources.feedparser.parse", return_value=mock_parsed):
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_response
        healthy, error = check_feed_health(make_source())

    assert healthy is True
    assert error == ""


def test_check_feed_health_returns_false_for_empty_feed():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"<rss/>"

    mock_parsed = MagicMock()
    mock_parsed.bozo = False
    mock_parsed.entries = []

    with patch("audit_sources.httpx.Client") as mock_client_cls, \
         patch("audit_sources.feedparser.parse", return_value=mock_parsed):
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_response
        healthy, error = check_feed_health(make_source())

    assert healthy is False
    assert "empty" in error.lower()


def test_check_feed_health_returns_false_on_network_exception():
    with patch("audit_sources.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.side_effect = \
            httpx.ConnectError("connection refused")
        healthy, error = check_feed_health(make_source())

    assert healthy is False
    assert "connection refused" in error.lower()


# ---------------------------------------------------------------------------
# send_slack_notification
# ---------------------------------------------------------------------------

def test_send_slack_notification_posts_to_webhook():
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("audit_sources.httpx.post", return_value=mock_response) as mock_post:
        send_slack_notification(
            webhook_url="https://hooks.slack.com/fake",
            message="Feed X was replaced.",
        )
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert "Feed X was replaced." in payload["text"]


def test_send_slack_notification_does_not_raise_on_network_error():
    with patch("audit_sources.httpx.post", side_effect=httpx.ConnectError("timeout")):
        # Should not raise — audit continues even if Slack is unreachable
        send_slack_notification(
            webhook_url="https://hooks.slack.com/fake",
            message="Something failed.",
        )


def test_send_slack_notification_logs_warning_on_non_200():
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "server error"

    with patch("audit_sources.httpx.post", return_value=mock_response), \
         patch("audit_sources.logger") as mock_logger:
        send_slack_notification(
            webhook_url="https://hooks.slack.com/fake",
            message="test",
        )
        mock_logger.warning.assert_called_once()
        extra = mock_logger.warning.call_args[1]["extra"]
        assert extra["status_code"] == 500


# ---------------------------------------------------------------------------
# ask_gemini_for_fix
# ---------------------------------------------------------------------------

def test_ask_gemini_for_fix_returns_valid_suggestion():
    source = make_source(name="Broken Feed", url="https://broken.com/rss", domain="ai")
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(
        text='name: "Fixed Feed"\nurl: "https://fixed.com/rss"\ndomain: "ai"'
    )

    result = ask_gemini_for_fix(source=source, error="HTTP 404", model=mock_model)

    assert result is not None
    assert result["name"] == "Fixed Feed"
    assert result["url"] == "https://fixed.com/rss"
    assert result["domain"] == "ai"


def test_ask_gemini_for_fix_strips_markdown_code_blocks():
    source = make_source()
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(
        text='```yaml\nname: "Clean Feed"\nurl: "https://clean.com/rss"\ndomain: "edutech"\n```'
    )

    result = ask_gemini_for_fix(source=source, error="empty", model=mock_model)

    assert result is not None
    assert result["name"] == "Clean Feed"


def test_ask_gemini_for_fix_returns_none_on_missing_keys():
    source = make_source()
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(
        text='name: "Incomplete"\nurl: "https://example.com/rss"'
        # missing 'domain' key
    )

    result = ask_gemini_for_fix(source=source, error="empty", model=mock_model)

    assert result is None


def test_ask_gemini_for_fix_returns_none_on_empty_response():
    source = make_source()
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(text=None)

    result = ask_gemini_for_fix(source=source, error="HTTP 404", model=mock_model)

    assert result is None


def test_ask_gemini_for_fix_returns_none_on_api_failure():
    source = make_source()
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("Gemini API down")

    result = ask_gemini_for_fix(source=source, error="HTTP 500", model=mock_model)

    assert result is None


def test_ask_gemini_for_fix_returns_none_on_invalid_yaml():
    source = make_source()
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(
        text="this is not yaml: {{{"
    )

    result = ask_gemini_for_fix(source=source, error="empty", model=mock_model)

    assert result is None


# ---------------------------------------------------------------------------
# truncate_log_file
# ---------------------------------------------------------------------------

def test_truncate_log_file_does_nothing_if_file_missing(tmp_path):
    missing = str(tmp_path / "nonexistent.log")
    truncate_log_file(file_path=missing, max_lines=10)  # should not raise


def test_truncate_log_file_does_nothing_if_under_limit(tmp_path):
    log_file = tmp_path / "audit.log"
    lines = [f"line {i}\n" for i in range(5)]
    log_file.write_text("".join(lines))

    truncate_log_file(file_path=str(log_file), max_lines=10)

    assert log_file.read_text() == "".join(lines)


def test_truncate_log_file_keeps_last_n_lines_when_over_limit(tmp_path):
    log_file = tmp_path / "audit.log"
    lines = [f"line {i}\n" for i in range(20)]
    log_file.write_text("".join(lines))

    truncate_log_file(file_path=str(log_file), max_lines=5)

    result = log_file.read_text()
    result_lines = result.splitlines()
    assert len(result_lines) == 5
    assert result_lines[0] == "line 15"
    assert result_lines[-1] == "line 19"


def test_truncate_log_file_does_nothing_at_exact_limit(tmp_path):
    log_file = tmp_path / "audit.log"
    lines = [f"line {i}\n" for i in range(10)]
    log_file.write_text("".join(lines))

    truncate_log_file(file_path=str(log_file), max_lines=10)

    assert len(log_file.read_text().splitlines()) == 10
