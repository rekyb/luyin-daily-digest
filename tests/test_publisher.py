from unittest.mock import MagicMock, patch
import httpx
import pytest

from publisher import post_to_slack


WEBHOOK_URL = "https://hooks.slack.com/services/fake/webhook/url"
SAMPLE_BLOCKS = [{"type": "section", "text": {"type": "mrkdwn", "text": "Hello"}}]


def test_post_to_slack_sends_post_request():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "ok"

    with patch("publisher.httpx.post", return_value=mock_response) as mock_post:
        post_to_slack(webhook_url=WEBHOOK_URL, blocks=SAMPLE_BLOCKS)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == WEBHOOK_URL
        payload = call_kwargs[1]["json"]
        assert payload["blocks"] == SAMPLE_BLOCKS


def test_post_to_slack_raises_on_non_ok_response():
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "invalid_payload"

    with patch("publisher.httpx.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Slack webhook returned 400"):
            post_to_slack(webhook_url=WEBHOOK_URL, blocks=SAMPLE_BLOCKS)


def test_post_to_slack_raises_on_network_error():
    with patch("publisher.httpx.post", side_effect=httpx.RequestError("timeout")):
        with pytest.raises(httpx.RequestError):
            post_to_slack(webhook_url=WEBHOOK_URL, blocks=SAMPLE_BLOCKS)
