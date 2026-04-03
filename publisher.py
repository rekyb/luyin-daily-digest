import httpx


def post_to_slack(webhook_url: str, blocks: list[dict]) -> None:
    payload = {"blocks": blocks}
    response = httpx.post(webhook_url, json=payload, timeout=10.0)
    if response.status_code != 200 or response.text != "ok":
        raise RuntimeError(
            f"Slack webhook returned {response.status_code}: {response.text!r}"
        )
