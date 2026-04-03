import logging
import os
import sys
from datetime import datetime, timezone

from config import load_config
from fetcher import (
    load_sources,
    fetch_all_sources,
    filter_recent,
    deduplicate,
    quota_select,
)
from summarizer import (
    SummarizedItem,
    make_gemini_model,
    summarize_all_items,
    generate_insight,
)
from formatter import DigestContent, build_slack_message
from publisher import post_to_slack


logger = logging.getLogger(__name__)

MINIMUM_ITEMS_THRESHOLD = 3
SOURCES_PATH = os.path.join(os.path.dirname(__file__), "sources.yaml")
FALLBACK_INSIGHT = (
    "⚠️ Summaries unavailable today — Gemini API could not be reached. "
    "Headlines and links are sourced directly from RSS feeds."
)


def run_digest() -> None:
    config = load_config()
    model = make_gemini_model(config.gemini_api_key)
    now = datetime.now(tz=timezone.utc)

    sources = load_sources(SOURCES_PATH)
    raw_items = fetch_all_sources(sources)
    recent_items = filter_recent(items=raw_items, now=now, max_age_hours=24)
    unique_items = deduplicate(recent_items)
    top_candidates, quick_links = quota_select(unique_items)

    if len(top_candidates) < MINIMUM_ITEMS_THRESHOLD:
        logger.info(
            "Skipping digest — too few items after deduplication",
            extra={"count": len(top_candidates), "threshold": MINIMUM_ITEMS_THRESHOLD},
        )
        return

    try:
        top_stories = summarize_all_items(items=top_candidates, model=model)
        insight = generate_insight(items=top_stories, model=model)
    except Exception as exc:
        logger.warning(
            "Gemini API failed — posting fallback digest",
            extra={"error": str(exc)},
        )
        top_stories = [
            SummarizedItem(
                title=item.title,
                url=item.url,
                summary="",
                source_name=item.source_name,
                domain=item.domain,
            )
            for item in top_candidates
        ]
        insight = FALLBACK_INSIGHT

    digest = DigestContent(
        top_stories=top_stories,
        quick_links=quick_links,
        insight=insight,
        generated_at=now,
    )
    blocks = build_slack_message(digest)
    post_to_slack(webhook_url=config.slack_webhook_url, blocks=blocks)
    logger.info("Digest posted successfully", extra={"timestamp": now.isoformat()})


def handler(event: dict, context: object) -> dict:
    """AWS Lambda entry point."""
    run_digest()
    return {"statusCode": 200, "body": "Digest posted"}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stdout,
    )
    run_digest()
