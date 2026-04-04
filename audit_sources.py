import os
import yaml
import httpx
import feedparser
import logging
from datetime import datetime
from dataclasses import asdict

from fetcher import Source, load_sources
from summarizer import GeminiModel, make_gemini_model
from config import load_config


AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "audit_log.txt")
SOURCES_PATH = os.path.join(os.path.dirname(__file__), "sources.yaml")
TIMEOUT = 15.0
USER_AGENT = "LuyinDailyDigestAuditor/1.0"
MAX_LOG_LINES = 500

logger = logging.getLogger(__name__)


def check_feed_health(source: Source, client: httpx.Client) -> tuple[bool, str]:
    """Check if a feed is accessible and returns valid entries. Returns (is_healthy, error_message)."""
    try:
        response = client.get(source.url)
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"

        parsed = feedparser.parse(response.content)
        if parsed.bozo and not parsed.entries:
            return False, f"Parse error: {parsed.bozo_exception}"

        if not parsed.entries:
            return False, "Feed is empty"

        return True, ""
    except Exception as exc:
        return False, str(exc)


def send_slack_notification(webhook_url: str, message: str) -> None:
    """Send an audit summary message to Slack."""
    payload = {"text": f"🔍 *Source Audit Report*\n{message}"}
    try:
        response = httpx.post(webhook_url, json=payload, timeout=10.0)
        if response.status_code != 200:
            logger.warning(
                "Slack notification returned non-200",
                extra={"status_code": response.status_code, "body": response.text},
            )
    except Exception as exc:
        logger.error(
            "Failed to send Slack notification",
            extra={"error": str(exc)},
        )


def ask_gemini_for_fix(
    source: Source, error: str, model: GeminiModel
) -> dict | None:
    """Use Gemini to find a corrected URL or a suitable replacement for a broken source."""
    prompt = f"""The following RSS feed source for an EdTech/AI/Business digest is failing:
Name: {source.name}
URL: {source.url}
Domain: {source.domain}
Error: {error}

Tasks:
1. Find the current, working RSS feed URL for "{source.name}".
2. If "{source.name}" no longer provides an RSS feed, suggest a high-quality replacement RSS feed in the "{source.domain}" domain.

Return ONLY a valid YAML object with these keys: 'name', 'url', 'domain'.
Example format:
name: "New Source Name"
url: "https://example.com/rss"
domain: "{source.domain}"

Do not include any preamble, markdown blocks, or explanation."""

    try:
        response = model.generate_content(prompt)
        if not response.text:
            return None
        raw_text = response.text.strip().replace("```yaml", "").replace("```", "").strip()
        data = yaml.safe_load(raw_text)
        if isinstance(data, dict) and all(k in data for k in ("name", "url", "domain")):
            return data
        return None
    except Exception as exc:
        logger.error(
            "Gemini failed to suggest fix",
            extra={"source": source.name, "error": str(exc)},
        )
        return None


def truncate_log_file(file_path: str, max_lines: int) -> None:
    """Keep only the last max_lines lines of the log file."""
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) > max_lines:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines[-max_lines:])


def run_audit() -> None:
    config = load_config()
    model = make_gemini_model(config.gemini_api_key)
    sources = load_sources(SOURCES_PATH)

    if not sources:
        logger.error("No sources found to audit", extra={"path": SOURCES_PATH})
        return

    updated_sources: list[dict] = []
    log_entries: list[str] = [
        f"--- Audit Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
    ]
    slack_alerts: list[str] = []
    any_changes = False

    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=TIMEOUT,
    ) as client:
        for source in sources:
            is_healthy, error = check_feed_health(source, client)

            if is_healthy:
                updated_sources.append(asdict(source))
                logger.info("Feed healthy", extra={"source": source.name})
            else:
                logger.warning(
                    "Feed unhealthy",
                    extra={"source": source.name, "error": error},
                )
                log_entries.append(
                    f"❌ FAILED: {source.name} ({source.url}) - Error: {error}"
                )

                logger.info("Requesting Gemini repair", extra={"source": source.name})
                suggestion = ask_gemini_for_fix(source, error, model)

                if suggestion:
                    updated_sources.append(suggestion)
                    msg = f"✅ REPLACED: '{source.name}' -> '{suggestion['name']}' ({suggestion['url']})"
                    logger.info("Source replaced", extra={"old": source.name, "new": suggestion["name"]})
                    log_entries.append(msg)
                    slack_alerts.append(msg)
                    any_changes = True
                else:
                    updated_sources.append(asdict(source))
                    msg = f"⚠️ STAYED: Could not find replacement for {source.name}"
                    log_entries.append(msg)
                    slack_alerts.append(msg)

    log_entries.append(
        f"--- Audit Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
    )

    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(log_entries) + "\n\n")
    truncate_log_file(AUDIT_LOG_PATH, MAX_LOG_LINES)

    if any_changes:
        with open(SOURCES_PATH, "w", encoding="utf-8") as f:
            yaml.dump({"sources": updated_sources}, f, sort_keys=False, allow_unicode=True, indent=2)
        logger.info("sources.yaml updated", extra={"path": SOURCES_PATH})

    if slack_alerts:
        send_slack_notification(config.slack_webhook_url, "\n".join(slack_alerts))

    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as gh:
            gh.write("### Source Audit Summary\n")
            gh.write("\n".join([f"- {entry}" for entry in log_entries]))


if __name__ == "__main__":
    run_audit()
