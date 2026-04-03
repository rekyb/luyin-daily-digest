from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    gemini_api_key: str
    slack_webhook_url: str


def load_config() -> Config:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    if not slack_webhook_url:
        raise RuntimeError("SLACK_WEBHOOK_URL environment variable is not set")

    return Config(
        gemini_api_key=gemini_api_key,
        slack_webhook_url=slack_webhook_url,
    )
