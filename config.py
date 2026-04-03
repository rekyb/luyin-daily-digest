from dataclasses import dataclass
import os
from dotenv import load_dotenv


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    gemini_api_key: str
    slack_webhook_url: str


def load_config() -> Config:
    # load_dotenv is a local-dev convenience only — has no effect on Lambda
    load_dotenv()

    try:
        gemini_api_key = os.environ["GEMINI_API_KEY"]
    except KeyError:
        raise ConfigurationError("GEMINI_API_KEY is not set in the environment")

    if not gemini_api_key:
        raise ConfigurationError("GEMINI_API_KEY is set but empty")

    try:
        slack_webhook_url = os.environ["SLACK_WEBHOOK_URL"]
    except KeyError:
        raise ConfigurationError("SLACK_WEBHOOK_URL is not set in the environment")

    if not slack_webhook_url:
        raise ConfigurationError("SLACK_WEBHOOK_URL is set but empty")

    return Config(
        gemini_api_key=gemini_api_key,
        slack_webhook_url=slack_webhook_url,
    )
