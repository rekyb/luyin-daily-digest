import logging
from dataclasses import dataclass
import google.generativeai as genai

from fetcher import FeedItem


logger = logging.getLogger(__name__)

TONE_RULES = """
Writing rules — follow these exactly:
- Write like a sharp, well-read human editor. Not like AI.
- Banned phrases: "it's worth noting", "delve into", "in conclusion", "importantly",
  "notably", "it seems", "it appears", "as we can see", "leverage", "unlock potential",
  "game-changer", "revolutionary", "in the realm of", "navigate the landscape"
- No hedging. State facts directly. Drop qualifiers like "somewhat", "rather", "quite", "very"
- Use em-dashes sparingly — not as a default connector
- Vary sentence length. Mix short sentences with longer ones. Avoid uniform rhythm.
- Active voice only. Write what happened, not what "has been seen to occur"
- No buzzword stacking (e.g., "AI-driven learning transformation ecosystems")
- Summaries must read like a journalist wrote them, not a product description
""".strip()


@dataclass(frozen=True)
class SummarizedItem:
    title: str
    url: str
    summary: str
    source_name: str
    domain: str


def build_summarization_prompt(item: FeedItem) -> str:
    return f"""Summarize the following article in 2-3 sentences.

{TONE_RULES}

Article title: {item.title}
Article content:
{item.content}

Write only the summary. No preamble, no labels, no quotes around it."""


def build_insight_prompt(items: list[SummarizedItem]) -> str:
    stories_text = "\n\n".join(
        f"- {item.title} ({item.source_name}): {item.summary}" for item in items
    )
    return f"""You are writing the Insight & Advisory section of a daily digest for an edtech product team.

{TONE_RULES}

Here are today's top stories:

{stories_text}

Write 2-3 paragraphs that:
1. Identify 1-2 cross-cutting themes across these stories
2. Connect those themes to what matters for an edtech product team
3. Close with 1-2 concrete advisory points: specific things the team might consider or watch

Write in a direct, editorial voice. No intro like "Today's digest..." or "These stories show...". Start with the insight."""


def make_gemini_model(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


def summarize_item(item: FeedItem, model: genai.GenerativeModel) -> SummarizedItem:
    prompt = build_summarization_prompt(item)
    response = model.generate_content(prompt)
    return SummarizedItem(
        title=item.title,
        url=item.url,
        summary=response.text.strip(),
        source_name=item.source_name,
        domain=item.domain,
    )


def generate_insight(
    items: list[SummarizedItem],
    model: genai.GenerativeModel,
) -> str:
    prompt = build_insight_prompt(items)
    response = model.generate_content(prompt)
    return response.text.strip()
