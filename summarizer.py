import logging
import concurrent.futures
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google import genai as google_genai

from fetcher import FeedItem


logger = logging.getLogger(__name__)

MAX_SUMMARIZE_WORKERS = 5

TONE_RULES = """
You are a sharp, well-read human editor writing for an edtech product team's daily digest.

Writing rules — follow these exactly:
- Write like a journalist, not like an AI assistant or product marketer.
- Banned phrases: "it's worth noting", "delve into", "in conclusion", "importantly",
  "notably", "it seems", "it appears", "as we can see", "leverage", "unlock potential",
  "game-changer", "revolutionary", "in the realm of", "navigate the landscape",
  "dive into", "crucial", "transformative", "cutting-edge", "shed light on", "foster"
- No hedging. State facts directly. Drop qualifiers like "somewhat", "rather", "quite", "very"
- Use em-dashes sparingly — not as a default connector
- Vary sentence length. Mix short punchy sentences with longer ones. Avoid uniform rhythm.
- Active voice only. Write what happened, not what "has been seen to occur"
- No buzzword stacking (e.g., "AI-driven learning transformation ecosystems")
""".strip()

SUMMARIZATION_EXAMPLES = """
Examples of the style required:

❌ Bad: "It's worth noting that this revolutionary platform is leveraging cutting-edge AI to \
transform the educational landscape, fostering deeper student engagement and unlocking the \
potential of personalized learning."
✅ Good: "Duolingo replaced 10% of its contractor workforce with AI-generated content, the \
company confirmed in an earnings call. It is one of the clearest public admissions yet that \
generative AI has displaced knowledge workers in a consumer edtech product."

❌ Bad: "The study delves into the crucial ways AI tools have been seen to improve student \
outcomes, shedding light on the transformative role of technology in navigating the complexities \
of modern education."
✅ Good: "Students using AI writing feedback scored 12 points higher on standardized essays than \
a control group in a 6-month Stanford trial — but only when teachers reviewed the AI suggestions \
with students, not when students used the tools alone."
""".strip()


@runtime_checkable
class GeminiModel(Protocol):
    def generate_content(self, prompt: str) -> object:
        ...


class GeminiClientAdapter:
    """Connector to the Google Gemini API. Wraps google-genai client."""

    def __init__(
        self,
        client: google_genai.Client,
        model_name: str,
        system_instruction: str,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._config = google_genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
        )

    def generate_content(self, prompt: str) -> object:
        return self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=self._config,
        )


@dataclass(frozen=True)
class SummarizedItem:
    title: str
    url: str
    summary: str
    source_name: str
    domain: str


def build_summarization_prompt(item: FeedItem) -> str:
    return f"""{SUMMARIZATION_EXAMPLES}

Now summarize the following article in 2-3 sentences using the same style as the ✅ Good examples above.

Article title: {item.title}
Article content:
{item.content}

Write only the summary. No preamble, no labels, no quotes around it."""


def build_insight_prompt(items: list[SummarizedItem]) -> str:
    stories_text = "\n\n".join(
        f"- {item.title} ({item.source_name}): {item.summary}" for item in items
    )
    return f"""Write the Insight & Advisory section of today's digest based on these top stories:

{stories_text}

Write 2-3 paragraphs that:
1. Identify 1-2 cross-cutting themes across these stories
2. Connect those themes to what matters for an edtech product team
3. Close with 1-2 concrete advisory points: specific things the team might consider or watch

Do not open with "Today's digest...", "These stories show...", or any similar intro. Start directly with the insight."""


def make_gemini_model(api_key: str) -> GeminiModel:
    client = google_genai.Client(api_key=api_key)
    return GeminiClientAdapter(
        client=client,
        model_name="gemini-2.0-flash",
        system_instruction=TONE_RULES,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def summarize_item(item: FeedItem, model: GeminiModel) -> SummarizedItem:
    prompt = build_summarization_prompt(item)
    response = model.generate_content(prompt)
    if not response.text:
        raise ValueError(f"Gemini returned no text for article: {item.title!r}")
    return SummarizedItem(
        title=item.title,
        url=item.url,
        summary=response.text.strip(),
        source_name=item.source_name,
        domain=item.domain,
    )


def summarize_all_items(items: list[FeedItem], model: GeminiModel) -> list[SummarizedItem]:
    """Summarize all items in parallel, preserving original order. Failed items are skipped."""
    results: dict[int, SummarizedItem] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_SUMMARIZE_WORKERS) as executor:
        future_to_index = {
            executor.submit(summarize_item, item, model): i
            for i, item in enumerate(items)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.warning(
                    "Failed to summarize item — skipping",
                    extra={"title": items[idx].title, "error": str(exc)},
                )
    return [results[i] for i in sorted(results)]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def generate_insight(items: list[SummarizedItem], model: GeminiModel) -> str:
    if not items:
        raise ValueError("generate_insight requires at least one summarized item")
    prompt = build_insight_prompt(items)
    response = model.generate_content(prompt)
    if not response.text:
        raise ValueError(f"Gemini returned no text for insight ({len(items)} stories)")
    return response.text.strip()
