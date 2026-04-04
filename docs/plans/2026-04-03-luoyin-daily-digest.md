# Luyin Daily Digest System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully automated daily Slack digest that fetches Edutech, AI, and business/product news from RSS feeds, summarizes with Gemini Flash, and posts to Slack at 08:00 WIB via AWS Lambda + EventBridge.

**Architecture:** A Python Lambda function orchestrates a linear pipeline — fetch RSS feeds → deduplicate → rank by domain weight → summarize with Gemini → generate insight → format Slack blocks → post via webhook. Each stage is a focused module with a single responsibility.

**Tech Stack:** Python 3.12, feedparser, google-generativeai, httpx, rapidfuzz, pyyaml, AWS Lambda, AWS EventBridge, Slack Incoming Webhooks

---

## File Map

| File | Responsibility |
|------|----------------|
| `config.py` | Load env vars into typed dataclass |
| `sources.yaml` | RSS source list with domain tags |
| `fetcher.py` | Fetch feeds, filter last 24h, deduplicate, quota-select top stories |
| `summarizer.py` | Gemini API calls for per-article summaries and the insight section |
| `formatter.py` | Assemble Slack Block Kit message from digest content |
| `publisher.py` | POST to Slack Incoming Webhook |
| `main.py` | Lambda handler — orchestrate the full pipeline |
| `requirements.txt` | Python dependencies |
| `deploy/lambda_deploy.sh` | Package and deploy to AWS Lambda |
| `tests/test_fetcher.py` | Unit tests for fetcher |
| `tests/test_summarizer.py` | Unit tests for summarizer |
| `tests/test_formatter.py` | Unit tests for formatter |
| `tests/test_publisher.py` | Unit tests for publisher |
| `tests/test_main.py` | Integration test for the full pipeline |

---

## Task 1: Project Setup

**Files:**
- Create: `luyin-daily-digest-system/requirements.txt`
- Create: `luyin-daily-digest-system/requirements-dev.txt`
- Create: `luyin-daily-digest-system/sources.yaml`
- Create: `luyin-daily-digest-system/config.py`
- Create: `luyin-daily-digest-system/.env.example`
- Create: `luyin-daily-digest-system/tests/__init__.py`

- [ ] **Step 1: Create project directory**

```bash
mkdir -p luyin-daily-digest-system/tests luyin-daily-digest-system/deploy
cd luyin-daily-digest-system
```

- [ ] **Step 2: Write requirements.txt**

```
google-generativeai>=0.8.0
feedparser>=6.0.11
httpx>=0.27.0
pyyaml>=6.0.0
rapidfuzz>=3.6.0
python-dotenv>=1.0.0
```

- [ ] **Step 3: Write requirements-dev.txt**

```
pytest>=8.0.0
pytest-mock>=3.12.0
```

- [ ] **Step 4: Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

Expected: All packages install without errors.

- [ ] **Step 5: Write .env.example**

```
GEMINI_API_KEY=your_gemini_api_key_here
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
```

- [ ] **Step 6: Write sources.yaml**

```yaml
sources:
  - name: "EdSurge"
    url: "https://edsurge.com/news.rss"
    domain: "edutech"

  - name: "eLearning Industry"
    url: "https://elearningindustry.com/feed"
    domain: "edutech"

  - name: "The Journal"
    url: "https://thejournal.com/rss-feeds/all-articles.aspx"  # verify URL
    domain: "edutech"

  - name: "Mindshift (KQED)"
    url: "https://www.kqed.org/mindshift/feed"
    domain: "edutech"

  - name: "MIT Technology Review"
    url: "https://www.technologyreview.com/topic/learning/feed"  # verify URL
    domain: "edutech"

  - name: "TechCrunch AI"
    url: "https://techcrunch.com/category/artificial-intelligence/feed/"
    domain: "ai"

  - name: "The Rundown AI"
    url: "https://www.therundown.ai/rss"  # verify URL
    domain: "ai"

  - name: "Import AI"
    url: "https://jack-clark.net/feed/"
    domain: "ai"

  - name: "Google AI Blog"
    url: "https://blog.google/technology/ai/rss/"  # verify URL
    domain: "ai"

  - name: "Arxiv cs.AI"
    url: "http://export.arxiv.org/rss/cs.AI"
    domain: "ai"

  - name: "Arxiv cs.CY"
    url: "http://export.arxiv.org/rss/cs.CY"
    domain: "ai"

  - name: "Harvard Business Review"
    url: "https://feeds.hbr.org/harvardbusiness"  # verify URL
    domain: "business"

  - name: "Product Hunt"
    url: "https://www.producthunt.com/feed"
    domain: "business"

  - name: "Lenny's Newsletter"
    url: "https://www.lennysnewsletter.com/feed"  # verify URL
    domain: "business"
```

- [ ] **Step 7: Write config.py**

```python
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
```

- [ ] **Step 8: Create tests/__init__.py**

```python
```

(Empty file — makes tests/ a package.)

- [ ] **Step 9: Commit**

```bash
git init
git add .
git commit -m "feat: project setup — config, sources, dependencies"
```

---

## Task 2: RSS Fetcher

**Files:**
- Create: `luyin-daily-digest-system/fetcher.py`
- Create: `luyin-daily-digest-system/tests/test_fetcher.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_fetcher.py`:

```python
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import feedparser
import pytest

from fetcher import (
    FeedItem,
    Source,
    load_sources,
    filter_recent,
    deduplicate,
    quota_select,
)


FIXED_NOW = datetime(2026, 4, 3, 8, 0, 0, tzinfo=timezone.utc)


def make_item(
    title: str,
    url: str,
    domain: str,
    hours_ago: float = 2.0,
    source_name: str = "Test Source",
) -> FeedItem:
    published = FIXED_NOW - timedelta(hours=hours_ago)
    return FeedItem(
        title=title,
        url=url,
        content="Some article content here.",
        source_name=source_name,
        domain=domain,
        published=published,
    )


def test_filter_recent_keeps_items_within_24h():
    items = [
        make_item("New", "http://a.com/1", "edutech", hours_ago=2),
        make_item("Old", "http://a.com/2", "edutech", hours_ago=25),
    ]
    result = filter_recent(items=items, now=FIXED_NOW, max_age_hours=24)
    assert len(result) == 1
    assert result[0].title == "New"


def test_filter_recent_empty_list():
    result = filter_recent(items=[], now=FIXED_NOW, max_age_hours=24)
    assert result == []


def test_deduplicate_removes_exact_url_duplicates():
    items = [
        make_item("Story A", "http://same.com/1", "edutech"),
        make_item("Story A copy", "http://same.com/1", "ai"),
    ]
    result = deduplicate(items)
    assert len(result) == 1
    assert result[0].url == "http://same.com/1"


def test_deduplicate_removes_similar_titles():
    items = [
        make_item("OpenAI Releases GPT-5 Model Today", "http://a.com/1", "ai"),
        make_item("OpenAI releases GPT-5 model", "http://b.com/2", "ai"),
    ]
    result = deduplicate(items)
    assert len(result) == 1


def test_deduplicate_keeps_distinct_items():
    items = [
        make_item("Google launches new AI tool", "http://a.com/1", "ai"),
        make_item("MIT study shows learning gaps widen", "http://b.com/2", "edutech"),
    ]
    result = deduplicate(items)
    assert len(result) == 2


def test_quota_select_enforces_domain_quotas():
    items = [make_item(f"Ed {i}", f"http://ed.com/{i}", "edutech") for i in range(6)]
    items += [make_item(f"AI {i}", f"http://ai.com/{i}", "ai") for i in range(4)]
    items += [make_item(f"Biz {i}", f"http://biz.com/{i}", "business") for i in range(3)]

    top, quick = quota_select(items)

    edutech_top = [i for i in top if i.domain == "edutech"]
    ai_top = [i for i in top if i.domain == "ai"]
    biz_top = [i for i in top if i.domain == "business"]

    assert len(edutech_top) <= 4
    assert len(ai_top) <= 2
    assert len(biz_top) <= 1
    assert len(top) <= 7
    assert len(quick) <= 5


def test_load_sources_returns_source_list(tmp_path):
    yaml_content = """
sources:
  - name: "Test Feed"
    url: "https://example.com/feed"
    domain: "edutech"
"""
    sources_file = tmp_path / "sources.yaml"
    sources_file.write_text(yaml_content)

    result = load_sources(path=str(sources_file))
    assert len(result) == 1
    assert result[0].name == "Test Feed"
    assert result[0].domain == "edutech"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_fetcher.py -v
```

Expected: `ModuleNotFoundError: No module named 'fetcher'`

- [ ] **Step 3: Write fetcher.py**

```python
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import time
import feedparser
import yaml
from rapidfuzz import fuzz


MAX_CONTENT_CHARS = 6000  # ~1500 tokens
TITLE_SIMILARITY_THRESHOLD = 80  # percent, for fuzzy dedup

DOMAIN_QUOTAS: dict[str, int] = {
    "edutech": 4,
    "ai": 2,
    "business": 1,
}
QUICK_LINKS_MAX = 5


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    domain: str


@dataclass(frozen=True)
class FeedItem:
    title: str
    url: str
    content: str
    source_name: str
    domain: str
    published: datetime


def load_sources(path: str = "sources.yaml") -> list[Source]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        Source(name=s["name"], url=s["url"], domain=s["domain"])
        for s in data["sources"]
    ]


def _parse_published(entry: feedparser.FeedParserDict) -> datetime:
    struct = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if struct:
        return datetime(*struct[:6], tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc)


def _extract_content(entry: feedparser.FeedParserDict) -> str:
    content = ""
    if hasattr(entry, "content") and entry.content:
        content = entry.content[0].get("value", "")
    if not content:
        content = getattr(entry, "summary", "") or getattr(entry, "description", "")
    return content[:MAX_CONTENT_CHARS]


def fetch_feed(source: Source) -> list[FeedItem]:
    parsed = feedparser.parse(source.url)
    items: list[FeedItem] = []
    for entry in parsed.entries:
        url = getattr(entry, "link", "")
        title = getattr(entry, "title", "").strip()
        if not url or not title:
            continue
        items.append(
            FeedItem(
                title=title,
                url=url,
                content=_extract_content(entry),
                source_name=source.name,
                domain=source.domain,
                published=_parse_published(entry),
            )
        )
    return items


def fetch_all_sources(sources: list[Source]) -> list[FeedItem]:
    all_items: list[FeedItem] = []
    for source in sources:
        try:
            all_items.extend(fetch_feed(source))
        except Exception as exc:
            print(f"WARNING: failed to fetch {source.name}: {exc}")
    return all_items


def filter_recent(
    items: list[FeedItem],
    now: datetime,
    max_age_hours: int = 24,
) -> list[FeedItem]:
    cutoff = now - timedelta(hours=max_age_hours)
    return [item for item in items if item.published >= cutoff]


def deduplicate(items: list[FeedItem]) -> list[FeedItem]:
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    result: list[FeedItem] = []

    for item in items:
        if item.url in seen_urls:
            continue

        is_duplicate = any(
            fuzz.ratio(item.title.lower(), seen.lower()) >= TITLE_SIMILARITY_THRESHOLD
            for seen in seen_titles
        )
        if is_duplicate:
            continue

        seen_urls.add(item.url)
        seen_titles.append(item.title)
        result.append(item)

    return result


def quota_select(
    items: list[FeedItem],
) -> tuple[list[FeedItem], list[FeedItem]]:
    """
    Split items into (top_stories, quick_links) using domain quotas.
    Items are sorted by recency within each domain bucket.
    """
    buckets: dict[str, list[FeedItem]] = {domain: [] for domain in DOMAIN_QUOTAS}
    for item in items:
        if item.domain in buckets:
            buckets[item.domain].append(item)

    for domain in buckets:
        buckets[domain].sort(key=lambda i: i.published, reverse=True)

    top_stories: list[FeedItem] = []
    for domain, quota in DOMAIN_QUOTAS.items():
        top_stories.extend(buckets[domain][:quota])

    top_urls = {item.url for item in top_stories}
    remainder = [item for item in items if item.url not in top_urls]
    remainder.sort(key=lambda i: i.published, reverse=True)
    quick_links = remainder[:QUICK_LINKS_MAX]

    return top_stories, quick_links
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_fetcher.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fetcher.py tests/test_fetcher.py
git commit -m "feat: rss fetcher with deduplication and quota selection"
```

---

## Task 3: AI Summarizer

**Files:**
- Create: `luyin-daily-digest-system/summarizer.py`
- Create: `luyin-daily-digest-system/tests/test_summarizer.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_summarizer.py`:

```python
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import pytest

from fetcher import FeedItem
from summarizer import (
    SummarizedItem,
    build_summarization_prompt,
    build_insight_prompt,
    summarize_item,
    generate_insight,
    TONE_RULES,
)


def make_feed_item(title: str = "Test Title", domain: str = "edutech") -> FeedItem:
    return FeedItem(
        title=title,
        url="https://example.com/article",
        content="Schools in Southeast Asia are adopting AI tutoring tools faster than expected.",
        source_name="EdSurge",
        domain=domain,
        published=datetime(2026, 4, 3, 6, 0, tzinfo=timezone.utc),
    )


def make_summarized_item() -> SummarizedItem:
    return SummarizedItem(
        title="Test Title",
        url="https://example.com/article",
        summary="A 2-3 sentence summary of the article.",
        source_name="EdSurge",
        domain="edutech",
    )


def test_build_summarization_prompt_contains_title_and_content():
    item = make_feed_item()
    prompt = build_summarization_prompt(item)
    assert item.title in prompt
    assert item.content in prompt


def test_build_summarization_prompt_includes_tone_rules():
    item = make_feed_item()
    prompt = build_summarization_prompt(item)
    assert TONE_RULES in prompt


def test_build_insight_prompt_includes_all_summaries():
    items = [make_summarized_item(), make_summarized_item()]
    items[1] = SummarizedItem(
        title="Second Item",
        url="https://example.com/2",
        summary="Another summary.",
        source_name="TechCrunch",
        domain="ai",
    )
    prompt = build_insight_prompt(items)
    assert "A 2-3 sentence summary" in prompt
    assert "Another summary" in prompt
    assert TONE_RULES in prompt


def test_summarize_item_returns_summarized_item():
    item = make_feed_item()
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(
        text="AI tutoring is spreading fast in Southeast Asia."
    )

    result = summarize_item(item=item, model=mock_model)

    assert isinstance(result, SummarizedItem)
    assert result.title == item.title
    assert result.url == item.url
    assert result.summary == "AI tutoring is spreading fast in Southeast Asia."
    assert result.source_name == item.source_name
    assert result.domain == item.domain


def test_generate_insight_returns_string():
    items = [make_summarized_item()]
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(
        text="Edtech teams should pay attention to regional AI adoption curves."
    )

    result = generate_insight(items=items, model=mock_model)
    assert isinstance(result, str)
    assert len(result) > 0


def test_summarize_item_raises_on_api_failure():
    item = make_feed_item()
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("API error")

    with pytest.raises(Exception, match="API error"):
        summarize_item(item=item, model=mock_model)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_summarizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'summarizer'`

- [ ] **Step 3: Write summarizer.py**

```python
from dataclasses import dataclass
import google.generativeai as genai

from fetcher import FeedItem


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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_summarizer.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add summarizer.py tests/test_summarizer.py
git commit -m "feat: gemini summarizer with tone and voice enforcement"
```

---

## Task 4: Slack Formatter

**Files:**
- Create: `luyin-daily-digest-system/formatter.py`
- Create: `luyin-daily-digest-system/tests/test_formatter.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_formatter.py`:

```python
from datetime import datetime, timezone
import pytest

from fetcher import FeedItem
from summarizer import SummarizedItem
from formatter import DigestContent, build_slack_message


def make_summarized(title: str, domain: str, url: str = "https://ex.com") -> SummarizedItem:
    return SummarizedItem(
        title=title,
        url=url,
        summary="A short summary sentence. Another sentence here.",
        source_name="EdSurge",
        domain=domain,
    )


def make_feed_item(title: str, url: str = "https://ex.com/q") -> FeedItem:
    return FeedItem(
        title=title,
        url=url,
        content="",
        source_name="Arxiv",
        domain="ai",
        published=datetime(2026, 4, 3, 6, 0, tzinfo=timezone.utc),
    )


FIXED_DATE = datetime(2026, 4, 3, 1, 0, 0, tzinfo=timezone.utc)  # 08:00 WIB


def make_digest() -> DigestContent:
    return DigestContent(
        top_stories=[make_summarized("Top Story 1", "edutech", "https://a.com/1")],
        quick_links=[make_feed_item("Quick Link 1", "https://b.com/1")],
        insight="This is the insight text.",
        generated_at=FIXED_DATE,
    )


def test_build_slack_message_returns_list_of_blocks():
    digest = make_digest()
    blocks = build_slack_message(digest)
    assert isinstance(blocks, list)
    assert len(blocks) > 0


def test_build_slack_message_contains_header():
    digest = make_digest()
    blocks = build_slack_message(digest)
    header_texts = [
        b.get("text", {}).get("text", "")
        for b in blocks
        if b.get("type") == "header"
    ]
    assert any("Luyin" in t for t in header_texts)
    assert any("2026" in t for t in header_texts)


def test_build_slack_message_contains_top_story_title():
    digest = make_digest()
    blocks = build_slack_message(digest)
    all_text = " ".join(
        str(b) for b in blocks
    )
    assert "Top Story 1" in all_text


def test_build_slack_message_contains_insight():
    digest = make_digest()
    blocks = build_slack_message(digest)
    all_text = " ".join(str(b) for b in blocks)
    assert "This is the insight text." in all_text


def test_build_slack_message_contains_quick_link():
    digest = make_digest()
    blocks = build_slack_message(digest)
    all_text = " ".join(str(b) for b in blocks)
    assert "Quick Link 1" in all_text


def test_build_slack_message_contains_footer_with_luoyin():
    digest = make_digest()
    blocks = build_slack_message(digest)
    all_text = " ".join(str(b) for b in blocks)
    assert "Lú yīn" in all_text


def test_build_slack_message_contains_story_url():
    digest = make_digest()
    blocks = build_slack_message(digest)
    all_text = " ".join(str(b) for b in blocks)
    assert "https://a.com/1" in all_text


def test_build_slack_message_block_count_under_50():
    # Slack has a 50-block limit per message
    top_stories = [make_summarized(f"Story {i}", "edutech", f"https://a.com/{i}") for i in range(7)]
    quick_links = [make_feed_item(f"Quick {i}", f"https://b.com/{i}") for i in range(5)]
    digest = DigestContent(
        top_stories=top_stories,
        quick_links=quick_links,
        insight="Insight text.",
        generated_at=FIXED_DATE,
    )
    blocks = build_slack_message(digest)
    assert len(blocks) <= 50
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_formatter.py -v
```

Expected: `ModuleNotFoundError: No module named 'formatter'`

- [ ] **Step 3: Write formatter.py**

```python
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from fetcher import FeedItem
from summarizer import SummarizedItem


WIB_OFFSET = timedelta(hours=7)


@dataclass(frozen=True)
class DigestContent:
    top_stories: list[SummarizedItem]
    quick_links: list[FeedItem]
    insight: str
    generated_at: datetime  # UTC


def _divider() -> dict:
    return {"type": "divider"}


def _section(text: str) -> dict:
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": text},
    }


def _header(text: str) -> dict:
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": text, "emoji": True},
    }


def _context(text: str) -> dict:
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": text}],
    }


def _format_date(dt: datetime) -> str:
    wib_dt = dt + WIB_OFFSET
    return wib_dt.strftime("%A, %d %B %Y")


def _format_time(dt: datetime) -> str:
    wib_dt = dt + WIB_OFFSET
    return wib_dt.strftime("%H:%M")


def _format_top_story(index: int, item: SummarizedItem) -> dict:
    text = (
        f"*{index}. {item.title}*\n"
        f"{item.summary}\n"
        f"🔗 <{item.url}|{item.source_name}> · _{item.domain}_"
    )
    return _section(text)


def _format_quick_links(items: list[FeedItem]) -> dict:
    lines = [f"• <{item.url}|{item.title}> · _{item.source_name}_" for item in items]
    return _section("🔗 *QUICK LINKS*\n" + "\n".join(lines))


def build_slack_message(digest: DigestContent) -> list[dict]:
    date_str = _format_date(digest.generated_at)
    time_str = _format_time(digest.generated_at)

    blocks: list[dict] = []

    # Header
    blocks.append(_header(f"📰 Luyin's Daily Digest — {date_str}"))
    blocks.append(_divider())

    # Top Stories
    blocks.append(_section("🔥 *TOP STORIES*"))
    blocks.append(_divider())
    for i, story in enumerate(digest.top_stories, start=1):
        blocks.append(_format_top_story(i, story))

    blocks.append(_divider())

    # Insight & Advisory
    blocks.append(_section("💡 *INSIGHT & ADVISORY*"))
    # Split insight into paragraphs to respect Slack's 3000-char block limit
    paragraphs = [p.strip() for p in digest.insight.split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        blocks.append(_section(paragraph))

    blocks.append(_divider())

    # Quick Links
    if digest.quick_links:
        blocks.append(_format_quick_links(digest.quick_links))
        blocks.append(_divider())

    # Footer
    blocks.append(
        _context(f"_This digest created by Lú yīn at {time_str} WIB, {date_str}_")
    )

    return blocks
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_formatter.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add formatter.py tests/test_formatter.py
git commit -m "feat: slack block kit formatter with footer and quick links"
```

---

## Task 5: Slack Publisher

**Files:**
- Create: `luyin-daily-digest-system/publisher.py`
- Create: `luyin-daily-digest-system/tests/test_publisher.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_publisher.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_publisher.py -v
```

Expected: `ModuleNotFoundError: No module named 'publisher'`

- [ ] **Step 3: Write publisher.py**

```python
import httpx


def post_to_slack(webhook_url: str, blocks: list[dict]) -> None:
    payload = {"blocks": blocks}
    response = httpx.post(webhook_url, json=payload, timeout=10.0)
    if response.status_code != 200 or response.text != "ok":
        raise RuntimeError(
            f"Slack webhook returned {response.status_code}: {response.text}"
        )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_publisher.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add publisher.py tests/test_publisher.py
git commit -m "feat: slack webhook publisher with error handling"
```

---

## Task 6: Lambda Handler (Main Orchestrator)

**Files:**
- Create: `luyin-daily-digest-system/main.py`
- Create: `luyin-daily-digest-system/tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_main.py`:

```python
from unittest.mock import MagicMock, patch, call
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
@patch("main.summarize_item")
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
    mock_summarize.side_effect = make_summarized(4)
    mock_insight.return_value = "Insight text."
    mock_format.return_value = [{"type": "section"}]

    run_digest()

    mock_post.assert_called_once()


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
@patch("main.summarize_item")
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
    mock_format.return_value = [{"type": "section"}]

    run_digest()  # Should not raise

    mock_post.assert_called_once()
    # Fallback message should mention summaries unavailable
    call_blocks = mock_format.call_args[0][0]
    insight_text = call_blocks.insight
    assert "unavailable" in insight_text.lower()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_main.py -v
```

Expected: `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write main.py**

```python
from datetime import datetime, timezone

from config import load_config
from fetcher import load_sources, fetch_all_sources, filter_recent, deduplicate, quota_select, FeedItem
from summarizer import make_gemini_model, summarize_item, generate_insight, SummarizedItem
from formatter import DigestContent, build_slack_message
from publisher import post_to_slack


MINIMUM_ITEMS_THRESHOLD = 3
FALLBACK_INSIGHT = "⚠️ Summaries unavailable today — Gemini API could not be reached. Headlines and links are sourced directly from RSS feeds."


def run_digest() -> None:
    config = load_config()
    model = make_gemini_model(config.gemini_api_key)
    now = datetime.now(tz=timezone.utc)

    sources = load_sources()
    raw_items = fetch_all_sources(sources)
    recent_items = filter_recent(items=raw_items, now=now, max_age_hours=24)
    unique_items = deduplicate(recent_items)
    top_candidates, quick_links = quota_select(unique_items)

    if len(top_candidates) < MINIMUM_ITEMS_THRESHOLD:
        print(f"Only {len(top_candidates)} items found after deduplication. Skipping digest.")
        return

    try:
        top_stories: list[SummarizedItem] = [
            summarize_item(item=item, model=model) for item in top_candidates
        ]
        insight = generate_insight(items=top_stories, model=model)
    except Exception as exc:
        print(f"WARNING: Gemini API failed: {exc}. Posting fallback digest.")
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
    print(f"Digest posted successfully at {now.isoformat()}")


def handler(event: dict, context: object) -> dict:
    """AWS Lambda entry point."""
    run_digest()
    return {"statusCode": 200, "body": "Digest posted"}
```

- [ ] **Step 4: Run all tests to confirm they pass**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: lambda handler orchestrating full digest pipeline with fallback"
```

---

## Task 7: Deploy to AWS Lambda

**Files:**
- Create: `luyin-daily-digest-system/deploy/lambda_deploy.sh`

- [ ] **Step 1: Prerequisites — verify AWS CLI is configured**

```bash
aws sts get-caller-identity
```

Expected: JSON output with your AWS account ID. If not, run `aws configure` and enter your credentials.

- [ ] **Step 2: Write deploy/lambda_deploy.sh**

```bash
#!/bin/bash
set -e

FUNCTION_NAME="luyin-daily-digest"
REGION="ap-southeast-1"
RUNTIME="python3.12"
HANDLER="main.handler"
ROLE_ARN="arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-basic-execution"  # replace with your role ARN

echo "==> Packaging Lambda function..."
rm -rf /tmp/lambda-package
mkdir -p /tmp/lambda-package

# Install dependencies into package dir
pip install -r requirements.txt -t /tmp/lambda-package --quiet

# Copy source files
cp main.py fetcher.py summarizer.py formatter.py publisher.py config.py sources.yaml /tmp/lambda-package/

# Zip it
cd /tmp/lambda-package
zip -r /tmp/lambda-package.zip . -q
cd -

echo "==> Deploying to AWS Lambda..."
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" &>/dev/null; then
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file fileb:///tmp/lambda-package.zip \
        --region "$REGION"
    echo "==> Function updated."
else
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime "$RUNTIME" \
        --role "$ROLE_ARN" \
        --handler "$HANDLER" \
        --zip-file fileb:///tmp/lambda-package.zip \
        --timeout 120 \
        --memory-size 256 \
        --region "$REGION"
    echo "==> Function created."
fi

echo "==> Setting environment variables..."
aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --environment "Variables={GEMINI_API_KEY=$GEMINI_API_KEY,SLACK_WEBHOOK_URL=$SLACK_WEBHOOK_URL}" \
    --region "$REGION"

echo "==> Done. Lambda deployed: $FUNCTION_NAME"
```

- [ ] **Step 3: Create IAM role for Lambda (run once in AWS Console or CLI)**

```bash
# Create the execution role
aws iam create-role \
    --role-name lambda-basic-execution \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach basic execution policy (CloudWatch Logs)
aws iam attach-role-policy \
    --role-name lambda-basic-execution \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

Expected: Role ARN printed. Copy it into `ROLE_ARN` in `lambda_deploy.sh`.

- [ ] **Step 4: Deploy the function**

```bash
export GEMINI_API_KEY=your_gemini_api_key
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
chmod +x deploy/lambda_deploy.sh
./deploy/lambda_deploy.sh
```

Expected: `==> Done. Lambda deployed: luyin-daily-digest`

- [ ] **Step 5: Test the Lambda function manually**

```bash
aws lambda invoke \
    --function-name luyin-daily-digest \
    --region ap-southeast-1 \
    --payload '{}' \
    /tmp/lambda-response.json
cat /tmp/lambda-response.json
```

Expected: `{"statusCode": 200, "body": "Digest posted"}` and a Slack message in the channel.

- [ ] **Step 6: Create EventBridge rule for daily 08:00 WIB trigger**

```bash
# Create the scheduled rule (cron = 01:00 UTC = 08:00 WIB)
aws events put-rule \
    --name "luyin-digest-daily" \
    --schedule-expression "cron(0 1 * * ? *)" \
    --state ENABLED \
    --region ap-southeast-1

# Grant EventBridge permission to invoke the Lambda
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws lambda add-permission \
    --function-name luyin-daily-digest \
    --statement-id EventBridgeDailyTrigger \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:ap-southeast-1:${ACCOUNT_ID}:rule/luyin-digest-daily" \
    --region ap-southeast-1

# Attach Lambda as the target
LAMBDA_ARN=$(aws lambda get-function --function-name luyin-daily-digest --region ap-southeast-1 --query Configuration.FunctionArn --output text)
aws events put-targets \
    --rule luyin-digest-daily \
    --targets "Id=1,Arn=${LAMBDA_ARN}" \
    --region ap-southeast-1
```

Expected: `{"FailedEntryCount": 0, "FailedEntries": []}` — rule is active.

- [ ] **Step 7: Verify RSS feed URLs in sources.yaml**

Open each URL marked `# verify URL` in sources.yaml in a browser. If a URL returns no content or 404, find the correct RSS feed URL for that source and update sources.yaml.

```bash
# Quick check — feedparser returns status 200 for valid feeds
python -c "
import feedparser, yaml
with open('sources.yaml') as f:
    sources = yaml.safe_load(f)['sources']
for s in sources:
    d = feedparser.parse(s['url'])
    status = d.get('status', 'N/A')
    count = len(d.entries)
    print(f'{s[\"name\"]}: status={status}, entries={count}')
"
```

Expected: Each source shows `status=200` and `entries > 0`. Fix any that don't.

- [ ] **Step 8: Final commit**

```bash
git add deploy/lambda_deploy.sh
git commit -m "feat: aws lambda deploy script and eventbridge schedule"
```

---

## Self-Review Checklist (completed)

| Spec Requirement | Covered in Task |
|---|---|
| 08:00 WIB daily schedule | Task 7 — EventBridge cron(0 1 * * ? *) |
| Slack Incoming Webhook delivery | Task 5 — publisher.py |
| Gemini Flash summarization | Task 3 — summarizer.py |
| Tone & voice rules in prompts | Task 3 — TONE_RULES constant |
| 5-7 top stories with summary + link | Task 6 — DigestContent top_stories |
| Insight & Advisory section | Task 3 — generate_insight() |
| Quick Links (headlines + links) | Task 4 — formatter.py quick links section |
| Footer: Lú yīn + date | Task 4 — formatter.py footer |
| Domain weighting 50/30/20 | Task 2 — DOMAIN_QUOTAS in fetcher.py |
| URL + fuzzy title deduplication | Task 2 — deduplicate() |
| Empty day handling (<3 items) | Task 6 — MINIMUM_ITEMS_THRESHOLD |
| Gemini fallback on API failure | Task 6 — try/except in run_digest() |
| sources.yaml editable config | Task 1 — sources.yaml |
| AWS Lambda + EventBridge infra | Task 7 — deploy script |
| Env vars for secrets | Task 1 — config.py + Lambda env vars |
