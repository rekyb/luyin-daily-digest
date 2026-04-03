from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import pytest

from fetcher import (
    FeedItem,
    Source,
    fetch_all_sources,
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

    assert len(edutech_top) == 4   # quota is 4, we provided 6
    assert len(ai_top) == 2        # quota is 2, we provided 4
    assert len(biz_top) == 1       # quota is 1, we provided 3
    assert len(top) == 7
    assert len(quick) == 5


def test_fetch_all_sources_continues_on_source_failure():
    good_source = Source(name="Good", url="http://good.com/feed", domain="edutech")
    bad_source = Source(name="Bad", url="http://bad.com/feed", domain="ai")

    good_item = make_item("Good story", "http://good.com/1", "edutech")

    def fake_fetch(source: Source) -> list[FeedItem]:
        if source.name == "Bad":
            raise ValueError("bozo feed")
        return [good_item]

    with patch("fetcher.fetch_feed", side_effect=fake_fetch):
        result = fetch_all_sources([good_source, bad_source])

    assert len(result) == 1
    assert result[0].title == "Good story"


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
