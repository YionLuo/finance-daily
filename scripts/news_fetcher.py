#!/usr/bin/env python3
"""
News fetcher module for Y Daily.
Fetches real news from multiple RSS feeds and free APIs.
Used by update_breaking.py, update_finance.py, and update_ai.py.
"""

import os
import re
import json
import hashlib
import feedparser
import requests
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus

# ============ Configuration ============

# RSS feed sources for financial/macro news
FINANCE_RSS_FEEDS = [
    # Google News - finance
    "https://news.google.com/rss/search?q=stock+market+today&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=breaking+financial+news&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=global+economy+macro&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=geopolitical+conflict+oil+energy&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=bitcoin+crypto+market&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=gold+silver+copper+commodity&hl=en&gl=US&ceid=US:en",
    # Google News - China tech
    "https://news.google.com/rss/search?q=China+tech+Tencent+Alibaba+Xiaomi&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=S%26P+500+Nasdaq+Dow+Jones&hl=en&gl=US&ceid=US:en",
    # Reuters & Bloomberg via Google News
    "https://news.google.com/rss/search?q=site:reuters.com+breaking+finance&hl=en&gl=US&ceid=US:en",
    # CNBC
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    # Yahoo Finance
    "https://finance.yahoo.com/news/rssindex",
    # MarketWatch
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    # FT
    "https://www.ft.com/rss/home",
]

# RSS feed sources for AI/tech news
AI_RSS_FEEDS = [
    # Google News - AI
    "https://news.google.com/rss/search?q=artificial+intelligence+breakthrough&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=large+language+model+GPT+Claude+Gemini&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=AI+chip+semiconductor+NVIDIA&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=AI+startup+funding&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=OpenAI+Anthropic+DeepMind+news&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=robotics+autonomous+driving+AI&hl=en&gl=US&ceid=US:en",
    # TechCrunch AI
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    # The Verge AI
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    # Ars Technica AI
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    # VentureBeat
    "https://venturebeat.com/category/ai/feed/",
]

# Targeted Google News searches for watchlist stocks
WATCHLIST_QUERIES = [
    "Tencent 0700 stock news",
    "Alibaba 9988 BABA stock news",
    "Xpeng XPEV stock news",
    "Meituan 3690 stock news",
    "Xiaomi 1810 stock news",
    "NVIDIA NVDA stock news",
    "Tesla TSLA stock news",
    "Apple AAPL stock news",
    "Microsoft MSFT stock news",
    "TSMC TSM semiconductor",
    "Broadcom AVGO stock",
    "Bitcoin BTC price",
    "Ethereum ETH crypto",
]

# Request timeout and user-agent
TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (Y Daily Bot/1.0; +https://yion.me)"


def _make_google_news_url(query):
    """Build a Google News RSS search URL."""
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"


def _parse_published(entry):
    """Parse the published date from an RSS entry. Returns datetime or None."""
    published = entry.get("published_parsed") or entry.get("updated_parsed")
    if published:
        try:
            return datetime(*published[:6], tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    # Try parsing from string
    date_str = entry.get("published") or entry.get("updated") or ""
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ]:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _clean_html(text):
    """Strip HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text or "").strip()


def fetch_rss_articles(feeds, max_age_hours=6, max_per_feed=10):
    """
    Fetch articles from a list of RSS feed URLs.
    Returns a list of article dicts with: title, summary, url, source, published.
    Only includes articles from the past max_age_hours.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    articles = []
    seen_urls = set()

    for feed_url in feeds:
        try:
            feed = feedparser.parse(
                feed_url,
                request_headers={"User-Agent": USER_AGENT},
            )
            count = 0
            for entry in feed.entries[:max_per_feed]:
                pub_date = _parse_published(entry)

                # Skip old articles
                if pub_date and pub_date < cutoff:
                    continue

                url = entry.get("link", "")
                # Dedup by URL
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = _clean_html(entry.get("title", ""))
                summary = _clean_html(entry.get("summary", entry.get("description", "")))
                source = entry.get("source", {}).get("title", "") or feed.feed.get("title", "")

                if not title:
                    continue

                articles.append({
                    "title": title[:300],
                    "summary": summary[:500],
                    "url": url,
                    "source": source,
                    "published": pub_date.isoformat() if pub_date else "",
                    "published_dt": pub_date,
                })
                count += 1

            if count > 0:
                print(f"  RSS [{feed.feed.get('title', feed_url[:50])}]: {count} articles")

        except Exception as e:
            print(f"  RSS error [{feed_url[:60]}]: {e}")
            continue

    # Sort by published date (newest first)
    articles.sort(key=lambda a: a.get("published_dt") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    # Remove internal field
    for a in articles:
        a.pop("published_dt", None)

    return articles


def fetch_watchlist_news(max_age_hours=6):
    """Fetch news specifically about watchlist stocks via Google News RSS."""
    feeds = [_make_google_news_url(q) for q in WATCHLIST_QUERIES]
    return fetch_rss_articles(feeds, max_age_hours=max_age_hours, max_per_feed=5)


def fetch_finance_news(max_age_hours=3):
    """Fetch financial/macro breaking news from RSS feeds."""
    return fetch_rss_articles(FINANCE_RSS_FEEDS, max_age_hours=max_age_hours)


def fetch_ai_news(max_age_hours=3):
    """Fetch AI/tech news from RSS feeds."""
    return fetch_rss_articles(AI_RSS_FEEDS, max_age_hours=max_age_hours)


def fetch_all_news(max_age_hours=3):
    """Fetch all news (finance + AI + watchlist)."""
    finance = fetch_finance_news(max_age_hours)
    ai = fetch_ai_news(max_age_hours)
    watchlist = fetch_watchlist_news(max_age_hours)

    print(f"\nTotal fetched: finance={len(finance)}, ai={len(ai)}, watchlist={len(watchlist)}")
    return {
        "finance": finance,
        "ai": ai,
        "watchlist": watchlist,
    }


def articles_to_context(articles, max_articles=30):
    """
    Convert articles to a text context string for LLM consumption.
    Truncates to max_articles to fit in context window.
    """
    lines = []
    for i, art in enumerate(articles[:max_articles], 1):
        pub = art.get("published", "unknown time")
        lines.append(f"[{i}] {art['title']}")
        if art.get("summary"):
            lines.append(f"    {art['summary'][:200]}")
        lines.append(f"    Source: {art.get('source', 'unknown')} | {pub}")
        lines.append(f"    URL: {art.get('url', '')}")
        lines.append("")
    return "\n".join(lines)


def dedup_by_title(articles, threshold=0.70):
    """Remove near-duplicate articles by title similarity using SequenceMatcher."""
    from difflib import SequenceMatcher

    result = []
    seen_keys = []  # normalized titles for comparison

    for art in articles:
        # Normalize: lowercase, remove non-alphanumeric, collapse spaces
        key = re.sub(r'\W+', ' ', art["title"].lower()).strip()

        is_dup = False
        for existing_key in seen_keys:
            # Quick length check: if lengths differ too much, skip
            len_ratio = min(len(key), len(existing_key)) / max(len(key), len(existing_key), 1)
            if len_ratio < 0.4:
                continue
            if SequenceMatcher(None, key[:60], existing_key[:60]).ratio() > threshold:
                is_dup = True
                break

        if not is_dup:
            result.append(art)
            seen_keys.append(key)

    return result


def fetch_topic_articles(keywords, max_age_hours=72, max_per_keyword=15):
    """
    Fetch articles related to a specific topic using keyword search via Google News RSS.

    Args:
        keywords: list of search keywords (English and/or Chinese)
        max_age_hours: how far back to search (default 72h for deep research)
        max_per_keyword: max articles per keyword search

    Returns:
        Deduplicated list of article dicts, sorted by recency.
    """
    all_articles = []
    feeds = [_make_google_news_url(kw) for kw in keywords]

    print(f"\n=== Topic Research: searching {len(keywords)} keywords ===")
    for kw, feed_url in zip(keywords, feeds):
        try:
            arts = fetch_rss_articles([feed_url], max_age_hours=max_age_hours, max_per_feed=max_per_keyword)
            print(f"  Keyword [{kw}]: {len(arts)} articles")
            all_articles.extend(arts)
        except Exception as e:
            print(f"  Keyword [{kw}] error: {e}")

    # Deduplicate
    deduped = dedup_by_title(all_articles)
    print(f"  Total after dedup: {len(deduped)} articles")
    return deduped


# ============ Self-test ============

if __name__ == "__main__":
    print("Testing news fetcher...")
    print("\n=== Finance News ===")
    fin = fetch_finance_news(max_age_hours=6)
    print(f"Got {len(fin)} finance articles")
    for a in fin[:5]:
        print(f"  - {a['title'][:80]}  [{a.get('source', '')}]")

    print("\n=== AI News ===")
    ai = fetch_ai_news(max_age_hours=6)
    print(f"Got {len(ai)} AI articles")
    for a in ai[:5]:
        print(f"  - {a['title'][:80]}  [{a.get('source', '')}]")
