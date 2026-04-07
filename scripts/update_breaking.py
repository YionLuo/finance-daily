#!/usr/bin/env python3
"""
Breaking News updater for Y Daily.
Runs hourly via GitHub Actions.

Flow:
1. Fetch real news from RSS feeds (Google News, CNBC, Yahoo, MarketWatch, etc.)
2. Use LLM to analyze, translate, tag, and structure the news
3. Apply 24h sliding window (remove stale entries)
4. Dedup against existing items
5. Merge new items into index.html
6. Validate JS syntax with node
"""

import os
import sys
import json
import re
import subprocess
from datetime import datetime, timezone, timedelta

# Allow importing from scripts/
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    read_html, write_html,
    extract_js_array, extract_js_string,
    replace_js_array, replace_js_string,
    format_date_cst, now_cst, CST,
    python_to_js_object_inline
)
from news_fetcher import (
    fetch_finance_news, fetch_ai_news, fetch_watchlist_news,
    articles_to_context, dedup_by_title,
)

# OpenAI for news analysis
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ============ Configuration ============

MAX_ITEMS = 20
WINDOW_HOURS = 24
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v3.2")


def get_openai_client():
    """Initialize OpenAI client with optional custom base URL."""
    if OpenAI is None:
        print("WARNING: openai package not installed. Cleanup-only mode.")
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. Using cleanup-only mode.")
        return None
    kwargs = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def analyze_news_with_llm(client, articles_context, category, existing_texts):
    """
    Use LLM to analyze real news articles and produce structured breaking news items.
    The LLM receives REAL articles from RSS feeds and transforms them.
    """
    if not client or not articles_context.strip():
        return []

    now = now_cst()
    time_str = now.strftime("%Y-%m-%d %H:%M CST")

    existing_str = "\n".join(f"- {t[:60]}" for t in list(existing_texts)[:15])

    prompt = f"""You are a professional financial news analyst for Y Daily (yion.me).
Current time: {time_str}

Below are REAL news articles fetched from RSS feeds in the past few hours.
Your task: Select the most important and impactful ones, translate to Chinese, and structure them.

Category: {"Finance/Macro/Geopolitics" if category == "finance" else "AI/Tech"}

=== RAW ARTICLES ===
{articles_context}
=== END ARTICLES ===

=== EXISTING ITEMS (avoid duplicates) ===
{existing_str if existing_str else "(none)"}
=== END EXISTING ===

Rules:
1. ONLY select genuinely important/impactful news. Skip fluff, opinion pieces, and minor updates.
2. Translate to Chinese. Be concise but complete.
3. Each item MUST have a real source URL from the articles above.
4. DO NOT duplicate any existing items listed above.
5. Aim for 3-8 high-quality items. Quality over quantity.
6. Tag each item appropriately.

Return a JSON array. Each item:
{{
  "time": "HH:MM",  // CST time (estimate from published time)
  "tag": "geo|macro|ai|tech|crypto",
  "tagText": "地缘|宏观|AI|科技|加密",
  "text": "Chinese news text, concise but informative",
  "source": "Source Name",
  "url": "article URL from the feed"
}}

If no articles are worth including (all fluff or duplicates), return [].
Return ONLY the JSON array, no markdown fencing.
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000,
        )
        raw = response.choices[0].message.content
        if not raw:
            print("LLM returned empty content")
            return []
        content = raw.strip()
        # Strip markdown code fences if present
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        items = json.loads(content)
        if not isinstance(items, list):
            return []
        return items
    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        return []


def clean_item(item):
    """Clean an item for output."""
    return {
        "time": item.get("time", ""),
        "tag": item.get("tag", "tech"),
        "tagText": item.get("tagText", "科技"),
        "text": item.get("text", ""),
        "source": item.get("source", ""),
        "url": item.get("url", ""),
    }


def dedup_items(existing, new_items):
    """Remove duplicates based on text similarity."""
    existing_texts = set()
    for item in existing:
        text = item.get("text", "")[:30]
        existing_texts.add(text)

    deduped = []
    for item in new_items:
        text = item.get("text", "")[:30]
        if text not in existing_texts:
            deduped.append(item)
            existing_texts.add(text)

    return deduped


def filter_by_window(items, window_hours=24):
    """Remove items older than window_hours."""
    now = now_cst()
    filtered = []

    for item in items:
        time_str = item.get("time", "")
        try:
            hour, minute = map(int, time_str.split(":"))
            item_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if item_time > now:
                item_time -= timedelta(days=1)
            if (now - item_time).total_seconds() <= window_hours * 3600:
                filtered.append(item)
            else:
                print(f"  Removed stale item [{time_str}]: {item.get('text', '')[:50]}...")
        except (ValueError, TypeError):
            filtered.append(item)

    return filtered


def sort_by_time_desc(items):
    """
    Sort items by time (HH:MM) from newest to oldest.
    Handles overnight wrap: if current hour < 12, times > 12 are considered yesterday.
    Otherwise, all times are today, with larger hours being more recent.
    """
    now = now_cst()

    def time_sort_key(item):
        time_str = item.get("time", "")
        try:
            hour, minute = map(int, time_str.split(":"))
            item_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            # If item time is in the future, it was yesterday
            if item_time > now:
                item_time -= timedelta(days=1)
            return item_time
        except (ValueError, TypeError):
            # Can't parse → put at the end
            return now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=2)

    return sorted(items, key=time_sort_key, reverse=True)


def validate_js(html_path):
    """Validate JS syntax using node."""
    try:
        node_paths = [
            "node",
            os.path.expanduser("~/.workbuddy/binaries/node/versions/22.12.0/bin/node"),
        ]
        for node in node_paths:
            try:
                result = subprocess.run(
                    [node, "-e",
                     f"const fs=require('fs');const h=fs.readFileSync('{html_path}','utf8');"
                     f"const m=h.match(/<script>([\\\\s\\\\S]*?)<\\\\/script>/);"
                     f"try{{new Function(m[1]);console.log('OK')}}catch(e){{console.log('ERR:',e.message)}}"],
                    capture_output=True, text=True, timeout=10,
                )
                if "OK" in result.stdout:
                    return True
                if "ERR:" in result.stdout:
                    print(f"JS validation error: {result.stdout}")
                    return False
            except FileNotFoundError:
                continue

        print("WARNING: node not found, skipping JS validation")
        return True
    except Exception as e:
        print(f"JS validation error: {e}")
        return False


def main():
    print("=" * 60)
    print(f"Breaking News Update - {format_date_cst()}")
    print("=" * 60)

    # Read current HTML
    html = read_html()

    # Extract current breaking news arrays
    finance_news = extract_js_array(html, 'breakingNews')
    ai_news = extract_js_array(html, 'aiBreakingNews')

    print(f"Current finance breaking news: {len(finance_news)} items")
    print(f"Current AI breaking news: {len(ai_news)} items")

    # Step 1: Apply 24h window to existing items
    print("\n--- Applying 24h window filter ---")
    finance_news = filter_by_window(finance_news, WINDOW_HOURS)
    ai_news = filter_by_window(ai_news, WINDOW_HOURS)
    print(f"After filtering: finance={len(finance_news)}, ai={len(ai_news)}")

    # Collect existing texts for dedup
    existing_finance_texts = {item.get("text", "")[:60] for item in finance_news}
    existing_ai_texts = {item.get("text", "")[:60] for item in ai_news}

    # Step 2: Fetch REAL news from RSS feeds
    print("\n--- Fetching news from RSS feeds ---")
    raw_finance = fetch_finance_news(max_age_hours=6)
    raw_finance += fetch_watchlist_news(max_age_hours=6)
    raw_finance = dedup_by_title(raw_finance)
    print(f"Total finance articles from RSS: {len(raw_finance)}")

    raw_ai = fetch_ai_news(max_age_hours=6)
    raw_ai = dedup_by_title(raw_ai)
    print(f"Total AI articles from RSS: {len(raw_ai)}")

    # Step 3: Use LLM to analyze and structure
    client = get_openai_client()

    if client and raw_finance:
        print("\n--- Analyzing finance news with LLM ---")
        finance_context = articles_to_context(raw_finance, max_articles=30)
        new_finance = analyze_news_with_llm(client, finance_context, "finance", existing_finance_texts)
        print(f"LLM produced {len(new_finance)} finance items")
    elif not raw_finance:
        print("\nNo new finance articles from RSS")
        new_finance = []
    else:
        print("\nNo LLM client - skipping analysis")
        new_finance = []

    if client and raw_ai:
        print("\n--- Analyzing AI news with LLM ---")
        ai_context = articles_to_context(raw_ai, max_articles=30)
        new_ai = analyze_news_with_llm(client, ai_context, "ai", existing_ai_texts)
        print(f"LLM produced {len(new_ai)} AI items")
    elif not raw_ai:
        print("\nNo new AI articles from RSS")
        new_ai = []
    else:
        print("\nNo LLM client - skipping analysis")
        new_ai = []

    # Step 4: Dedup and merge
    new_finance = dedup_items(finance_news, [clean_item(i) for i in new_finance])
    new_ai = dedup_items(ai_news, [clean_item(i) for i in new_ai])

    print(f"\nNew unique finance items: {len(new_finance)}")
    print(f"New unique AI items: {len(new_ai)}")

    # Prepend new items and sort by time (newest first)
    finance_news = new_finance + finance_news
    ai_news = new_ai + ai_news

    finance_news = sort_by_time_desc(finance_news)
    ai_news = sort_by_time_desc(ai_news)

    # Step 5: Trim to max items
    finance_news = finance_news[:MAX_ITEMS]
    ai_news = ai_news[:MAX_ITEMS]

    # Step 6: Update HTML
    now_str = format_date_cst()
    html = replace_js_array(html, 'breakingNews', finance_news)
    html = replace_js_string(html, 'breakingDate', now_str)
    html = replace_js_array(html, 'aiBreakingNews', ai_news)
    html = replace_js_string(html, 'aiBreakingDate', now_str)

    # Write
    write_html(html)
    print(f"\n--- Updated index.html ---")
    print(f"Finance breaking news: {len(finance_news)} items")
    print(f"AI breaking news: {len(ai_news)} items")
    print(f"Updated time: {now_str}")

    # Step 7: Validate JS
    html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'index.html'))
    if validate_js(html_path):
        print("JS validation: OK")
    else:
        print("ERROR: JS validation failed! Reverting...")
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
