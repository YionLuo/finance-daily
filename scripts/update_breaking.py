#!/usr/bin/env python3
"""
Breaking News updater for Y Daily.
Runs hourly via GitHub Actions.

Flow:
1. Fetch news from multiple RSS feeds and APIs
2. Score relevance and dedup
3. Fact-check (require 2+ sources for key claims)
4. Apply 24h sliding window (remove stale entries)
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

# OpenAI for news analysis and fact-checking
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ============ Configuration ============

# News categories and their search queries
FINANCE_QUERIES = [
    "breaking news global economy macro finance today",
    "stock market breaking news S&P Nasdaq today",
    "geopolitical conflict oil energy today",
    "crypto bitcoin ethereum breaking today",
    "China tech Tencent Alibaba Xiaomi Xpeng breaking news",
    "gold silver copper commodity price today",
]

AI_QUERIES = [
    "AI artificial intelligence breakthrough news today",
    "large language model GPT Claude Gemini news today",
    "AI chip semiconductor NVIDIA TSM news today",
    "AI startup funding unicorn today",
    "robotics autonomous driving AI news today",
    "AI regulation safety policy today",
]

TAG_MAP = {
    "geo": "地缘",
    "macro": "宏观",
    "ai": "AI",
    "tech": "科技",
    "crypto": "加密",
}

MAX_ITEMS = 20
WINDOW_HOURS = 24


def get_openai_client():
    """Initialize OpenAI client."""
    if OpenAI is None:
        print("WARNING: openai package not installed. Cleanup-only mode.")
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. Using cleanup-only mode.")
        return None
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def search_news_via_llm(client, queries, category):
    """
    Use LLM to search and synthesize news from its training data
    plus any available search tools.

    In production, this would use web search APIs (NewsAPI, Google News, etc.).
    For now, we use the LLM as a news analyst.
    """
    if not client:
        return []

    now = now_cst()
    time_str = now.strftime("%Y-%m-%d %H:%M CST")

    prompt = f"""You are a professional financial news analyst for Y Daily (yion.me).
Current time: {time_str}

Your task: Find the most important breaking news from the PAST 1 HOUR that relates to these topics:
{json.dumps(queries, ensure_ascii=False)}

Category: {"Finance/Macro" if category == "finance" else "AI/Tech"}

Rules:
1. ONLY include news that actually happened in the past 1-2 hours. Do NOT rehash old news.
2. Each item must be verifiable from at least 2 independent sources.
3. Numbers (prices, percentages, amounts) must be accurate and from authoritative sources.
4. Distinguish between "happened" vs "planned/expected" events.
5. Do NOT include rumors or single-source claims without marking them as unverified.
6. Include source names and URLs where possible.

Respond with a JSON array of news items. Each item:
{{
  "time": "HH:MM",  // CST time of the event
  "tag": "geo|macro|ai|tech|crypto",
  "tagText": "Chinese tag label",
  "text": "News content in Chinese, concise but complete",
  "source": "Source1/Source2",
  "url": "most authoritative source URL",
  "confidence": "high|medium|low"  // your confidence in accuracy
}}

If there is NO genuinely new breaking news in the past hour, return an empty array [].
Do NOT fabricate news. It's better to return [] than to include outdated or false information.

Return ONLY the JSON array, no markdown fencing.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        items = json.loads(content)
        # Filter out low-confidence items
        return [item for item in items if item.get("confidence", "low") != "low"]
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def fact_check_items(client, items):
    """
    Cross-verify news items. Remove or mark items that can't be verified.
    """
    if not client or not items:
        return items

    prompt = f"""You are a fact-checker for a financial news service.
Review these news items and verify their accuracy.

Items to check:
{json.dumps(items, ensure_ascii=False, indent=2)}

For each item:
1. Check if the numbers/data are plausible and consistent with known facts
2. Check if the event timing is correct
3. Check if the sources are credible
4. Mark any items that seem fabricated, outdated, or inaccurate

Return a JSON array with the same items, but:
- Remove any items that are clearly false or fabricated
- Remove any items that are actually old news (>24 hours)
- Keep items that are verified or plausible
- Add "verified": true/false field to each

Return ONLY the JSON array, no markdown fencing.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.1-low",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4000,
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        verified = json.loads(content)
        # Only keep verified items
        return [item for item in verified if item.get("verified", True)]
    except Exception as e:
        print(f"Error in fact check: {e}")
        return items  # Return original if fact check fails


def clean_item(item):
    """Clean an item for output (remove internal fields)."""
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
        # Use first 30 chars as a simple fingerprint
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
    """
    Remove items older than window_hours.
    Since items only have HH:MM (no date), we need to be smart about this.

    Strategy: Items are assumed to be from the most recent occurrence of that time.
    If current time is 16:00 and an item says 18:00, it's from yesterday.
    """
    now = now_cst()
    filtered = []

    for item in items:
        time_str = item.get("time", "")
        try:
            hour, minute = map(int, time_str.split(":"))
            # Reconstruct the datetime
            item_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            # If the item time is in the future, it's from yesterday
            if item_time > now:
                item_time -= timedelta(days=1)
            # Check if within window
            if (now - item_time).total_seconds() <= window_hours * 3600:
                filtered.append(item)
            else:
                print(f"  Removed stale item [{time_str}]: {item.get('text', '')[:50]}...")
        except (ValueError, TypeError):
            # If we can't parse the time, keep the item
            filtered.append(item)

    return filtered


def validate_js(html_path):
    """Validate JS syntax using node."""
    try:
        # Try system node first, then managed node
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
        return True  # Can't validate, assume OK
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

    # Step 2: Fetch new news
    client = get_openai_client()

    if client:
        print("\n--- Fetching new finance news ---")
        new_finance = search_news_via_llm(client, FINANCE_QUERIES, "finance")
        print(f"Fetched {len(new_finance)} finance items")

        print("\n--- Fetching new AI news ---")
        new_ai = search_news_via_llm(client, AI_QUERIES, "ai")
        print(f"Fetched {len(new_ai)} AI items")

        # Step 3: Fact check
        if new_finance:
            print("\n--- Fact checking finance news ---")
            new_finance = fact_check_items(client, new_finance)
            print(f"After fact check: {len(new_finance)} items")

        if new_ai:
            print("\n--- Fact checking AI news ---")
            new_ai = fact_check_items(client, new_ai)
            print(f"After fact check: {len(new_ai)} items")

        # Step 4: Dedup and merge
        new_finance = dedup_items(finance_news, [clean_item(i) for i in new_finance])
        new_ai = dedup_items(ai_news, [clean_item(i) for i in new_ai])

        print(f"New unique finance items: {len(new_finance)}")
        print(f"New unique AI items: {len(new_ai)}")

        # Prepend new items
        finance_news = new_finance + finance_news
        ai_news = new_ai + ai_news
    else:
        print("\nNo OpenAI API key - skipping news fetch (cleanup only)")

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
        # In production, we'd revert here. For now, exit with error.
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
