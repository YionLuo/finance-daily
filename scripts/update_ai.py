#!/usr/bin/env python3
"""
AI Daily Report generator for Y Daily.
Runs daily at 22:10 CST via GitHub Actions.

Flow:
1. Fetch real AI/tech news from RSS feeds (TechCrunch, The Verge, Google News, etc.)
2. Use LLM to analyze and generate structured AI report
3. Inject new AI issue into index.html
4. Validate JS syntax
"""

import os
import sys
import json
import re
import subprocess
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    read_html, write_html,
    extract_js_array,
    replace_js_array, replace_js_string,
    format_date_cst, now_cst, CST,
    create_llm_client, llm_chat_with_retry,
)
from news_fetcher import (
    fetch_ai_news, articles_to_context, dedup_by_title,
)

FOCUS_AREAS = "大模型 · 智能体 · 具身智能 · AI Coding · AI for Science"

LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")


def generate_ai_report(client, news_context):
    """Use LLM to generate the daily AI report based on REAL news."""
    now = now_cst()
    yesterday = now - timedelta(days=1)
    time_range = f"{yesterday.strftime('%Y.%m.%d')} – {now.strftime('%Y.%m.%d')}"
    date_cn = f"{now.year}年{now.month}月{now.day}日"
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekdays[now.weekday()]

    prompt = f"""You are the chief AI analyst at Y Daily (yion.me), an AI intelligence platform.
Current time: {format_date_cst(now)}

Below are REAL AI/tech news articles from the past 24 hours, fetched from RSS feeds (TechCrunch, The Verge, Ars Technica, VentureBeat, Google News, etc.).

=== TODAY'S AI NEWS ===
{news_context}
=== END NEWS ===

Generate today's AI daily report based on THESE REAL articles above.

FOCUS AREAS: {FOCUS_AREAS}
Key players to track: OpenAI, Anthropic, Google DeepMind, Meta AI, 字节(豆包), 阿里(通义千问), 腾讯(混元), 智谱, DeepSeek, 百度(文心), NVIDIA, Apple

You MUST produce a complete AI issue object matching this EXACT structure:

{{
  "id": "ai-{now.strftime('%Y-%m-%d')}",
  "date": "{date_cn}",
  "weekday": "{weekday}",
  "title": "50 chars max, key AI events summary",
  "summary": "100 chars max, core narrative",
  "tags": [
    {{"text": "tag text", "type": "up|down|warn"}},
    // 3-5 tags
  ],
  "focusAreas": "{FOCUS_AREAS}",
  "timeRange": "{time_range}",
  "generatedAt": "{format_date_cst(now)}",
  "briefings": [
    // 3 HTML strings with <span class=\\"briefing-number\\">N</span><strong>...
  ],
  "research": [
    {{
      "icon": "emoji",
      "iconBg": "#hex",
      "title": "research title",
      "detail": "research detail",
      "impact": "impact assessment",
      "source": "source name",
      "url": "source URL from articles above"
    }}
    // 2-3 research items
  ],
  "application": [
    {{
      "icon": "emoji",
      "iconBg": "#hex",
      "title": "application title",
      "detail": "detail",
      "impact": "impact",
      "source": "source name",
      "url": "source URL"
    }}
    // 2-3 application items
  ],
  "industry": [
    {{
      "icon": "emoji",
      "iconBg": "#hex",
      "title": "industry news title",
      "detail": "detail",
      "impact": "impact",
      "source": "source name",
      "url": "source URL"
    }}
    // 2-3 industry items
  ],
  "outlook": [
    {{"time": "timeframe", "event": "upcoming event", "focus": "area"}}
    // 4-6 forward-looking items
  ]
}}

CRITICAL RULES:
1. Base your report ONLY on the real articles provided above. Do NOT fabricate.
2. ALL numbers (parameters, funding, benchmarks) must come from the articles.
3. Include source URLs from the articles in research, application, and industry items.
4. Use Chinese for all text content.
5. Include HTML formatting where appropriate.
6. If there's not enough data for a section, use fewer items rather than fabricating.

Return ONLY the JSON object. No markdown fencing.
"""

    try:
        raw = llm_chat_with_retry(
            client, [{"role": "user", "content": prompt}],
            model=LLM_MODEL, max_tokens=8000, temperature=0.2, max_retries=3,
        )
        content = re.sub(r'^```json\s*', '', raw)
        content = re.sub(r'\s*```$', '', content)
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse LLM response as JSON: {e}")
        return None
    except Exception as e:
        print(f"ERROR generating AI report: {e}")
        return None


def validate_report(report):
    """Basic validation of the AI report structure."""
    required_keys = ["id", "date", "weekday", "title", "summary", "tags",
                     "briefings", "research", "application", "industry", "outlook"]
    for key in required_keys:
        if key not in report:
            print(f"ERROR: Missing key '{key}' in report")
            return False
    return True


def validate_js(html_path):
    """Validate JS syntax."""
    try:
        node_paths = ["node", os.path.expanduser("~/.workbuddy/binaries/node/versions/22.12.0/bin/node")]
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
        print("WARNING: node not found, skipping validation")
        return True
    except Exception as e:
        print(f"JS validation error: {e}")
        return False


def main():
    print("=" * 60)
    print(f"AI Daily Report - {format_date_cst()}")
    print("=" * 60)

    client = create_llm_client(required=True)
    html = read_html()

    # Step 1: Fetch REAL AI news from RSS feeds
    print("\n--- Fetching AI news from RSS feeds ---")
    raw_ai = fetch_ai_news(max_age_hours=24)
    raw_ai = dedup_by_title(raw_ai)
    print(f"Total AI articles: {len(raw_ai)}")

    news_context = articles_to_context(raw_ai, max_articles=40)

    # Step 2: Generate report
    print("\n--- Generating AI daily report ---")
    report = generate_ai_report(client, news_context)
    if not report:
        print("ERROR: Failed to generate report")
        sys.exit(1)

    if not validate_report(report):
        print("ERROR: Report validation failed")
        sys.exit(1)

    print(f"Title: {report['title']}")

    # Step 3: Insert/replace in aiIssues
    ai_issues = extract_js_array(html, 'aiIssues')
    today_id = report["id"]
    existing_idx = next((i for i, iss in enumerate(ai_issues) if iss.get("id") == today_id), None)
    if existing_idx is not None:
        print(f"Replacing existing AI issue for {today_id}")
        ai_issues[existing_idx] = report
    else:
        print(f"Adding new AI issue for {today_id}")
        ai_issues.insert(0, report)

    html = replace_js_array(html, 'aiIssues', ai_issues)

    # Step 4: Write and validate
    write_html(html)

    html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'index.html'))
    if validate_js(html_path):
        print("\nJS validation: OK")
    else:
        print("\nERROR: JS validation failed!")
        sys.exit(1)

    print(f"\nDone! AI report for {report['date']} generated successfully.")


if __name__ == "__main__":
    main()
