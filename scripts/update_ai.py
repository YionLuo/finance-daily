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

LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek/deepseek-chat")


def generate_ai_report(client, news_context):
    """Use LLM to generate the daily AI report based on REAL news."""
    now = now_cst()
    yesterday = now - timedelta(days=1)
    time_range = f"{yesterday.strftime('%Y.%m.%d')} – {now.strftime('%Y.%m.%d')}"
    date_cn = f"{now.year}年{now.month}月{now.day}日"
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekdays[now.weekday()]

    prompt = f"""You are an AI industry analyst at Y Daily (yion.me). Your readers are AI practitioners — engineers, PMs, founders — who live in this space daily. They don't need you to explain what a transformer is. They need you to tell them what's ACTUALLY shifting in the industry.

Current time: {format_date_cst(now)}

=== TODAY'S AI NEWS (real, from RSS) ===
{news_context}
=== END NEWS ===

FOCUS AREAS: {FOCUS_AREAS}
Key players: OpenAI, Anthropic, Google DeepMind, Meta AI, 字节(豆包), 阿里(通义千问), 腾讯(混元), 智谱, DeepSeek, 百度(文心), NVIDIA, Apple

=== YOUR ANALYTICAL FRAMEWORK ===

Before writing, think through (do NOT output thinking, only final JSON):

1. POWER DYNAMICS: Who gained or lost leverage today? A model release isn't just a product — it's a competitive move. What does it mean for the ecosystem?

2. TECHNICAL SIGNIFICANCE: Is this a genuine capability leap, or incremental marketing? Your readers can tell the difference. Be honest — "this is iteration, not revolution" is a valid take.

3. SECOND-ORDER EFFECTS: A new model is news. What it means for API pricing, open-source dynamics, talent wars, or regulation — that's insight.

4. THE PRACTITIONER'S QUESTION: For someone building AI products right now, what does today's news mean for their stack, their roadmap, or their hiring?

5. WHAT TO IGNORE: Not every press release matters. If something is noise, skip it. Fewer items with real insight > many items with surface takes.

=== OUTPUT FORMAT ===

JSON object. ALL text in Chinese. Be opinionated — your readers want a point of view, not a summary.

{{
  "id": "ai-{now.strftime('%Y-%m-%d')}",
  "date": "{date_cn}",
  "weekday": "{weekday}",
  "title": "≤50 chars — the TREND, not the event (bad: 'OpenAI发新模型', good: 'API价格战进入下半场')",
  "summary": "≤120 chars — one sentence a CTO would forward to their team",
  "tags": [
    {{"text": "tag", "type": "up|down|warn"}},
    // 3-5 tags
  ],
  "focusAreas": "{FOCUS_AREAS}",
  "timeRange": "{time_range}",
  "generatedAt": "{format_date_cst(now)}",
  "briefings": [
    // 3 HTML strings — NOT news summaries, but STRATEGIC TAKES.
    // <span class=\\"briefing-number\\">N</span><strong>Opinionated headline</strong> — Why a practitioner should care.
  ],
  "research": [
    {{
      "icon": "emoji",
      "iconBg": "#hex",
      "title": "What happened — be specific",
      "detail": "The technical substance — what's new and what's not. Skip the marketing language.",
      "impact": "So what? For practitioners. 'This means you should/shouldn't X because Y.'",
      "source": "source name",
      "url": "source URL"
    }}
    // 1-3 items. Only genuinely significant research. Incremental papers = skip.
  ],
  "application": [
    {{
      "icon": "emoji",
      "iconBg": "#hex",
      "title": "product/application title",
      "detail": "What it does and what's novel about the approach.",
      "impact": "Market implication — who should be worried, who benefits, what pattern does this confirm?",
      "source": "source name",
      "url": "source URL"
    }}
    // 1-3 items.
  ],
  "industry": [
    {{
      "icon": "emoji",
      "iconBg": "#hex",
      "title": "deal/hire/policy/strategy title",
      "detail": "The facts.",
      "impact": "Read between the lines — what does this signal about strategy, not just what was announced?",
      "source": "source name",
      "url": "source URL"
    }}
    // 1-3 items.
  ],
  "outlook": [
    {{"time": "next 1-4 weeks", "event": "specific event to watch", "focus": "why it matters to practitioners"}}
    // 3-5 items. Concrete things, not vague trends.
  ]
}}

RULES:
1. ONLY use data from articles above. Do NOT fabricate.
2. Thin news day? Write fewer items. Never pad.
3. ALL text in Chinese. HTML formatting as shown.
4. Return ONLY JSON. No markdown fencing.
"""

    try:
        raw = llm_chat_with_retry(
            client, [{"role": "user", "content": prompt}],
            model=LLM_MODEL, max_tokens=4096, temperature=0.5, max_retries=3,
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
