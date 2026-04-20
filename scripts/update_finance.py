#!/usr/bin/env python3
"""
Finance Daily Report generator for Y Daily.
Runs daily at 22:00 CST via GitHub Actions.

Flow:
1. Fetch real news from RSS feeds (macro, stocks, watchlist)
2. Fetch Fear & Greed index from CNN
3. Use LLM to analyze and generate structured report
4. Inject new issue into index.html
5. Validate JS syntax
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
    extract_js_array, extract_js_string, extract_js_object,
    replace_js_array, replace_js_string, replace_js_object,
    format_date_cst, now_cst, CST,
    _dict_to_js,
    create_llm_client, llm_chat_with_retry,
)
from news_fetcher import (
    fetch_finance_news, fetch_watchlist_news,
    articles_to_context, dedup_by_title,
)

import requests

# ============ Configuration ============

WATCHLIST = {
    "hk": ["腾讯 0700", "阿里 9988", "小鹏 XPEV", "美团 3690", "小米 1810"],
    "us": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "semi": ["台积电 TSM", "博通 AVGO"],
    "crypto": ["BTC", "ETH"],
    "commodity": ["黄金", "白银", "铜"],
}

FOCUS_AREAS = "AI与大模型 · 半导体与算力 · 中美科技博弈 · 云计算与企业AI · 智能驾驶与机器人 · 科技监管与反垄断 · 宏观与流动性"

FEAR_GREED_API = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek/deepseek-chat")


def fetch_fear_greed():
    """Fetch CNN Fear & Greed Index data."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Y Daily Bot)"}
        resp = requests.get(FEAR_GREED_API, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            fg = data.get("fear_and_greed", {})
            return {
                "score": round(fg.get("score", 0), 2),
                "rating": fg.get("rating", "neutral"),
                "timestamp": fg.get("timestamp", ""),
                "previousClose": round(fg.get("previous_close", 0), 2),
                "previous1Week": round(fg.get("previous_1_week", 0), 2),
                "previous1Month": round(fg.get("previous_1_month", 0), 2),
                "previous1Year": round(fg.get("previous_1_year", 0), 2),
            }
    except Exception as e:
        print(f"Warning: Failed to fetch Fear & Greed data: {e}")
    return None


def generate_report(client, news_context, fg_data):
    """Use LLM to generate the daily finance report based on REAL news."""
    now = now_cst()
    yesterday = now - timedelta(days=1)
    time_range = f"{yesterday.strftime('%Y.%m.%d')} – {now.strftime('%Y.%m.%d')}"
    date_cn = f"{now.year}年{now.month}月{now.day}日"
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekdays[now.weekday()]

    watchlist_flat = []
    for group in WATCHLIST.values():
        watchlist_flat.extend(group)

    fg_str = ""
    if fg_data:
        fg_str = f"\nFear & Greed Index: {fg_data['score']} ({fg_data['rating']}), Previous close: {fg_data['previousClose']}, 1 week ago: {fg_data['previous1Week']}, 1 month ago: {fg_data['previous1Month']}"

    prompt = f"""You are the senior macro strategist at Y Daily (yion.me). Your readers are finance professionals and serious investors — they already know what happened. Your job is to tell them what it MEANS.

Current time: {format_date_cst(now)}

=== TODAY'S NEWS (real, from RSS) ===
{news_context}
=== END NEWS ===
{fg_str}

WATCHLIST: {json.dumps(watchlist_flat, ensure_ascii=False)}
FOCUS AREAS: {FOCUS_AREAS}

=== YOUR ANALYTICAL FRAMEWORK ===

Before writing the report, think through these steps (do NOT output your thinking, only the final JSON):

1. NARRATIVE THREAD: What is the ONE story that connects today's most important events? Not "5 things happened" — find the thread. (e.g., "The market is pricing in a policy pivot, but the bond market disagrees")

2. SECOND-ORDER EFFECTS: For each major event, ask "and then what?" at least twice. The obvious take is worthless — your readers already thought of it. Find the non-obvious implication.

3. CONTRADICTIONS: Where are markets or narratives contradicting each other? These are the most valuable signals.

4. WHAT'S MISSING: What SHOULD be in the news but isn't? Silence from a key player, a missing data point, or an event that didn't happen can be as important as what did.

5. ACTIONABLE EDGE: What does a professional need to watch in the next 24-72 hours based on today's events?

=== OUTPUT FORMAT ===

Produce a JSON object with this structure. ALL text in Chinese. Be opinionated — bland analysis is worse than wrong analysis.

{{
  "id": "{now.strftime('%Y-%m-%d')}",
  "date": "{date_cn}",
  "weekday": "{weekday}",
  "title": "≤50 chars — the narrative, not the headline (bad: '美股涨了', good: '市场在赌降息但债市不买账')",
  "summary": "≤120 chars — the 'so what' in one sentence, written for a fund manager",
  "tags": [
    {{"text": "恒生科技 +X.XX%", "type": "up|down|warn"}},
    // 4-5 tags, include market moves with real numbers from articles
  ],
  "watchlist": {json.dumps(watchlist_flat[:10], ensure_ascii=False)},
  "focusAreas": "{FOCUS_AREAS}",
  "timeRange": "{time_range}",
  "generatedAt": "{format_date_cst(now)}",
  "briefings": [
    // 3 HTML strings. NOT news summaries — these are ANALYTICAL TAKES.
    // Each: <span class=\\"briefing-number\\">N</span><strong>Sharp headline</strong> — Why it matters, not what happened.
    // Use <span class=\\"data up\\">+X%</span> / <span class=\\"data down\\">-X%</span> for real numbers.
  ],
  "macroEvents": [
    {{
      "icon": "emoji",
      "iconBg": "#hex",
      "title": "event title — be specific, not generic",
      "fact": "The key FACT with numbers (HTML spans for data). Keep it tight — one paragraph max.",
      "source": "source name",
      "url": "source URL from articles",
      "mappings": [
        // Bull/bear mappings should be NON-OBVIOUS. Not "利好科技股" — tell me WHY and WHICH ones.
        {{"type": "bull", "text": "<strong>Target</strong> — specific reasoning with second-order logic"}},
        {{"type": "bear", "text": "<strong>Target</strong> — the risk nobody is talking about"}}
      ]
    }}
    // 3-5 macro events. Quality > quantity. Skip if a news item is noise.
  ],
  "stocks": [
    {{
      "name": "emoji Name",
      "code": "TICKER",
      "news": [
        {{"label": "📰 headline", "content": "detail with <span class=\\"data\\">numbers</span>", "source": "source", "url": "URL"}}
      ],
      "assessment": "HTML — your TAKE on this stock. Not 'positive catalyst' — be specific about why, and what could go wrong."
    }}
    // Only watchlist stocks with REAL news. No news = skip, don't fabricate.
  ],
  "alerts": [
    {{"time": "next 24-72h", "event": "specific upcoming event", "target": "TICKER or macro"}}
    // 4-6 forward-looking items. Things your reader should literally put on their calendar.
  ]
}}

RULES:
1. ONLY use data from the articles above. Do NOT fabricate numbers, events, or URLs.
2. If articles are thin today, write fewer items — never pad with generic filler.
3. ALL text in Chinese. HTML formatting as shown.
4. Chinese market convention: 涨=红(up), 跌=绿(down).
5. Return ONLY the JSON object. No markdown fencing.
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
        print(f"ERROR generating report: {e}")
        return None


def validate_report(report):
    """Basic validation of the generated report structure."""
    required_keys = ["id", "date", "weekday", "title", "summary", "tags",
                     "briefings", "macroEvents", "stocks", "alerts"]
    for key in required_keys:
        if key not in report:
            print(f"ERROR: Missing key '{key}' in report")
            return False
    if len(report.get("briefings", [])) < 2:
        print("ERROR: Need at least 2 briefings")
        return False
    if len(report.get("macroEvents", [])) < 2:
        print("ERROR: Need at least 2 macro events")
        return False
    return True


def validate_js(html_path):
    """Validate JS syntax using node."""
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
        print("WARNING: node not found, skipping JS validation")
        return True
    except Exception as e:
        print(f"JS validation error: {e}")
        return False


def main():
    print("=" * 60)
    print(f"Finance Daily Report - {format_date_cst()}")
    print("=" * 60)

    client = create_llm_client(required=True)
    html = read_html()

    # Step 1: Fetch Fear & Greed
    print("\n--- Fetching Fear & Greed Index ---")
    fg_data = fetch_fear_greed()
    if fg_data:
        print(f"Score: {fg_data['score']} ({fg_data['rating']})")
        html = replace_js_object(html, 'fearGreedData', fg_data)
    else:
        print("Skipped (fetch failed)")

    # Step 2: Fetch REAL news from RSS feeds
    print("\n--- Fetching news from RSS feeds ---")
    raw_finance = fetch_finance_news(max_age_hours=24)
    raw_watchlist = fetch_watchlist_news(max_age_hours=24)
    all_articles = dedup_by_title(raw_finance + raw_watchlist)
    print(f"Total articles: {len(all_articles)}")

    news_context = articles_to_context(all_articles, max_articles=50)

    # Step 3: Generate report
    print("\n--- Generating daily report ---")
    report = generate_report(client, news_context, fg_data)
    if not report:
        print("ERROR: Failed to generate report")
        sys.exit(1)

    if not validate_report(report):
        print("ERROR: Report validation failed")
        sys.exit(1)

    print(f"Title: {report['title']}")
    print(f"Macro events: {len(report.get('macroEvents', []))}")
    print(f"Stock entries: {len(report.get('stocks', []))}")

    # Step 4: Insert new issue at the beginning of issues array
    issues = extract_js_array(html, 'issues')

    today_id = report["id"]
    existing_idx = next((i for i, iss in enumerate(issues) if iss.get("id") == today_id), None)
    if existing_idx is not None:
        print(f"Replacing existing issue for {today_id}")
        issues[existing_idx] = report
    else:
        print(f"Adding new issue for {today_id}")
        issues.insert(0, report)

    html = replace_js_array(html, 'issues', issues)

    # Step 5: Write and validate
    write_html(html)

    html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'index.html'))
    if validate_js(html_path):
        print("\nJS validation: OK")
    else:
        print("\nERROR: JS validation failed!")
        sys.exit(1)

    print(f"\nDone! Report for {report['date']} generated successfully.")


if __name__ == "__main__":
    main()
