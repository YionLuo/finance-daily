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

LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")


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

    prompt = f"""You are the chief analyst at Y Daily (yion.me), a financial intelligence platform.
Current time: {format_date_cst(now)}

Below are REAL news articles from the past 24 hours, fetched from RSS feeds (Reuters, CNBC, Bloomberg, MarketWatch, Yahoo Finance, Google News, etc.).

=== TODAY'S NEWS ===
{news_context}
=== END NEWS ===
{fg_str}

Generate today's finance daily report based on THESE REAL articles above.

WATCHLIST: {json.dumps(watchlist_flat, ensure_ascii=False)}
FOCUS AREAS: {FOCUS_AREAS}

You MUST produce a complete issue object matching this EXACT structure:

{{
  "id": "{now.strftime('%Y-%m-%d')}",
  "date": "{date_cn}",
  "weekday": "{weekday}",
  "title": "50 chars max, key events summary",
  "summary": "100 chars max, core narrative",
  "tags": [
    {{"text": "恒生科技 +X.XX%", "type": "up|down|warn"}},
    // 4-5 tags total
  ],
  "watchlist": {json.dumps(watchlist_flat[:10], ensure_ascii=False)},
  "focusAreas": "{FOCUS_AREAS}",
  "timeRange": "{time_range}",
  "generatedAt": "{format_date_cst(now)}",
  "briefings": [
    // 3 HTML strings, each starting with <span class=\\"briefing-number\\">N</span><strong>...
    // Use <span class=\\"data up\\">+X%</span> for up data, <span class=\\"data down\\">-X%</span> for down
  ],
  "macroEvents": [
    {{
      "icon": "emoji",
      "iconBg": "#hex",
      "title": "event title",
      "fact": "core facts with HTML spans for data",
      "source": "source name",
      "url": "source URL from the articles above",
      "mappings": [
        {{"type": "bull", "text": "<strong>Target</strong> — reasoning"}},
        {{"type": "bear", "text": "<strong>Target</strong> — reasoning"}}
      ]
    }}
    // 3-5 macro events
  ],
  "stocks": [
    {{
      "name": "emoji Name",
      "code": "TICKER",
      "news": [
        {{"label": "📰 headline", "content": "detail with <span class=\\"data\\">numbers</span>", "source": "source", "url": "URL"}}
      ],
      "assessment": "HTML string with impact analysis"
    }}
    // Cover watchlist stocks with actual news
  ],
  "alerts": [
    {{"time": "timeframe", "event": "upcoming event", "target": "TICKER"}}
    // 6-8 forward-looking alerts
  ]
}}

CRITICAL RULES:
1. Base your report ONLY on the real articles provided above. Do NOT fabricate any data or events.
2. ALL numbers (prices, percentages) must come from the articles. If not in articles, do not guess.
3. Include source URLs from the articles in macroEvents and stock news items.
4. Use Chinese for all text content.
5. Include HTML formatting (<strong>, <span class="data">, etc.) as shown.
6. The data represents Chinese market convention: 涨=红(up), 跌=绿(down).
7. If there's not enough data for a section, use fewer items rather than fabricating.

Return ONLY the JSON object. No markdown fencing.
"""

    try:
        raw = llm_chat_with_retry(
            client, [{"role": "user", "content": prompt}],
            model=LLM_MODEL, max_tokens=4096, temperature=0.2, max_retries=3,
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
