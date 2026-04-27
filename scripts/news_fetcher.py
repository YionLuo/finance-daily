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


# ============ Web Search (for agent tool use) ============

def web_search(query, max_results=10, max_age_hours=72):
    """
    Search for articles via Google News RSS. Used as a tool by the research agent.

    Args:
        query: search query string
        max_results: max number of results to return
        max_age_hours: how far back to search

    Returns:
        list of dicts with title, summary, url, source, published
    """
    feed_url = _make_google_news_url(query)
    articles = fetch_rss_articles([feed_url], max_age_hours=max_age_hours, max_per_feed=max_results)
    return articles[:max_results]


def _resolve_google_news_url(url):
    """
    Resolve a Google News RSS redirect URL to the actual article URL.
    Google News URLs look like: https://news.google.com/rss/articles/CBMi...
    They return a 302/303 redirect or an HTML page with a redirect.
    """
    if "news.google.com/rss/articles/" not in url:
        return url

    try:
        # Method 1: Use HEAD request to follow redirects
        resp = requests.head(url, headers={"User-Agent": USER_AGENT},
                             timeout=10, allow_redirects=True)
        if resp.url and "news.google.com" not in resp.url:
            return resp.url
    except Exception:
        pass

    try:
        # Method 2: GET and look for redirect in HTML/meta tags
        resp = requests.get(url, headers={"User-Agent": USER_AGENT},
                            timeout=10, allow_redirects=True)
        # Check for meta refresh or JS redirect
        import re as _re
        meta_match = _re.search(r'<meta[^>]*?url=(["\']?)([^"\'\s>]+)', resp.text, _re.IGNORECASE)
        if meta_match:
            return meta_match.group(2)
        # Check for window.location
        loc_match = _re.search(r'window\.location\s*=\s*["\']([^"\']+)', resp.text)
        if loc_match:
            return loc_match.group(1)
        # Check for data-url or href in the page
        href_match = _re.search(r'href="(https?://(?!news\.google)[^"]+)"', resp.text)
        if href_match:
            return href_match.group(1)
    except Exception:
        pass

    return url  # Fallback: return original


def fetch_financial_data(ticker: str) -> str:
    """
    Fetch real-time financial data via Yahoo Finance. No API key required.
    Tries multiple methods with fallback to handle rate limits.
    Returns key metrics: price, market cap, P/E, revenue, margins, cash flow, etc.
    Also adds business model context notes to help avoid analytical errors.
    """
    ticker = ticker.strip().upper()
    ts = __import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    # Business model context for commonly misanalyzed companies
    BUSINESS_MODEL_NOTES = {
        "NVDA": (
            "⚠️ 分析提醒：NVIDIA 是 fabless 芯片设计公司（台积电代工），自身 CAPEX 相对营收极低。"
            "如果 FCF < 净利润，更可能是营运资金变动（应收账款增长、库存变化）导致，而非'建设数据中心'。"
            "数据中心的 CAPEX 是客户（AWS/Azure/GCP）的支出，不是 NVIDIA 的。"
        ),
        "AMD": (
            "⚠️ 分析提醒：AMD 是 fabless 芯片设计公司（台积电代工），CAPEX 极低。"
            "FCF 与净利润的差异主要来自营运资金变动和一次性项目。"
        ),
        "QCOM": (
            "⚠️ 分析提醒：Qualcomm 是 fabless 芯片设计公司，无自有晶圆厂。"
        ),
        "INTC": (
            "⚠️ 分析提醒：Intel 是 IDM（有自有晶圆厂），CAPEX 极高是正常的。"
            "与 NVIDIA/AMD 的 fabless 模式不可直接类比。"
        ),
        "TSM": (
            "⚠️ 分析提醒：台积电是纯代工厂（foundry），高 CAPEX 是核心商业模式。"
            "CAPEX/营收比高达 40-50% 是行业常态，不应视为负面信号。"
        ),
    }

    def _v(d, key):
        x = d.get(key, {})
        if not isinstance(x, dict):
            return str(x) if x else "N/A"
        return x.get("fmt") or (str(x.get("raw")) if x.get("raw") is not None else "N/A")

    # Browser-like headers to avoid rate limiting
    yahoo_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
    }

    # Method 1: Yahoo Finance v10 quoteSummary API
    try:
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        params = {"modules": "financialData,defaultKeyStatistics,summaryDetail,price"}
        resp = requests.get(url, params=params, headers=yahoo_headers, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            result_list = data.get("quoteSummary", {}).get("result", None)
            if result_list:
                result = result_list[0]
                fd = result.get("financialData", {})
                ks = result.get("defaultKeyStatistics", {})
                sd = result.get("summaryDetail", {})
                pr = result.get("price", {})
                lines = [
                    f"=== {ticker} 财务数据 (Yahoo Finance, 实时) ===",
                    f"当前价格: {_v(pr, 'regularMarketPrice')} {_v(pr, 'currency')}",
                    f"市值: {_v(pr, 'marketCap')}",
                    f"P/E (TTM): {_v(sd, 'trailingPE')}",
                    f"Forward P/E: {_v(sd, 'forwardPE')}",
                    f"EPS (TTM): {_v(ks, 'trailingEps')}",
                    f"营收 (TTM): {_v(fd, 'totalRevenue')}",
                    f"毛利率: {_v(fd, 'grossMargins')}",
                    f"营业利润率: {_v(fd, 'operatingMargins')}",
                    f"净利率: {_v(fd, 'profitMargins')}",
                    f"自由现金流: {_v(fd, 'freeCashflow')}",
                    f"现金及等价物: {_v(fd, 'totalCash')}",
                    f"总债务: {_v(fd, 'totalDebt')}",
                    f"52周区间: {_v(sd, 'fiftyTwoWeekLow')} - {_v(sd, 'fiftyTwoWeekHigh')}",
                    f"分析师目标价(均值): {_v(fd, 'targetMeanPrice')}",
                    f"分析师评级: {_v(fd, 'recommendationKey')}",
                    f"数据源: Yahoo Finance v10 (TTM数据，口径可能与10-K/10-Q有差异) | 获取时间: {ts}",
                    f"⚠️ 注意：以上为 TTM（最近12个月滚动）数据，非最新季报原始数据。建议搜索公司最新10-K/10-Q核实关键数字。",
                ]
                # Add business model context if available
                bm_note = BUSINESS_MODEL_NOTES.get(ticker, "")
                if bm_note:
                    lines.append(bm_note)
                return "\n".join(lines)
    except Exception:
        pass

    # Method 2: yfinance library (if installed)
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if info and price:
            currency = info.get("currency", "")
            market_cap = info.get("marketCap", "N/A")
            if isinstance(market_cap, (int, float)) and market_cap > 1e9:
                market_cap = f"{market_cap/1e9:.1f}B"
            lines = [
                f"=== {ticker} 财务数据 (yfinance) ===",
                f"当前价格: {price} {currency}",
                f"市值: {market_cap}",
                f"P/E (TTM): {info.get('trailingPE', 'N/A')}",
                f"Forward P/E: {info.get('forwardPE', 'N/A')}",
                f"EPS (TTM): {info.get('trailingEps', 'N/A')}",
                f"营收 (TTM): {info.get('totalRevenue', 'N/A')}",
                f"毛利率: {info.get('grossMargins', 'N/A')}",
                f"营业利润率: {info.get('operatingMargins', 'N/A')}",
                f"净利率: {info.get('profitMargins', 'N/A')}",
                f"自由现金流: {info.get('freeCashflow', 'N/A')}",
                f"现金及等价物: {info.get('totalCash', 'N/A')}",
                f"总债务: {info.get('totalDebt', 'N/A')}",
                f"52周区间: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
                f"分析师目标价(均值): {info.get('targetMeanPrice', 'N/A')}",
                f"分析师评级: {info.get('recommendationKey', 'N/A')}",
                f"数据源: yfinance (TTM数据，口径可能与10-K/10-Q有差异) | 获取时间: {ts}",
                f"⚠️ 注意：以上为 TTM（最近12个月滚动）数据，非最新季报原始数据。建议搜索公司最新10-K/10-Q核实关键数字。",
            ]
            # Add business model context if available
            bm_note = BUSINESS_MODEL_NOTES.get(ticker, "")
            if bm_note:
                lines.append(bm_note)
            return "\n".join(lines)
    except Exception:
        pass

    # Method 3: Yahoo Finance v8 chart API (simpler, less rate-limited)
    try:
        chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        resp2 = requests.get(chart_url, params={"interval": "1d", "range": "5d"},
                             headers=yahoo_headers, timeout=TIMEOUT)
        if resp2.status_code == 200:
            cdata = resp2.json()
            meta = cdata.get("chart", {}).get("result", [{}])[0].get("meta", {})
            price = meta.get("regularMarketPrice", "N/A")
            currency = meta.get("currency", "")
            return (
                f"=== {ticker} 基础行情 (Yahoo Finance chart API) ===\n"
                f"当前价格: {price} {currency}\n"
                f"52周区间: {meta.get('fiftyTwoWeekLow', 'N/A')} - {meta.get('fiftyTwoWeekHigh', 'N/A')}\n"
                f"注意：完整财务数据获取受限，仅提供基础行情。建议用 web_search 补充财务细节。\n"
                f"数据源: Yahoo Finance v8 | 获取时间: {ts}"
            )
    except Exception:
        pass

    return (
        f"Error: 无法获取 {ticker} 的财务数据（Yahoo Finance 限速或 ticker 不正确）。\n"
        f"建议通过 web_search 搜索 '{ticker} revenue earnings P/E 2026' 获取相关信息。"
    )


def fetch_url_content(url, max_chars=10000):
    """
    Fetch and extract text content from a URL. Used by the agent to read full articles.
    Handles Google News redirect URLs automatically.

    Args:
        url: URL to fetch (can be a Google News redirect URL)
        max_chars: max characters to return (default 10000 for richer content)

    Returns:
        str: extracted text content, or error message
    """
    try:
        # Resolve Google News redirects first
        resolved_url = _resolve_google_news_url(url)
        if resolved_url != url:
            print(f"    Resolved: {url[:60]}... → {resolved_url[:80]}")

        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(resolved_url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        html_text = resp.text

        # Simple content extraction: strip scripts/styles, then tags
        # Remove script and style blocks
        cleaned = re.sub(r'<script[^>]*>.*?</script>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<style[^>]*>.*?</style>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        # Remove nav, footer, aside, header blocks
        cleaned = re.sub(r'<(nav|footer|aside|header)[^>]*>.*?</\1>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        # Strip tags
        text = re.sub(r'<[^>]+>', ' ', cleaned)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) < 100:
            return f"Error: page content too short ({len(text)} chars), likely a redirect or paywall page. URL: {resolved_url}"

        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return text
    except Exception as e:
        err_str = str(e)
        if "403" in err_str or "401" in err_str:
            return f"Error: paywall/auth required ({resolved_url[:60]})"
        return f"Error fetching URL: {e}"


# Tool definitions for OpenAI function calling format
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search Google News for articles on a specific topic. Returns titles, summaries, URLs and sources. Use this to find information you need for your analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. Be specific. English works best for global topics."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url_content",
            "description": "Fetch and read the full text content of a specific URL. Use this to get details from an article you found via web_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_financial_data",
            "description": "Fetch real-time financial data for a stock ticker via Yahoo Finance (free, no API key). Returns: current price, market cap, P/E ratio, forward P/E, EPS, revenue (TTM), gross/operating/net margins, free cash flow, cash, total debt, 52-week range, analyst target price and rating. Use this whenever the topic involves a specific company to ground your analysis in real numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. NVDA, AAPL, 0700.HK, 9988.HK, TSLA, MSFT"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]


def execute_tool_call(tool_name, arguments):
    """Execute a tool call from the agent and return the result as string."""
    if tool_name == "web_search":
        query = arguments.get("query", "")
        results = web_search(query)
        if not results:
            return "No results found for this query."
        lines = []
        for i, art in enumerate(results, 1):
            lines.append(f"[{i}] {art['title']}")
            if art.get('summary'):
                lines.append(f"    {art['summary'][:200]}")
            lines.append(f"    Source: {art.get('source', '')} | URL: {art.get('url', '')}")
            lines.append("")
        return "\n".join(lines)

    elif tool_name == "fetch_url_content":
        url = arguments.get("url", "")
        return fetch_url_content(url)

    elif tool_name == "fetch_financial_data":
        ticker = arguments.get("ticker", "")
        return fetch_financial_data(ticker)

    else:
        return f"Unknown tool: {tool_name}"


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
