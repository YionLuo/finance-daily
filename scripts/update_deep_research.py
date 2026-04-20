#!/usr/bin/env python3
"""
Deep Research Report Generator for Y Daily.
Replaces the old finance-daily and ai-daily scripts.

Three-stage pipeline:
  Stage 1 — Topic Selection: Analyze breaking news, pick the best deep-dive topic
  Stage 2 — Research Gathering: Search for more articles on the chosen topic
  Stage 3 — Report Generation: LLM generates a long-form analysis with citations

Outputs: Updates the `deepResearch` array in index.html
"""

import os
import sys
import json
import re
from datetime import datetime, timezone, timedelta

# Add script dir to path
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    read_html, write_html,
    extract_js_array, extract_js_string,
    replace_js_array,
    create_llm_client, llm_chat_with_retry,
    now_cst, format_date_cst, CST,
    python_to_js_array,
)
from news_fetcher import fetch_topic_articles, articles_to_context


# ============ Constants ============

MAX_RESEARCH_ENTRIES = 30  # Keep last 30 reports
WEEKDAY_MAP = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


# ============ Stage 1: Topic Selection ============

TOPIC_SELECTION_PROMPT = """你是 Y Daily 首席分析师。你需要从今日的 Breaking News 中选出最具深度分析价值的 1 个专题。

选题标准（按优先级排序）：
1. 时效性：最近 24 小时内发生的重大事件
2. 深度价值：有足够的多方观点和数据支撑，适合写 3000-5000 字的深度分析
3. 读者关注度：与金融市场、科技行业、地缘政治等热门领域相关
4. 独特视角：不是简单的事件报道，而是可以挖掘深层逻辑和影响链条的话题

已发布的近期报告主题（务必避免重复）：
{existing_topics}

今日金融快讯：
{finance_news}

今日 AI/科技快讯：
{ai_news}

请选出 1 个最佳专题，输出严格的 JSON 格式（不要包含 markdown 代码块标记）：
{{
  "topic": "专题名称（简洁有力，15字以内）",
  "topicReason": "选题理由（2-3句话）",
  "angle": "分析切入角度（1句话描述你会从什么角度深入分析）",
  "searchKeywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "searchKeywordsCN": ["中文关键词1", "中文关键词2", "中文关键词3"]
}}

searchKeywords 应该是 5 个英文搜索关键词/短语，用于在 Google News 中搜索更多相关文章。
searchKeywordsCN 应该是 3 个中文搜索关键词。
"""


def select_topic(client, breaking_news, ai_breaking_news, existing_topics):
    """
    Stage 1: Analyze breaking news and select the best topic for deep research.
    Returns: dict with topic, topicReason, angle, searchKeywords, searchKeywordsCN
    """
    print("\n=== Stage 1: Topic Selection ===")

    # Format existing topics for dedup
    topics_str = "\n".join(f"- {t}" for t in existing_topics) if existing_topics else "（无历史报告）"

    # Format breaking news
    finance_lines = []
    for item in breaking_news[:20]:
        finance_lines.append(f"[{item.get('time', '')}] [{item.get('tagText', '')}] {item.get('text', '')}")
        if item.get('source'):
            finance_lines.append(f"    来源: {item['source']}")

    ai_lines = []
    for item in ai_breaking_news[:20]:
        ai_lines.append(f"[{item.get('time', '')}] [{item.get('tagText', '')}] {item.get('text', '')}")
        if item.get('source'):
            ai_lines.append(f"    来源: {item['source']}")

    prompt = TOPIC_SELECTION_PROMPT.format(
        existing_topics=topics_str,
        finance_news="\n".join(finance_lines) or "（无金融快讯）",
        ai_news="\n".join(ai_lines) or "（无AI快讯）",
    )

    response = llm_chat_with_retry(
        client,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )

    # Parse JSON response
    # Strip markdown fences if present
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', response, flags=re.MULTILINE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        topic_info = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"WARNING: Failed to parse topic JSON: {e}")
        print(f"Raw response:\n{response[:500]}")
        # Fallback: use the hottest news item
        fallback_text = (breaking_news[0].get("text", "市场动态分析") if breaking_news
                         else ai_breaking_news[0].get("text", "AI行业分析") if ai_breaking_news
                         else "全球市场综述")
        topic_info = {
            "topic": fallback_text[:15],
            "topicReason": "基于当日最热门新闻自动选题",
            "angle": "多角度综合分析",
            "searchKeywords": ["breaking news today", "market analysis", "global economy"],
            "searchKeywordsCN": ["市场分析", "全球经济"],
        }

    print(f"  Selected topic: {topic_info.get('topic', 'unknown')}")
    print(f"  Reason: {topic_info.get('topicReason', '')}")
    print(f"  Keywords: {topic_info.get('searchKeywords', [])}")
    return topic_info


# ============ Stage 2: Research Gathering ============

def gather_research(topic_info, max_articles=50):
    """
    Stage 2: Use topic keywords to search for more articles.
    Returns: list of article dicts
    """
    print("\n=== Stage 2: Research Gathering ===")

    keywords = topic_info.get("searchKeywords", []) + topic_info.get("searchKeywordsCN", [])
    if not keywords:
        keywords = [topic_info.get("topic", "market analysis")]

    articles = fetch_topic_articles(keywords, max_age_hours=72, max_per_keyword=15)

    # Limit total
    if len(articles) > max_articles:
        articles = articles[:max_articles]

    print(f"  Final research pool: {len(articles)} articles")
    return articles


# ============ Stage 3a: Generate Research Prompt ============

RESEARCH_PROMPT_GENERATION = """你是 Y Daily 的研究总监。你的任务不是写报告，而是为今天的选题设计一份**定制化的深度研究指令**。

选题：{topic}
选题理由：{topic_reason}
分析角度：{angle}

你搜集到了 {article_count} 篇相关文章，以下是标题和来源摘要：
{article_summaries}

=== Y Daily 的分析偏好 ===
- 核心关注：AI 和互联网行业的技术/产品/商业动态
- 投资视角：关注对金融市场的影响，尤其是美股和港股
- 读者画像：AI 从业者（工程师、PM、创始人）+ 有科技股投资偏好的专业投资者
- 风格：像 Ben Thompson (Stratechery) 或 Matt Levine (Money Stuff) 那样——有清晰的论点、有逻辑链条、敢下判断

=== 你的任务 ===
根据这个具体选题和你看到的文章素材，设计一份研究指令（prompt），用来指导一个高能力分析模型完成深度分析。

你需要输出以下内容（纯文本，不是 JSON）：

1. **核心研究问题**（1个）：这篇报告要回答的那个最关键的问题是什么？不是"介绍X"，而是"X为什么会Y？这对Z意味着什么？"

2. **必须覆盖的分析维度**（3-5个）：根据这个话题的特殊性，列出必须分析的维度。不要用通用框架（"背景/影响/展望"），要根据这个话题定制。比如：
   - 如果是一个公司的战略动作 → 分析竞争格局变化、定价策略逻辑、对生态系统的影响
   - 如果是政策变化 → 分析真实意图（vs 表面意图）、执行路径、谁受益谁受损
   - 如果是技术突破 → 分析技术可行性、商业化路径、对现有玩家的颠覆程度

3. **必须回答的尖锐问题**（3个）：这些是读者心里会有但一般报告不敢回答的问题。比如"这真的是突破还是营销？""谁在这件事里说谎？""一年后回头看这件事还重要吗？"

4. **投资视角要求**：这个话题对哪些美股/港股有影响？需要分析什么价格/估值逻辑？

5. **写作风格指令**：给分析模型的具体文风要求（结合这个话题的特点）。
"""


def generate_research_prompt(client, topic_info, articles):
    """
    Stage 3a: Generate a customized deep research prompt for this specific topic.
    Uses the fast model (deepseek-chat).
    """
    print("\n=== Stage 3a: Generating Research Prompt ===")

    # Build article summaries (just titles + sources, not full text)
    summaries = []
    for i, art in enumerate(articles[:40], 1):
        title = art.get("title", "")[:100]
        source = art.get("source", "")
        summaries.append(f"[{i}] {title} ({source})")

    prompt = RESEARCH_PROMPT_GENERATION.format(
        topic=topic_info.get("topic", ""),
        topic_reason=topic_info.get("topicReason", ""),
        angle=topic_info.get("angle", ""),
        article_count=len(articles),
        article_summaries="\n".join(summaries),
    )

    research_prompt = llm_chat_with_retry(
        client,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.5,
    )

    print(f"  Research prompt generated ({len(research_prompt)} chars)")
    return research_prompt


# ============ Stage 3b: Execute Deep Analysis ============

# Model for deep analysis (reasoning model with chain-of-thought)
ANALYSIS_MODEL = os.environ.get("ANALYSIS_MODEL", "deepseek/deepseek-r1")

REPORT_EXECUTION_PROMPT = """你是一位资深的行业分析师，现在要根据以下研究指令和素材，撰写一篇深度分析报告。

=== 研究指令（由研究总监制定）===
{research_prompt}

=== 新闻素材（共 {article_count} 篇，带编号和来源）===
{articles_context}

=== 输出要求 ===

1. **严格基于提供的真实文章**，关键论点必须标注来源编号如 [1][3]
2. 语言：中文，专业但不晦涩
3. 报告要有**一个清晰的核心论点**贯穿全文，不是面面俱到的综述
4. 章节结构根据上面的研究指令来定（不要用固定模板），每个章节解决一个具体问题
5. 每个章节的 content 字段包含 HTML 格式（<p>段落、<strong>加粗、<ul><li>列表等）
6. 敢于给出判断——"我们认为X因为Y"——而不是"有待观察"
7. 总字数 3000-6000 字

输出严格的 JSON 格式（不要包含 markdown 代码块标记）：
{{
  "title": "报告标题（有态度、有判断，不是中性描述，20字以内）",
  "subtitle": "副标题（核心论点的一句话浓缩，30字以内）",
  "summary": "200字以内的报告摘要——不是内容概要，而是'我们的核心判断是什么、为什么、这意味着什么'",
  "tags": [
    {{"text": "标签", "type": "up|down|warn"}}
  ],
  "keyTakeaways": [
    "核心判断1（一句有态度的话，不是中性描述）",
    "核心判断2",
    "核心判断3",
    "核心判断4"
  ],
  "relatedTickers": ["相关美股/港股代码"],
  "sections": [
    {{
      "id": "sec-1",
      "title": "章节标题（根据研究指令定制，不是固定模板）",
      "content": "<p>HTML 格式内容。引用来源用 [编号]。</p>"
    }}
    // 4-7 个章节，结构由研究指令决定
  ],
  "sourceIndices": [1, 3, 5]
}}

tags type: "up"=利好/积极, "down"=利空/消极, "warn"=警示/关注。
sourceIndices: 实际引用过的文章编号列表。
"""


def generate_report(client, topic_info, articles, breaking_context=""):
    """
    Stage 3: Two-step report generation.
    3a: Generate a customized research prompt (fast model)
    3b: Execute deep analysis with the generated prompt (reasoning model)
    """

    # ====== Stage 3a: Generate research prompt ======
    research_prompt = generate_research_prompt(client, topic_info, articles)

    # ====== Stage 3b: Execute deep analysis ======
    print(f"\n=== Stage 3b: Executing Deep Analysis (model: {ANALYSIS_MODEL}) ===")

    articles_context = articles_to_context(articles, max_articles=50)

    execution_prompt = REPORT_EXECUTION_PROMPT.format(
        research_prompt=research_prompt,
        article_count=len(articles),
        articles_context=articles_context,
    )

    response = llm_chat_with_retry(
        client,
        messages=[{"role": "user", "content": execution_prompt}],
        model=ANALYSIS_MODEL,
        max_tokens=16384,
        temperature=0.6,
    )

    # Parse JSON response (R1 may include <think> tags, strip them)
    cleaned = response
    # Strip <think>...</think> block if present (DeepSeek R1 reasoning trace)
    think_end = cleaned.find("</think>")
    if think_end != -1:
        cleaned = cleaned[think_end + len("</think>"):]
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        report = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse report JSON: {e}")
        print(f"Raw response (first 1500 chars):\n{response[:1500]}")
        raise

    print(f"  Report title: {report.get('title', 'untitled')}")
    print(f"  Sections: {len(report.get('sections', []))}")
    return report


# ============ Source extraction ============

def extract_sources(articles, source_indices):
    """Extract cited sources from the article list based on indices."""
    sources = []
    for idx in source_indices:
        if 1 <= idx <= len(articles):
            art = articles[idx - 1]
            sources.append({
                "title": art.get("title", "")[:100],
                "url": art.get("url", ""),
                "publisher": art.get("source", ""),
            })
    return sources


# ============ Main ============

def main():
    print("=" * 60)
    print("Y Daily — Deep Research Report Generator")
    print("=" * 60)

    now = now_cst()
    date_id = now.strftime("%Y-%m-%d")
    date_display = f"{now.year}年{now.month}月{now.day}日"
    weekday = WEEKDAY_MAP[now.weekday()]

    # Read current HTML
    html = read_html()

    # Extract existing data
    breaking_news = extract_js_array(html, "breakingNews")
    ai_breaking_news = extract_js_array(html, "aiBreakingNews")
    deep_research = extract_js_array(html, "deepResearch")

    print(f"\nCurrent state:")
    print(f"  Breaking news: {len(breaking_news)} items")
    print(f"  AI breaking news: {len(ai_breaking_news)} items")
    print(f"  Existing deep research reports: {len(deep_research)}")

    # Check if already generated today
    if deep_research and deep_research[0].get("id") == date_id:
        print(f"\nWARNING: Report for {date_id} already exists. Overwriting.")

    # Get recent topics (last 7 days) for dedup
    existing_topics = [r.get("topic", "") for r in deep_research[:7] if r.get("topic")]

    # Need at least some news to work with
    if not breaking_news and not ai_breaking_news:
        print("\nERROR: No breaking news available. Cannot select topic.")
        sys.exit(1)

    # Create LLM client
    client = create_llm_client(required=True)

    # ====== Stage 1: Topic Selection ======
    topic_info = select_topic(client, breaking_news, ai_breaking_news, existing_topics)

    # ====== Stage 2: Research Gathering ======
    articles = gather_research(topic_info)

    if len(articles) < 3:
        print("WARNING: Very few articles found. Report may be thin.")

    # ====== Stage 3: Report Generation ======
    report = generate_report(client, topic_info, articles)

    # Build source list
    source_indices = report.get("sourceIndices", [])
    sources = extract_sources(articles, source_indices)

    # Estimate read time (rough: 500 chars/min for Chinese)
    total_content = ""
    for sec in report.get("sections", []):
        total_content += sec.get("content", "")
    # Strip HTML tags for char count
    text_only = re.sub(r'<[^>]+>', '', total_content)
    read_minutes = max(5, len(text_only) // 500)

    # Build the deep research entry
    entry = {
        "id": date_id,
        "date": date_display,
        "weekday": weekday,
        "title": report.get("title", topic_info.get("topic", "深度研究报告")),
        "subtitle": report.get("subtitle", ""),
        "summary": report.get("summary", ""),
        "tags": report.get("tags", []),
        "topic": topic_info.get("topic", ""),
        "topicReason": topic_info.get("topicReason", ""),
        "readTime": f"{read_minutes}分钟",
        "generatedAt": format_date_cst(now),
        "keyTakeaways": report.get("keyTakeaways", []),
        "relatedTickers": report.get("relatedTickers", []),
        "sections": report.get("sections", []),
        "sources": sources,
    }

    # Insert or replace today's entry
    if deep_research and deep_research[0].get("id") == date_id:
        deep_research[0] = entry
    else:
        deep_research.insert(0, entry)

    # Trim to max entries
    if len(deep_research) > MAX_RESEARCH_ENTRIES:
        deep_research = deep_research[:MAX_RESEARCH_ENTRIES]

    # Write back to HTML
    html = replace_js_array(html, "deepResearch", deep_research)
    write_html(html)

    print(f"\n{'=' * 60}")
    print(f"SUCCESS: Deep research report generated!")
    print(f"  Topic: {entry['title']}")
    print(f"  Sections: {len(entry['sections'])}")
    print(f"  Sources: {len(entry['sources'])}")
    print(f"  Read time: {entry['readTime']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
