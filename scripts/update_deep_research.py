#!/usr/bin/env python3
"""
Deep Research Report Generator for Y Daily.

Four-stage pipeline that mimics how a senior analyst works:

  Stage 1 — Topic Selection: Pick the best deep-dive topic from breaking news
  Stage 2 — Brain Dump: Activate the model's internal knowledge about this topic
  Stage 3 — Research Agent: Model researches with web_search tool, thinking + searching iteratively
  Stage 4 — Format: Convert the analysis into structured JSON for the frontend

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
from news_fetcher import (
    fetch_topic_articles, articles_to_context,
    AGENT_TOOLS, execute_tool_call,
)

# ============ Constants ============

MAX_RESEARCH_ENTRIES = 30
WEEKDAY_MAP = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek/deepseek-chat")
MAX_AGENT_ROUNDS = 12  # Safety limit for agent loop


# ============ Stage 1: Topic Selection (unchanged) ============

TOPIC_SELECTION_PROMPT = """你是 Y Daily 首席分析师。从今日 Breaking News 中选出最具深度分析价值的 1 个专题。

选题标准：
1. 时效性：最近 24 小时内的重大事件
2. 深度价值：有足够多的角度和数据支撑深度分析
3. 与 AI/互联网/科技行业 或 金融市场（美股/港股）相关
4. 不是简单的事件报道，而是可以挖掘深层逻辑的话题

已发布的近期报告主题（避免重复）：
{existing_topics}

今日金融快讯：
{finance_news}

今日 AI/科技快讯：
{ai_news}

输出严格 JSON（不要 markdown 代码块）：
{{
  "topic": "专题名称（15字以内）",
  "topicReason": "选题理由（2-3句话）",
  "angle": "分析切入角度（1句话）"
}}
"""


def select_topic(client, breaking_news, ai_breaking_news, existing_topics):
    """Stage 1: Select the best topic from breaking news."""
    print("\n=== Stage 1: Topic Selection ===")

    topics_str = "\n".join(f"- {t}" for t in existing_topics) if existing_topics else "（无历史报告）"

    finance_lines = []
    for item in breaking_news[:20]:
        finance_lines.append(f"[{item.get('time', '')}] [{item.get('tagText', '')}] {item.get('text', '')}")

    ai_lines = []
    for item in ai_breaking_news[:20]:
        ai_lines.append(f"[{item.get('time', '')}] [{item.get('tagText', '')}] {item.get('text', '')}")

    prompt = TOPIC_SELECTION_PROMPT.format(
        existing_topics=topics_str,
        finance_news="\n".join(finance_lines) or "（无金融快讯）",
        ai_news="\n".join(ai_lines) or "（无AI快讯）",
    )

    response = llm_chat_with_retry(client, [{"role": "user", "content": prompt}], max_tokens=512, temperature=0.3)
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', response, flags=re.MULTILINE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE).strip()

    try:
        topic_info = json.loads(cleaned)
    except json.JSONDecodeError:
        fallback = (breaking_news[0].get("text", "市场动态")[:15] if breaking_news
                    else ai_breaking_news[0].get("text", "AI行业")[:15] if ai_breaking_news
                    else "全球市场综述")
        topic_info = {"topic": fallback, "topicReason": "自动选题", "angle": "综合分析"}

    print(f"  Topic: {topic_info.get('topic', '?')}")
    print(f"  Reason: {topic_info.get('topicReason', '')}")
    return topic_info


# ============ Stage 2: Brain Dump ============

BRAIN_DUMP_PROMPT = """你是一位在 AI 和科技金融领域有 15 年经验的资深分析师。
当前时间：{current_time}

今天你要深度分析的主题是：「{topic}」
角度：{angle}

在你开始搜索任何新信息之前，先把你关于这个话题**已经知道的东西**倒出来。这是你的专业积累，包括但不限于：

1. **行业背景**：这个领域的基本结构、关键玩家、商业模式
2. **历史脉络**：相关的重要事件、转折点、周期性规律
3. **分析框架**：分析这类问题通常要看哪些维度（技术、商业、政策、竞争格局…）
4. **常见误区**：大众或媒体在这个话题上通常有哪些错误认知
5. **关键数据点**：你记得的重要数据（市场规模、增速、关键公司的财务指标等）
6. **你的初步判断**：基于经验，你对这件事的第一反应是什么

注意：你的知识可能截止到 2024-2025 年，这没关系——先把背景知识倒出来，后续会通过搜索补充 2026 年的最新信息。标注你不确定的地方（如"截至 2024 年数据为 X，2026 年需确认"）。

不要搜索，不要引用新闻。纯粹基于你的知识库输出。
用中文，2000 字左右。
"""


def brain_dump(client, topic_info):
    """Stage 2: Activate the model's internal knowledge about the topic."""
    print("\n=== Stage 2: Brain Dump (Knowledge Activation) ===")

    prompt = BRAIN_DUMP_PROMPT.format(
        topic=topic_info.get("topic", ""),
        angle=topic_info.get("angle", ""),
        current_time=format_date_cst(now_cst()),
    )

    response = llm_chat_with_retry(
        client, [{"role": "user", "content": prompt}],
        max_tokens=4096, temperature=0.5,
    )

    print(f"  Brain dump: {len(response)} chars")
    return response


# ============ Stage 3: Research Agent Loop ============

RESEARCH_SYSTEM_PROMPT = """你是 Y Daily 的首席分析师，正在对「{topic}」进行深度研究。

当前时间：{current_time}（注意：现在是 2026 年，你的内部知识可能截止到 2024-2025，所有关于当前状态的判断必须基于搜索到的最新信息，不要用旧数据当新事实。）

你的读者是 AI/科技行业资深从业者和专业投资者。他们不需要科普，需要的是独到洞察和有论据支撑的判断。

=== 你的知识储备（背景参考，数据可能过时）===
{brain_dump}

=== 今日相关新闻（2026 年最新）===
{breaking_context}

=== 你的工作流程 ===

第一步：研究
- 用 web_search 搜索你需要的最新信息（2026 年的数据、事件、观点）
- 用 fetch_url_content 读取关键文章的全文以获取细节
- 特别注意搜索：最新的数据/财报、不同立场的观点、被忽略的反面证据

第二步：列提纲
- 搜集够了后，先输出你的报告提纲（5-7 个章节标题 + 每章核心论点）
- 确认提纲覆盖了：核心论点、支撑证据、反面观点、投资影响

第三步：写报告
- 确认提纲后直接开始写完整报告
- **报告正文必须 4000-6000 字**，每个章节至少 500 字——这不是新闻简报，是深度研究
- 每个关键论点都要有具体数据和来源支撑

=== Y Daily 的分析偏好 ===
- 核心关注：AI 和互联网行业的技术/产品/商业动态
- 投资视角：关注对美股和港股的影响，给出具体标的和逻辑
- 风格：有一个清晰的核心论点贯穿全文，不是面面俱到的综述
- 敢于下判断——"我们认为 X 因为 Y"——而不是"有待观察"
- 指出主流叙事中的盲点或错误
- 所有数据和事实必须标注来源

=== 报告格式 ===
直接输出中文文章，不要 JSON。

必须包含：
- 标题（有态度、有判断，不是中性描述）
- 副标题（核心论点一句话浓缩）
- 正文 4000-6000 字，分 5-7 个章节，每章有小标题
- 关键数据和论点后标注来源（文章标题或 URL）
- 末尾附「核心判断」（4-5 条，每条一句话，有态度）
- 末尾附「关注标的」（相关美股/港股代码 + 一句话逻辑）

开始研究。
"""


def research_agent_loop(client, topic_info, brain_dump_text, breaking_context):
    """
    Stage 3: Agent loop with web_search and fetch_url_content tools.
    Model decides what to search, reads results, and writes the final analysis.
    """
    print(f"\n=== Stage 3: Research Agent Loop (max {MAX_AGENT_ROUNDS} rounds) ===")

    system_prompt = RESEARCH_SYSTEM_PROMPT.format(
        topic=topic_info.get("topic", ""),
        brain_dump=brain_dump_text,
        breaking_context=breaking_context,
        current_time=format_date_cst(now_cst()),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""开始研究「{topic_info.get('topic', '')}」。

请按以下步骤进行：
1. 先用 web_search 搜索 5-8 个不同角度的关键词，收集足够的素材
2. 对重要文章用 fetch_url_content 读取全文获取细节数据
3. 搜集足够素材后（至少搜索 5 次以上），先列出报告提纲
4. 然后写出完整报告（必须 4000 字以上，每个章节至少 500 字）

重要：不要急于写报告。先充分搜索，读几篇关键文章全文，确保你有足够的最新数据和多方观点。"""},
    ]

    collected_sources = []
    final_text = None
    search_count = 0  # Track number of searches done

    for round_num in range(MAX_AGENT_ROUNDS):
        print(f"\n  --- Round {round_num + 1} ---")

        try:
            from openai import OpenAI
            # We need raw API call for tool use since llm_chat_with_retry doesn't support it
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
            raw_client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)

            response = raw_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=AGENT_TOOLS,
                temperature=0.5,
                max_tokens=16384,
            )
        except Exception as e:
            print(f"  LLM call failed: {e}")
            break

        choice = response.choices[0]
        message = choice.message

        # Check if model wants to call tools
        if message.tool_calls:
            # Add assistant message with tool calls
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                print(f"  Tool: {func_name}({json.dumps(func_args, ensure_ascii=False)[:100]})")

                result = execute_tool_call(func_name, func_args)

                # Track searches
                if func_name == "web_search":
                    search_count += 1
                    search_results = result.split("\n")
                    for line in search_results:
                        if line.strip().startswith("[") and "]" in line:
                            collected_sources.append(line.strip())

                # Truncate very long results
                if len(result) > 5000:
                    result = result[:5000] + "\n...(truncated)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            print(f"  Processed {len(message.tool_calls)} tool call(s), total searches: {search_count}")

        else:
            # Model returned text
            text = message.content or ""

            # If model tried to write too early (not enough research), push it back
            if search_count < 4 and round_num < 5:
                print(f"  Output received but only {search_count} searches done. Requesting more research.")
                messages.append(message)
                messages.append({"role": "user", "content": f"你只搜索了 {search_count} 次，素材还不够。请继续用 web_search 搜索更多角度的信息，特别是：反面观点、具体数据（财报/融资/市场规模）、2026年最新动态。至少再搜索 3 次再开始写。"})
                continue

            # If output is too short, ask to expand
            if len(text) < 3000 and round_num < MAX_AGENT_ROUNDS - 2:
                print(f"  Output too short ({len(text)} chars). Requesting expansion.")
                messages.append(message)
                messages.append({"role": "user", "content": f"报告只有约 {len(text)} 字，远低于 4000 字的最低要求。请扩展每个章节的分析深度——补充更多数据、对比、案例和推理过程。不要重写，在现有基础上扩展。目标至少 4000 字。"})
                continue

            final_text = text
            print(f"  Final output: {len(final_text)} chars")
            break

    if not final_text:
        print("  WARNING: Agent loop ended without producing final text")
        # Try one more call without tools to force output
        messages.append({"role": "user", "content": "请现在输出你的完整分析报告。"})
        try:
            response = raw_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.5,
                max_tokens=16384,
            )
            final_text = response.choices[0].message.content
            print(f"  Forced output: {len(final_text or '')} chars")
        except Exception as e:
            print(f"  Force output failed: {e}")
            final_text = "研究报告生成失败"

    return final_text, collected_sources


# ============ Stage 4: Format to JSON ============

FORMAT_PROMPT = """把以下深度分析报告转换为 JSON 格式。保留所有内容，只改变格式。

=== 原始报告 ===
{report_text}

=== 输出 JSON ===
严格输出以下 JSON 结构（不要 markdown 代码块）：
{{
  "title": "报告标题（从报告中提取）",
  "subtitle": "副标题（从报告中提取）",
  "summary": "200字以内摘要（从报告中提取或概括核心论点）",
  "tags": [
    {{"text": "标签", "type": "up|down|warn"}}
  ],
  "keyTakeaways": ["核心判断1", "核心判断2", "核心判断3"],
  "relatedTickers": ["股票代码1", "股票代码2"],
  "sections": [
    {{
      "id": "sec-1",
      "title": "章节标题",
      "content": "<p>将章节正文转为 HTML（<p>段落、<strong>加粗、<ul><li>列表）</p>"
    }}
  ],
  "sourceIndices": []
}}

tags type: "up"=利好, "down"=利空, "warn"=警示。
sections 的 content 用 HTML 格式，保留原文的所有分析内容和来源标注。
"""


def format_to_json(client, report_text):
    """Stage 4: Convert plain-text analysis into structured JSON."""
    print("\n=== Stage 4: Format to JSON ===")

    prompt = FORMAT_PROMPT.format(report_text=report_text[:12000])

    response = llm_chat_with_retry(
        client, [{"role": "user", "content": prompt}],
        max_tokens=16384, temperature=0.1,
    )

    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', response, flags=re.MULTILINE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE).strip()
    # Strip <think> block if present
    think_end = cleaned.find("</think>")
    if think_end != -1:
        cleaned = cleaned[think_end + len("</think>"):].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  ERROR: JSON parse failed: {e}")
        print(f"  Raw (first 500): {cleaned[:500]}")
        # Fallback: create minimal structure
        return {
            "title": "深度分析报告",
            "subtitle": "",
            "summary": report_text[:200],
            "tags": [],
            "keyTakeaways": [],
            "relatedTickers": [],
            "sections": [{"id": "sec-1", "title": "分析报告", "content": f"<p>{report_text[:3000]}</p>"}],
            "sourceIndices": [],
        }


# ============ Main ============

def main():
    print("=" * 60)
    print("Y Daily — Deep Research Report Generator (Agent Pipeline)")
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
    print(f"  Existing reports: {len(deep_research)}")

    if deep_research and deep_research[0].get("id") == date_id:
        print(f"\nWARNING: Report for {date_id} already exists. Overwriting.")

    if not breaking_news and not ai_breaking_news:
        print("\nERROR: No breaking news available.")
        sys.exit(1)

    # Create LLM client
    client = create_llm_client(required=True)

    # Get recent topics for dedup
    existing_topics = [r.get("topic", "") for r in deep_research[:7] if r.get("topic")]

    # ====== Stage 1: Topic Selection ======
    topic_info = select_topic(client, breaking_news, ai_breaking_news, existing_topics)

    # ====== Stage 2: Brain Dump ======
    brain_dump_text = brain_dump(client, topic_info)

    # Build breaking news context
    breaking_lines = []
    for item in (breaking_news + ai_breaking_news)[:30]:
        breaking_lines.append(f"[{item.get('time', '')}] {item.get('text', '')}")
    breaking_context = "\n".join(breaking_lines)

    # ====== Stage 3: Research Agent Loop ======
    report_text, sources = research_agent_loop(client, topic_info, brain_dump_text, breaking_context)

    # ====== Stage 4: Format to JSON ======
    report = format_to_json(client, report_text)

    # Estimate read time
    total_content = ""
    for sec in report.get("sections", []):
        total_content += sec.get("content", "")
    text_only = re.sub(r'<[^>]+>', '', total_content)
    read_minutes = max(5, len(text_only) // 500)

    # Build entry
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
        "sources": report.get("sources", []),
    }

    # Insert or replace
    if deep_research and deep_research[0].get("id") == date_id:
        deep_research[0] = entry
    else:
        deep_research.insert(0, entry)

    if len(deep_research) > MAX_RESEARCH_ENTRIES:
        deep_research = deep_research[:MAX_RESEARCH_ENTRIES]

    # Write back
    html = replace_js_array(html, "deepResearch", deep_research)
    write_html(html)

    print(f"\n{'=' * 60}")
    print(f"SUCCESS: Deep research report generated!")
    print(f"  Topic: {entry['title']}")
    print(f"  Sections: {len(entry['sections'])}")
    print(f"  Read time: {entry['readTime']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
