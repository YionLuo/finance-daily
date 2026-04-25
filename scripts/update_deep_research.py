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

# Auto-detect model names based on API endpoint
_base_url = os.environ.get("OPENAI_BASE_URL", "")
_is_openrouter = "openrouter" in _base_url

LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek/deepseek-v4-flash" if _is_openrouter else "deepseek-v4-flash")
WRITER_MODEL = os.environ.get("WRITER_MODEL", "deepseek/deepseek-v4-pro" if _is_openrouter else "deepseek-v4-pro")

MAX_AGENT_ROUNDS = 12


# ============ Stage 1: Topic Selection (unchanged) ============

TOPIC_SELECTION_PROMPT = """你是 Y Daily 首席分析师。从今日 Breaking News 中选出最具深度分析价值的 1 个专题。

选题标准：
1. 时效性：最近 24 小时内的重大事件
2. 深度价值：有足够多的角度和数据支撑深度分析
3. 与 AI/互联网/科技行业 或 金融市场（美股/港股）相关
4. 不是简单的事件报道，而是可以挖掘深层逻辑的话题

⚠️ 绝对不能重复的近期报告主题：
{existing_topics}
选题必须与以上所有主题**完全不同**。不能是同一事件的不同角度，不能是同一公司的不同切入点。如果近期已经写过亚马逊/Anthropic，就不要再选任何涉及这两家公司的话题。选一个全新的领域。

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

    response = llm_chat_with_retry(client, [{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.3)
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

BRAIN_DUMP_PROMPT = """你是一位在 AI 和科技金融领域有 15 年经验的资深分析师，服务对象是专业投资人。
当前时间：{current_time}（现在是 2026 年 4 月）

今天你要深度分析的主题是：「{topic}」
角度：{angle}

在你开始搜索任何新信息之前，先梳理你对这个话题的**结构性认知**——重点是分析框架和历史规律，而不是具体产品版本或公司估值（这些会通过搜索获取最新数据）。

请输出：

1. **行业结构**：竞争格局、关键玩家定位、价值链分布、各环节利润率
2. **历史投资案例**：类似事件在历史上驱动了什么具体的股价反应？有哪些赚钱/亏钱的教科书案例？
3. **分析框架**：分析这类问题应该看哪些维度？哪个因素是决定交易机会的关键？
4. **可能的交易结构**：这类事件通常对应哪些交易类型？（并购套利 / 事件驱动 / 行业轮动 / 对冲组合 / 波动率交易等）
5. **常见误区**：大众/媒体在这个话题上容易犯什么错？被高估的是什么？被低估的是什么？
6. **需要搜索验证的关键问题**：你知道大致方向但不确定最新数据的点（列 6-10 个具体问题，后续用搜索回答——尤其要包含"具体的标的公司财务数据"、"进行中的相关交易"、"近期监管判例"）

⚠️ 重要提醒：
- 你的训练数据截止到 2024 年底左右。现在是 2026 年 4 月，AI 行业已经发生了很多变化。
- **不要写具体的产品版本号、模型名称、融资金额、公司估值**——这些很可能已经过时。
- 聚焦于**结构性认知**（竞争逻辑、商业模式、历史规律、交易模式），这些不会因为半年的时间而失效。
- 所有具体数据点标注「待搜索确认」

用中文，1500-2000 字（精炼，重框架轻细节）。
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


# ============ Stage 3a: Research Collection (chat model + tools) ============

RESEARCH_COLLECTOR_PROMPT = """你是 Y Daily 的研究助理，服务对象是专业投资人。正在为「{topic}」收集**投研级**素材（不是评论级）。

当前时间：{current_time}（现在是 2026 年）

你有 web_search 和 fetch_url_content 工具。

你的任务是**只做搜索收集，不写报告**。收集足够的素材后，输出一份结构化的「研究素材包」给分析师。

=== 今日相关新闻 ===
{breaking_context}

=== ⚠️ 关键要求：你必须读全文 ===
web_search 只能返回标题和摘要片段，信息量远远不够写深度投研报告。
你**必须**对搜索结果中最重要的 5-8 篇文章使用 fetch_url_content 读取全文。
如果你只搜不读全文，分析师会因为素材太薄而写出空洞的评论文章。

=== 投研素材的收集要点（和普通新闻评论不同）===
普通评论只需要新闻报道。但投研需要：
- **具体财务数据**：相关公司最新季报、收入/毛利率/现金流、估值倍数
- **进行中的交易案例**：类似话题正在发生的并购/融资/IPO
- **历史案例对比**：过去类似事件驱动了哪些具体股价反应
- **监管判例/行业数据**：近期相关的监管决定、审批尺度
- **多方观点**：看多的理由、看空的理由、中立的质疑
- **资金流向信号**：大行研报、机构持仓变化、期权市场异动

=== 收集流程 ===
1. 先用 web_search 搜索 8-12 个不同角度的关键词，包括：
   - 主话题 + "2026"
   - **涉及的关键公司 + "quarterly results" / "revenue" / "R&D expense" / "capex" / "free cash flow"**（硬财务数据优先）
   - **涉及的关键公司 + "P/E ratio" / "valuation" / "analyst target price"**（估值数据）
   - 主话题 + "deal" / "acquisition" / "merger" / "IPO"
   - 主话题 + "supply chain" + 具体公司名 / "India capacity" / "Vietnam production"（产能和供应链细节）
   - 主话题 + "regulation" / "ruling" / "precedent"
   - 中英文混用，不同渠道都试

2. 从搜索结果中挑出最重要、信息量最大的 5-8 篇文章
3. 对这些文章逐一使用 fetch_url_content 读取全文（这一步不能跳过！）
4. **必做**：至少搜一次相关公司的最新财报或投资者信息页面，拿到真实的 R&D 金额、现金流、营收等数字
5. 搜集够了后，输出以下格式：

---素材汇总---
## 核心事件
（事件的基本事实，用搜索和全文中获取的详细信息）

## 关键数据点
（搜索到的具体数字：财务数据、交易规模、市场份额等，标注来源 URL）

## 相关公司财务信息
（涉及的关键公司的营收、利润、估值、业务结构等，如有）

## 正面观点/看多理由
（支持/看好的分析和论据，标注来源）

## 反面观点/风险
（质疑/看空的分析和论据，标注来源）

## 进行中的相关交易/事件
（同主题下正在发生的其他交易、监管行动、产品发布，标注来源）

## 历史案例对比
（如搜到历史上类似事件的案例数据）

## 行业背景
（相关的行业趋势和竞争动态，标注来源）

## 来源列表
（所有参考文章的标题和 URL）
---素材汇总结束---

只输出搜索到的真实信息。不要编造任何数据或引用来源。如果某个方面搜不到信息，就写"未找到相关信息"。
"""

MAX_COLLECT_ROUNDS = 15


def research_collect(client, topic_info, breaking_context):
    """
    Stage 3a: Collect research materials using chat model + search tools.
    Returns the collected research materials as text.
    """
    print(f"\n=== Stage 3a: Research Collection (max {MAX_COLLECT_ROUNDS} rounds) ===")

    from openai import OpenAI
    from news_fetcher import AGENT_TOOLS, execute_tool_call

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    raw_client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)

    system_prompt = RESEARCH_COLLECTOR_PROMPT.format(
        topic=topic_info.get("topic", ""),
        breaking_context=breaking_context,
        current_time=format_date_cst(now_cst()),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"开始为「{topic_info.get('topic', '')}」收集研究素材。先用 web_search 搜索 8-12 个不同角度的关键词。"},
    ]

    collected_sources = []
    search_count = 0
    fetch_count = 0
    final_materials = None

    for round_num in range(MAX_COLLECT_ROUNDS):
        print(f"\n  --- Collect Round {round_num + 1} ---")

        try:
            response = raw_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=AGENT_TOOLS,
                temperature=0.3,
                max_tokens=8192,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            break

        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                print(f"  Tool: {func_name}({json.dumps(func_args, ensure_ascii=False)[:100]})")
                result = execute_tool_call(func_name, func_args)

                if func_name == "web_search":
                    search_count += 1
                    for line in result.split("\n"):
                        if line.strip().startswith("[") and "]" in line:
                            collected_sources.append(line.strip())

                if func_name == "fetch_url_content":
                    fetch_count += 1

                if len(result) > 5000:
                    result = result[:5000] + "\n...(truncated)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            print(f"  Processed {len(message.tool_calls)} tool call(s), total searches: {search_count}, fetches: {fetch_count}")

            # If searched enough but hasn't read any full articles, push to read
            if search_count >= 6 and fetch_count == 0 and round_num >= 2:
                messages.append({"role": "user", "content": "你已经搜索了足够多的关键词。现在请从搜索结果中挑选最重要的 5 篇文章，使用 fetch_url_content 逐一读取全文。这一步非常关键，搜索摘要的信息量不够写深度报告。"})

            # If not enough searches after round 1, push back
            elif search_count < 5 and round_num == 1:
                messages.append({"role": "user", "content": "继续搜索更多角度，特别是反面观点和具体数据。至少再搜 3-5 次。"})

        else:
            final_materials = message.content or ""
            print(f"  Materials collected: {len(final_materials)} chars, from {search_count} searches")
            break

    if not final_materials:
        messages.append({"role": "user", "content": "请现在输出你收集到的素材汇总。"})
        try:
            response = raw_client.chat.completions.create(
                model=LLM_MODEL, messages=messages, max_tokens=8192, temperature=0.2,
            )
            final_materials = response.choices[0].message.content or ""
        except Exception as e:
            final_materials = "素材收集失败"
            print(f"  Force output failed: {e}")

    return final_materials, collected_sources


KEY_QUESTION_PROMPT = """你是 Y Daily 的分析师。基于以下研究素材，生成这份报告必须回答的「关键问题清单」。

=== 研究素材 ===
{research_materials}

=== 任务 ===
仔细阅读素材，找出这份报告如果要做到"专业深度"，必须回答的关键问题。

问题类型包括：
1. **核心假设验证型**：报告隐含的假设（如"X 导致 Y"），需要什么数据才能验证？
2. **财务穿透型**：涉及的公司，哪条业务线受影响？影响多少营收/利润？
3. **反面力量型**：谁会反抗这个结论？有什么反面证据？
4. **历史类比型**：类似事件过去发生过吗？结果如何？
5. **催化剂型**：未来什么事件能验证/推翻核心假设？

=== 输出格式（严格 JSON）===
{{
  "questions": [
    {{
      "id": "Q1",
      "type": "假设验证型",
      "question": "问题原文",
      "why_crucial": "为什么这个问题必须回答（1句话）",
      "current_status": "待搜索 / 部分回答 / 已回答",
      "evidence_snippet": "素材中相关的片段（如果有的话）"
    }}
  ]
}}

生成 5-8 个问题。问题必须具体，不能泛泛而谈。
例如❌ "英伟达未来会怎样？"
例如✅ "Graviton 在 AI 推理场景的 benchmark 数据 vs H200，性能/成本差距是多少？当前素材是否包含此数据？"

只输出 JSON，不要其他文字。
"""

def generate_key_questions(client, research_materials):
    """Stage 3.25: Generate key questions that the report must answer."""
    print("\n=== Stage 3.25: Generate Key Questions ===")

    prompt = KEY_QUESTION_PROMPT.format(
        research_materials=research_materials[:20000]
    )

    response = llm_chat_with_retry(
        client,
        [{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.2,
    )

    # Parse JSON
    cleaned = re.sub(r'^```(?:json)?\s*\n', '', response, flags=re.MULTILINE)
    cleaned = re.sub(r'\n```\s*$', '', cleaned, flags=re.MULTILINE).strip()
    think_end = cleaned.find("</think>")
    if think_end != -1:
        cleaned = cleaned[think_end + len("</think>"):].strip()

    try:
        result = json.loads(cleaned)
        questions = result.get("questions", [])
        print(f"  Generated {len(questions)} key questions:")
        for q in questions:
            print(f"    {q['id']} [{q['type']}] {q['question'][:60]}... (status: {q['current_status']})")
        return questions
    except (json.JSONDecodeError, AttributeError):
        print("  WARNING: Failed to parse questions JSON, using fallback")
        return []


# ============ Stage 3b: Report Writing (reasoning model, no tools) ============

WRITER_PROMPT = """你是 Y Daily 的首席分析师。你的任务是产出专业、克制、有深度的投研分析报告。

当前时间：{current_time}

=== 关键问题清单（必须逐一回答，不能跳过）===
{key_questions}

=== 分析框架参考 ===
{brain_dump}

=== 研究素材（全部来自真实搜索，可直接引用）===
{research_materials}

=== 写作原则（严格遵守）===
1. **围绕问题写，不要自由发挥**
   报告的每一个章节都必须对应"关键问题清单"中的某一个或几个问题。
   如果某个问题素材不足以回答，明确写"当前证据不足，需要关注 X 信号"，不要猜测。

2. **多空观点必须并列呈现**
   不要强行统一结论。如果证据支持多方，就写多方；支持空方，就写空方；都有，就并列。
   每条观点必须标注证据强度：【已验证】有 2 个以上独立来源
                             【部分验证】只有 1 个来源或数据不完整
                             【未验证-推测】纯逻辑推导，无数据支撑

3. **禁止输出交易指令**
   不要写"做多 XXX"、"做空 XXX"、"建议买入/卖出"。
   改为写："如果 X 条件被验证（见催化剂日历），则 Y 情景可能触发，对 Z 标的的影响方向是…，幅度取决于 W。"

4. **假设必须显式标注**
   文中每一个"因为 A 所以 B"的推导，如果 A 是假设（未验证），必须用括号标注（假设，待验证）。
   如果通篇都是假设，在章节开头明确标注：⚠️ 本节主要基于未验证假设，确定性低。

5. **数据严格要求**（同之前）
   - 只使用素材中明确包含的数据
   - 无具体数字时用定性分析 + 标注"（基于行业逻辑推断，需核查）"
   - 禁止编造引用来源

=== 报告结构（必须严格按照这个结构）===
1. **核心问题**（1 段）
   用 1-2 句话说明：这份报告试图回答什么关键问题？当前哪些已有答案、哪些仍不确定？

2. **关键假设与验证状态**（1 个小节，可用表格）
   列出报告依赖的核心假设，每个标注：【已验证】/【部分验证】/【未验证】
   这是读者判断报告可信度的核心参考，不能省略。

3. **分问题深度分析**（2-4 个小节）
   每个小节对应 1-2 个关键问题。
   小节内先写"多方观点"（证据 + 强度标注），再写"空方观点"（证据 + 强度标注）。
   不要强行给出结论。如果证据不足以判断，直接写"当前证据不足以判断，需等待 X 信号"。

4. **我们的判断**（1 段，确定性评级）
   基于已验证的部分，给出一个总体判断，但必须附带确定性评级：
   - 🟢 高确定性（已验证假设 ≥70%，有硬数据支撑）
   - 🟡 中等确定性（已验证假设 40-70%，或关键数据缺失）
   - 🔴 低确定性（主要基于未验证假设，或正反证据强度接近）
   如果评级是🟡或🔴，必须说明"要提升确定性，需要验证 X、Y、Z"。

5. **催化剂日历与待验证信号**（1 个小节）
   列出未来 1-2 周内可能验证关键假设的具体事件：
   - 财报发布日（公司名 + 日期）
   - 行业会议 / 产品发布
   - 监管决定
   每个事件标注：如果事件结果 X，则验证/推翻哪个假设。

=== 写作风格 ===
- Stratechery 的结构感 + SemiAnalysis 的数据穿透
- 克制。不确定就说不确定。
- 每一个结论都必须有对应的证据强度标注。
- 禁止："短期/中期/长期"、"存在不确定性"、"需要密切关注"（这些都是废话，改成具体描述）

目标长度：3500-5000 字（质量 > 长度）。
"""


def write_report(client, topic_info, brain_dump_text, research_materials, key_questions=None):
    """
    Stage 3b: Write the deep analysis report using reasoning model (R1).
    No tool use — pure text generation with deep thinking.
    key_questions: list of question dicts from generate_key_questions()
    """
    # Format key questions for prompt
    questions_text = ""
    if key_questions:
        lines = []
        for q in key_questions:
            status_icon = {"待搜索": "❓", "部分回答": "⚡", "已回答": "✅"}.get(q.get('current_status', ''), '❓')
            lines.append(f"{q['id']}. [{q['type']}] {q['question']} {status_icon} {q.get('why_crucial', '')}")
        questions_text = "\n".join(lines)
    else:
        questions_text = "（未生成关键问题清单，请自行判断分析重点）"

    # 素材质量检查：如果缺乏硬数据，先补搜
    has_numbers = bool(re.findall(r'\$[\d.]+[BMT]|\d+%|\d+亿|\d+万亿|revenue|营收|利润|市值', research_materials))
    if not has_numbers:
        print("  ⚠️ Materials lack hard financial data, requesting supplementary search...")
        # 用 LLM 快速搜几个关键数字
        from news_fetcher import web_search
        topic = topic_info.get("topic", "")
        supplement = []
        for q in [f"{topic} revenue 2026", f"{topic} market cap", f"{topic} quarterly earnings"]:
            results = web_search(q, max_results=3)
            for r in results:
                supplement.append(f"[补充搜索] {r.get('title', '')} - {r.get('summary', '')[:200]}")
        if supplement:
            research_materials += "\n\n## 补充财务数据搜索\n" + "\n".join(supplement)
            print(f"  Added {len(supplement)} supplementary results")

    print(f"\n=== Stage 3b: Report Writing (model: {WRITER_MODEL}) ===")

    prompt = WRITER_PROMPT.format(
        brain_dump=brain_dump_text,
        research_materials=research_materials[:25000],
        current_time=format_date_cst(now_cst()),
        key_questions=questions_text,
    )

    try:
        response = client.chat.completions.create(
            model=WRITER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.6,
        )
        text = response.choices[0].message.content or ""

        # Strip <think> block if present (R1 reasoning trace)
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()

        print(f"  Report written: {len(text)} chars")
        return text
    except Exception as e:
        print(f"  ERROR writing with {WRITER_MODEL}: {e}")
        print(f"  Falling back to {LLM_MODEL}...")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.5,
        )
        text = response.choices[0].message.content or ""
        print(f"  Fallback report: {len(text)} chars")
        return text



# ============ Stage 3.5: Fact Check (with search verification) ============

FACT_CHECK_SYSTEM = """你是一位严格的事实核查编辑。你的工作是验证一篇深度分析报告中的事实性声明和逻辑质量。

⚠️ 你和报告的作者是不同的人。作者可能会编造看起来很真实的数据。你不能相信任何没有可靠来源的具体数字。

你有 web_search 工具可以验证事实。

=== 待核查报告 ===
{report_text}

=== 你的工作流程 ===

第一步：提取所有事实性声明
从报告中找出所有**具体的事实性声明**，特别是：
- 具体金额（融资、估值、营收、市值）
- 具体百分比（市场份额、增长率、转化率）
- 具体事件（收购、合作、诉讼、产品发布）
- 公司间的具体合作关系
- 引用的报告名称或研究机构
- 产品版本号和发布时间

第二步：用 web_search 验证
对每个关键声明，用 web_search 搜索验证。比如：
- 报告说"亚马逊投资 Anthropic 250 亿" → 搜索 "Amazon Anthropic investment amount 2026"
- 报告说"OpenAI 与腾讯合资入华" → 搜索 "OpenAI Tencent joint venture China 2026"
- 报告说"某公司估值 800 亿" → 搜索 "company name valuation 2026"

第三步：检查逻辑质量与投研专业度
除了事实核查，还要检查：
- **不当类比**：历史类比是否准确？比如"A公司收购B，和C收购D一样"——但如果实际情况有重大差异（如D在被收后仍有重大突破），就应该指出类比不成立
- **遗漏关键玩家**：分析产业格局时是否遗漏了不可忽视的参与者（如谈AI芯片不提英伟达、谈云市场不提阿里云/华为云等）
- **财务逻辑漏洞**：大额承诺是否有可行性分析？钱从哪来？是否存在循环融资嫌疑？
- **投研专业度缺失**：报告是否只停留在"评论"层面？
  - 提到了股票代码（relatedTickers）但正文中没有对应的具体财务影响分析 → missing_player
  - 用了"短期/中期/长期"这种无交易价值的时间颗粒度 → logic_gap
  - 只说"需要关注"但没说关注什么具体信号 → logic_gap
  - 只说"利好/利空某行业"但没列出具体标的或交易结构 → logic_gap
- **假设未验证却当事实用**（新增）：报告中是否有"因为 A 所以 B"，但 A 是未验证的假设？
  - 例如：报告说"Graviton 性能优于 H200，所以 NVDA 市场份额将下降"，但素材里没有 Graviton vs H200 的 benchmark 数据
  - 这种"逻辑跳跃"比事实错误更危险，必须标记为 assumption_not_backed

第四步：检查报告内部一致性（新增）
- 报告前面说"做多 X"，后面说"做空 X" → self_contradiction
- 报告标注某假设"未验证"，但结论却基于该假设给出确定性判断 → consistency_gap
- 多方和空方论据强度明显不对称，但报告没有指出 → bias_unchecked

第五步：输出核查结果
完成搜索验证后，输出 JSON：
{{
  "total_claims": 核查的事实条数,
  "verified": 搜索确认正确的条数,
  "unverifiable": 搜索找不到佐证但也没有反证的条数,
  "false_or_fabricated": 搜索结果与声明矛盾、或完全找不到任何相关信息的条数,
  "logic_issues": 逻辑问题的条数,
  "assumption_issues": 假设未验证却当事实用的条数,
  "consistency_issues": 报告内部自相矛盾的条数,
  "issues": [
    {{
      "claim": "有问题的声明原文",
      "search_result": "你搜到的实际信息是什么",
      "verdict": "false / fabricated / unverifiable / bad_analogy / missing_player / logic_gap / assumption_not_backed / self_contradiction / consistency_gap",
      "fix": "建议的修正文本（如果应该删除就写'删除此句'）"
    }}
  ]
}}

verdict 含义：
- false: 与搜索到的事实矛盾
- fabricated: 完全找不到相关信息，很可能是编造的
- unverifiable: 无法确认
- bad_analogy: 历史类比不准确或有重大遗漏
- missing_player: 分析遗漏了关键参与者
- logic_gap: 财务/商业逻辑有漏洞
- assumption_not_backed: 假设未验证却当事实用（如"因为 A 所以 B"，但 A 未验证）
- self_contradiction: 报告内部自相矛盾（前面说 X，后面说非 X）
- consistency_gap: 标注"未验证"但结论却很确定

重要：只输出最终 JSON。每个 issue 都必须有 search_result 说明你搜到了什么。
"""

MAX_FACT_CHECK_ROUNDS = 8


def fact_check(client, report_text):
    """
    Stage 3.5: Fact-check the report using web search verification.
    The fact checker is an independent agent that searches to verify claims.
    Returns: (cleaned_report_text, fact_check_result)
    """
    print("\n=== Stage 3.5: Fact Check (search-verified) ===")

    from news_fetcher import AGENT_TOOLS, execute_tool_call

    system_prompt = FACT_CHECK_SYSTEM.format(report_text=report_text[:10000])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "开始核查。先列出需要验证的关键事实声明，然后逐个用 web_search 验证。"},
    ]

    final_result = None

    for round_num in range(1, MAX_FACT_CHECK_ROUNDS + 1):
        print(f"\n  --- Fact Check Round {round_num} ---")

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=AGENT_TOOLS,
                temperature=0.1,
                max_tokens=8192,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            break

        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                print(f"  Verify: {func_name}({json.dumps(func_args, ensure_ascii=False)[:100]})")
                result = execute_tool_call(func_name, func_args)

                if len(result) > 3000:
                    result = result[:3000] + "\n...(truncated)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            print(f"  Verified {len(message.tool_calls)} claim(s)")

        else:
            # Final output — parse JSON result
            text = message.content or ""
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE).strip()

            try:
                final_result = json.loads(cleaned)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                json_match = re.search(r'\{[\s\S]*\}', cleaned)
                if json_match:
                    try:
                        final_result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        print(f"  WARNING: Could not parse fact check JSON")
                        final_result = None
            break

    if not final_result:
        print("  Fact check did not produce parseable result, skipping corrections")
        return report_text, None

    # Print summary
    total = final_result.get("total_claims", 0)
    verified = final_result.get("verified", 0)
    fabricated = final_result.get("false_or_fabricated", 0)
    logic_issues = final_result.get("logic_issues", 0)
    issues = final_result.get("issues", [])

    print(f"\n  Claims checked: {total}")
    print(f"  Verified: {verified}, Unverifiable: {final_result.get('unverifiable', 0)}, False/Fabricated: {fabricated}, Logic issues: {logic_issues}")

    if issues:
        print(f"  Issues ({len(issues)}):")
        for issue in issues[:10]:
            verdict = issue.get("verdict", "?")
            icon = "🚫" if verdict in ("false", "fabricated") else "🔍" if verdict in ("bad_analogy", "missing_player", "logic_gap") else "⚠️"
            print(f"    {icon} [{verdict}] {issue.get('claim', '')[:80]}")
            print(f"       搜索结果: {issue.get('search_result', '')[:80]}")
            print(f"       修正: {issue.get('fix', '')[:80]}")

    # Apply corrections for both factual errors AND logic issues
    fixable_issues = [i for i in issues if i.get("verdict") in ("false", "fabricated", "bad_analogy", "missing_player", "logic_gap", "assumption_not_backed", "self_contradiction", "consistency_gap")]

    if fixable_issues:
        print(f"\n  Applying corrections for {len(fixable_issues)} false/fabricated claims...")

        fix_prompt = f"""以下报告有 {len(fixable_issues)} 处经核查发现的问题，必须修正。

=== 原始报告 ===
{report_text[:10000]}

=== 经核查发现的问题 ===
{json.dumps(fixable_issues, ensure_ascii=False, indent=2)}

修正规则：
1. 对于 "false" 的：用 fix 字段建议的正确表述替换
2. 对于 "fabricated" 的：直接删除该句或该段落，不要用新编的内容替换
3. 对于 "bad_analogy" 的：修正类比，说明适用边界，或用更准确的类比替换
4. 对于 "missing_player" 的：在相关段落补充对遗漏玩家的分析
5. 对于 "logic_gap" 的：补充财务/商业逻辑分析，质疑可行性
6. 不要添加任何新的事实声明（逻辑分析除外）
7. 保持报告其余部分不变

输出修正后的完整报告文本。"""

        fixed = llm_chat_with_retry(
            client, [{"role": "user", "content": fix_prompt}],
            max_tokens=8192, temperature=0.1,
        )

        if fixed and len(fixed) > len(report_text) * 0.4:
            print(f"  Report corrected: {len(report_text)} → {len(fixed)} chars")
            return fixed, final_result
        else:
            print(f"  Correction failed, keeping original")

    return report_text, final_result


# ============ Stage 4: Format to JSON ============

FORMAT_PROMPT = """把以下深度分析报告转换为 JSON 格式。保留所有内容，只改变格式。

⚠️ 格式化规则：
- 标题和副标题中的引号必须配对
- 中文使用中文标点（""、''），不要混用半角引号
- sections content 中的 HTML 必须完整闭合
- 输出必须是合法 JSON，不要输出 markdown 代码块标记

=== 原始报告 ===
{report_text}

=== 输出 JSON ===
严格输出以下 JSON 结构（不要 markdown 代码块、不要任何前缀文字）：
{{
  "title": "报告标题（从报告中提取，如有Y Daily投研前缀请保留）",
  "subtitle": "副标题（报告日期、服务对象等信息）",
  "summary": "200字以内摘要（概括核心论点与交易推论）",
  "tags": [
    {{"text": "标签名", "type": "up|down|warn"}}
  ],
  "keyTakeaways": ["核心判断1（含具体操作建议）", "核心判断2", "核心判断3"],
  "relatedTickers": ["AAPL", "NVDA"],
  "sections": [
    {{
      "id": "sec-1",
      "title": "章节标题",
      "content": "<p>章节正文HTML</p>"
    }}
  ],
  "sourceIndices": []
}}

tags type: "up"=利好, "down"=利空, "warn"=警示。至少3个tags。
keyTakeaways: 至少3条。
relatedTickers: 报告中提到的所有股票代码。
sections: 每个一级章节（如：核心论点、事件背景、财务穿透分析、交易机会池、风险与反面力量、催化剂日历、核心判断）作为一个section。
sections 的 content 用 HTML：<p>段落、<strong>加粗、<ul><li>列表、<table>表格、<ol>有序列表、<h3>子标题。
"""


def _clean_llm_json(response):
    """Clean LLM response and extract JSON."""
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', response, flags=re.MULTILINE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE).strip()
    # Strip <think> block if present
    think_end = cleaned.find("</think>")
    if think_end != -1:
        cleaned = cleaned[think_end + len("</think>"):].strip()
    # Try to find JSON object boundaries
    first_brace = cleaned.find('{')
    if first_brace > 0:
        cleaned = cleaned[first_brace:]
    # Find matching closing brace
    depth = 0
    last_brace = -1
    for i, c in enumerate(cleaned):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                last_brace = i
                break
    if last_brace > 0:
        cleaned = cleaned[:last_brace + 1]
    return cleaned


def _md_to_html(text):
    """Basic Markdown to HTML conversion for fallback."""
    lines = text.split('\n')
    html_parts = []
    in_list = False
    in_table = False
    table_rows = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            if in_table and table_rows:
                html_parts.append(_build_table(table_rows))
                table_rows = []
                in_table = False
            html_parts.append('')
            continue

        # Table row
        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            table_rows.append(stripped)
            continue
        elif in_table and table_rows:
            html_parts.append(_build_table(table_rows))
            table_rows = []
            in_table = False

        # Headers
        if stripped.startswith('### '):
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            html_parts.append(f'<h3>{_inline_md(stripped[4:])}</h3>')
        elif stripped.startswith('## '):
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            html_parts.append(f'<h3>{_inline_md(stripped[3:])}</h3>')
        elif stripped.startswith('# '):
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            # Skip top-level title (already extracted)
            continue
        elif stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list:
                html_parts.append('<ul>')
                in_list = True
            html_parts.append(f'<li>{_inline_md(stripped[2:])}</li>')
        elif re.match(r'^\d+\.\s', stripped):
            content = re.sub(r'^\d+\.\s', '', stripped)
            if not in_list:
                html_parts.append('<ol>')
                in_list = True
            html_parts.append(f'<li>{_inline_md(content)}</li>')
        elif stripped == '---' or stripped == '***':
            continue  # Skip horizontal rules
        else:
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            html_parts.append(f'<p>{_inline_md(stripped)}</p>')

    if in_list:
        html_parts.append('</ul>')
    if in_table and table_rows:
        html_parts.append(_build_table(table_rows))

    return '\n'.join(html_parts)


def _inline_md(text):
    """Convert inline markdown (bold, italic, links) to HTML."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    return text


def _build_table(rows):
    """Build HTML table from markdown table rows."""
    if len(rows) < 2:
        return ''
    html = '<table><thead><tr>'
    headers = [c.strip() for c in rows[0].strip('|').split('|')]
    for h in headers:
        html += f'<th>{_inline_md(h)}</th>'
    html += '</tr></thead><tbody>'
    for row in rows[2:]:  # Skip separator row
        if row.strip().replace('-', '').replace('|', '').replace(' ', '') == '':
            continue
        cells = [c.strip() for c in row.strip('|').split('|')]
        html += '<tr>'
        for c in cells:
            html += f'<td>{_inline_md(c)}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html


def _split_md_sections(text):
    """Split markdown text into sections by top-level headers (## or # with number)."""
    lines = text.split('\n')
    sections = []
    current_title = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        # Match section headers: ## N. Title or ## Title
        is_section = False
        if re.match(r'^#{1,2}\s+(?:\d+[\.\、]?\s*)?', stripped):
            # Skip the very first # title (report title)
            title_text = re.sub(r'^#{1,2}\s+(?:\d+[\.\、]?\s*)?', '', stripped).strip()
            if title_text and not stripped.startswith('### '):
                is_section = True

        if is_section:
            if current_title is not None and current_lines:
                sections.append((current_title, '\n'.join(current_lines)))
            current_title = title_text
            current_lines = []
        else:
            current_lines.append(line)

    if current_title is not None and current_lines:
        sections.append((current_title, '\n'.join(current_lines)))

    # If no sections found, treat entire text as one section
    if not sections:
        sections = [("分析报告", text)]

    return sections


def _extract_title_from_md(text):
    """Extract the first # or ## heading as title."""
    for line in text.split('\n')[:10]:
        stripped = line.strip()
        if stripped.startswith('# ') and not stripped.startswith('## '):
            return stripped[2:].strip()
    return None


def _extract_tickers_from_text(text):
    """Extract stock ticker symbols from text."""
    # Match patterns like (AAPL), (NVDA), (TSM), ticker: AAPL, etc.
    tickers = set()
    # US tickers in parentheses
    for m in re.finditer(r'\(([A-Z]{1,5})\)', text):
        t = m.group(1)
        if len(t) >= 2 and t not in {'AI', 'US', 'EU', 'UK', 'HK', 'CN', 'GDP', 'CPI', 'PPI',
                                       'IEA', 'IPO', 'ETF', 'CEO', 'CFO', 'CTO', 'PPA', 'IRR',
                                       'THE', 'FOR', 'AND', 'BUT', 'NOT', 'ARE', 'WAS', 'HAS',
                                       'PCB', 'UAE', 'IMF', 'SPR', 'LNG', 'PPA', 'TCO', 'API'}:
            tickers.add(t)
    return list(tickers)[:15]


def _extract_tags_from_text(text, tickers):
    """Extract meaningful tags from report text."""
    tags = []
    # Look for key phrases that indicate direction
    up_patterns = [r'做多\s*[「""]?([^「""」\s,，]{2,8})', r'增持\s*[「""]?([^「""」\s,，]{2,8})']
    down_patterns = [r'做空\s*[「""]?([^「""」\s,，]{2,8})', r'减仓\s*[「""]?([^「""」\s,，]{2,8})']
    warn_patterns = [r'观望\s*[「""]?([^「""」\s,，]{2,8})', r'风险\s*[：:]\s*[「""]?([^「""」\s,，]{2,8})']

    for pat in up_patterns:
        for m in re.finditer(pat, text):
            t = m.group(1).strip('」""')
            if 2 <= len(t) <= 8:
                tags.append({"text": t, "type": "up"})
    for pat in down_patterns:
        for m in re.finditer(pat, text):
            t = m.group(1).strip('」""')
            if 2 <= len(t) <= 8:
                tags.append({"text": t, "type": "down"})
    for pat in warn_patterns:
        for m in re.finditer(pat, text):
            t = m.group(1).strip('」""')
            if 2 <= len(t) <= 8:
                tags.append({"text": t, "type": "warn"})

    # Deduplicate
    seen = set()
    unique_tags = []
    for tag in tags:
        key = tag["text"]
        if key not in seen:
            seen.add(key)
            unique_tags.append(tag)

    # If not enough tags, add from tickers
    if len(unique_tags) < 3 and tickers:
        for t in tickers[:3]:
            if t not in seen:
                unique_tags.append({"text": t, "type": "warn"})
                seen.add(t)

    return unique_tags[:8]


def format_to_json(client, report_text):
    """Stage 4: Convert plain-text analysis into structured JSON."""
    print("\n=== Stage 4: Format to JSON ===")
    print(f"  Report text length: {len(report_text)} chars")

    # Try LLM formatting with retry
    max_input = 24000
    truncated = report_text[:max_input]
    prompt = FORMAT_PROMPT.format(report_text=truncated)

    for attempt in range(2):
        if attempt > 0:
            print(f"  Retry #{attempt}...")

        response = llm_chat_with_retry(
            client, [{"role": "user", "content": prompt}],
            max_tokens=8192, temperature=0.1,
        )

        cleaned = _clean_llm_json(response)

        try:
            result = json.loads(cleaned)
            # Validate: must have sections with real content
            sections = result.get("sections", [])
            if sections and len(sections) >= 2:
                print(f"  ✅ JSON parsed: {len(sections)} sections, title='{result.get('title', '')[:40]}'")
                return result
            elif sections:
                print(f"  ⚠️ JSON parsed but only {len(sections)} section(s), retrying...")
            else:
                print(f"  ⚠️ JSON parsed but no sections, retrying...")
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse failed (attempt {attempt+1}): {e}")
            print(f"  Raw (first 300): {cleaned[:300]}")

    # ====== Fallback: Parse Markdown directly ======
    print("  ⚠️ LLM formatting failed, using Markdown fallback parser")

    title = _extract_title_from_md(report_text) or "深度分析报告"
    tickers = _extract_tickers_from_text(report_text)
    tags = _extract_tags_from_text(report_text, tickers)
    md_sections = _split_md_sections(report_text)

    sections = []
    all_content_text = []
    for i, (sec_title, sec_content) in enumerate(md_sections):
        html_content = _md_to_html(sec_content)
        sections.append({
            "id": f"sec-{i+1}",
            "title": f"{i+1}. {sec_title}" if not re.match(r'^\d', sec_title) else sec_title,
            "content": html_content,
        })
        all_content_text.append(sec_content)

    # Build summary from first 200 chars of content
    full_text = '\n'.join(all_content_text)
    summary_text = re.sub(r'[#*\-|]', '', full_text[:300]).strip()
    summary_text = re.sub(r'\s+', ' ', summary_text)[:200]

    # Extract key takeaways from "核心判断" or last section
    key_takeaways = []
    for sec_title, sec_content in md_sections:
        if '核心判断' in sec_title or '执行摘要' in sec_title or 'Key Takeaway' in sec_title:
            for line in sec_content.split('\n'):
                stripped = line.strip()
                if re.match(r'^[\d\-\*]\s*[\.\、]?\s*', stripped) and len(stripped) > 20:
                    clean = re.sub(r'^[\d\-\*]\s*[\.\、]?\s*', '', stripped)
                    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
                    if len(clean) > 20:
                        key_takeaways.append(clean[:200])
            break

    result = {
        "title": title,
        "subtitle": "",
        "summary": summary_text,
        "tags": tags,
        "keyTakeaways": key_takeaways[:5],
        "relatedTickers": tickers,
        "sections": sections,
        "sourceIndices": [],
    }

    print(f"  Fallback result: {len(sections)} sections, {len(tags)} tags, {len(tickers)} tickers")
    return result


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

    # ====== Stage 2.5: Pre-fetch top breaking news articles ======
    print("\n=== Stage 2.5: Pre-fetch Breaking News Articles ===")
    prefetch_articles = []
    topic_lower = topic_info.get("topic", "").lower()
    all_breaking = breaking_news + ai_breaking_news

    # Collect URLs from breaking news (skip Google News redirects)
    fetchable_urls = []
    for item in all_breaking:
        url = item.get("url", "")
        if url and "news.google.com" not in url:
            fetchable_urls.append((item.get("text", "")[:80], url))

    # Fetch up to 8 articles
    from news_fetcher import fetch_url_content
    for title, url in fetchable_urls[:8]:
        try:
            content = fetch_url_content(url, max_chars=5000)
            if content and not content.startswith("Error") and len(content) > 200:
                prefetch_articles.append(f"### {title}\nSource: {url}\n{content}\n")
                print(f"  ✅ {title[:60]}... ({len(content)}c)")
            else:
                print(f"  ❌ {title[:60]}... (failed or too short)")
        except Exception as e:
            print(f"  ❌ {title[:60]}... ({e})")

    prefetch_text = "\n".join(prefetch_articles)
    print(f"  Pre-fetched: {len(prefetch_articles)} articles, {len(prefetch_text)} chars total")

    # Combine breaking context with pre-fetched content
    enriched_context = breaking_context
    if prefetch_text:
        enriched_context += "\n\n=== 以下是预抓取的相关文章全文 ===\n" + prefetch_text

    # ====== Stage 3a: Research Collection (chat model + tools) ======
    research_materials, sources = research_collect(client, topic_info, enriched_context)

    # ====== Stage 3.25: Generate Key Questions ======
    key_questions = generate_key_questions(client, research_materials)

    # ====== Stage 3b: Report Writing (reasoning model) ======
    # Combine research materials with pre-fetched articles for maximum context
    full_materials = research_materials
    if prefetch_text:
        full_materials += "\n\n=== 预抓取的新闻全文（高质量来源）===\n" + prefetch_text
    report_text = write_report(client, topic_info, brain_dump_text, full_materials, key_questions)

    # ====== Stage 3.5: Fact Check ======
    report_text, fact_check_result = fact_check(client, report_text)

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
