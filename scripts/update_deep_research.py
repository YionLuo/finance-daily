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
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
WRITER_MODEL = os.environ.get("WRITER_MODEL", "deepseek-reasoner")  # DeepSeek R1 reasoning model
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
当前时间：{current_time}（现在是 2026 年 4 月）

今天你要深度分析的主题是：「{topic}」
角度：{angle}

在你开始搜索任何新信息之前，先梳理你对这个话题的**结构性认知**——重点是分析框架和历史规律，而不是具体产品版本或公司估值（这些会通过搜索获取最新数据）。

请输出：

1. **行业结构**：这个领域的竞争格局、关键玩家的定位、商业模式
2. **历史规律**：类似事件在历史上的先例、周期性模式、可以类比的案例
3. **分析框架**：分析这类问题应该看哪些维度？什么因素是决定性的？
4. **常见误区**：大众/媒体在这个话题上容易犯什么错？被高估的是什么？被低估的是什么？
5. **需要搜索验证的关键问题**：你知道大致方向但不确定最新数据的点（列 5-8 个具体问题，后续用搜索回答）

⚠️ 重要提醒：
- 你的训练数据截止到 2024 年底左右。现在是 2026 年 4 月，AI 行业已经发生了很多变化。
- **不要写具体的产品版本号、模型名称、融资金额、公司估值**——这些很可能已经过时。
- 比如：不要写"Claude 3"或"GPT-4"这种具体版本，因为现在可能已经是 Claude 4.6 和 GPT-5 了。
- 聚焦于**结构性认知**（竞争逻辑、商业模式、行业规律），这些不会因为半年的时间而失效。
- 所有具体数据点标注「待搜索确认」

用中文，1500 字左右（精炼一些，重框架轻细节）。
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

RESEARCH_COLLECTOR_PROMPT = """你是 Y Daily 的研究助理，正在为「{topic}」收集分析素材。

当前时间：{current_time}（现在是 2026 年）

你有 web_search 和 fetch_url_content 工具。

你的任务是**只做搜索收集，不写报告**。收集足够的素材后，输出一份结构化的「研究素材包」给分析师。

=== 今日相关新闻 ===
{breaking_context}

=== 收集要求 ===
1. 搜索 8-12 个不同角度的关键词（中英文都试，搜索时带 "2026"）
2. 对最重要的 3-5 篇文章用 fetch_url_content 读取全文
3. 寻找：正面报道、反面观点、具体数据（财报/融资/市场规模）、行业分析文章
4. 搜集够了后，输出以下格式：

---素材汇总---
## 核心事件
（事件的基本事实，只用搜索到的信息）

## 关键数据点
（搜索到的具体数字，标注来源 URL）

## 正面观点
（支持/看好的分析和论据，标注来源）

## 反面观点/风险
（质疑/看空的分析和论据，标注来源）

## 行业背景
（相关的行业趋势和竞争动态，标注来源）

## 来源列表
（所有参考文章的标题和 URL）
---素材汇总结束---

只输出搜索到的真实信息。不要编造任何数据或引用来源。如果某个方面搜不到信息，就写"未找到相关信息"。
"""

MAX_COLLECT_ROUNDS = 10


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

                if len(result) > 5000:
                    result = result[:5000] + "\n...(truncated)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            print(f"  Processed {len(message.tool_calls)} tool call(s), total searches: {search_count}")

            # If not enough searches after round 1, push back
            if search_count < 5 and round_num == 1:
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


# ============ Stage 3b: Report Writing (reasoning model, no tools) ============

WRITER_PROMPT = """你是 Y Daily 的首席分析师。以下是研究助理为你收集的素材，请基于这些素材撰写一篇深度分析报告。

当前时间：{current_time}（2026 年 4 月）

=== 分析框架参考 ===
{brain_dump}

=== 研究素材（全部来自真实搜索，可直接引用）===
{research_materials}

=== Y Daily 的分析偏好 ===
- 核心关注：AI 和互联网行业的技术/产品/商业动态
- 投资视角：关注对美股和港股的影响
- 读者画像：AI 从业者和专业投资者——不需要科普，需要洞察
- 风格：有清晰的核心论点，敢于下判断

=== 关于数据的严格要求 ===
- **只使用上面素材中明确包含的数据和事实**
- 如果素材里没有具体数字，就用定性分析——"显著增长""占据主导地位"比编一个百分比强得多
- 绝对不要编造引用来源（如 Gartner、McKinsey、IDC 报告）——除非素材中真的有
- 善用历史类比、逻辑推理、对比分析来展开论证——这比假数据有价值得多

=== 必须覆盖的分析维度（漏掉任何一个都算不合格）===
1. **财务逻辑检验**：涉及大额交易/投资时，必须质疑钱从哪来、商业模型是否成立、是否存在"左手倒右手"或"卖方融资"嫌疑。不要只说"百亿投资"就完事，要算账。
2. **产业链全景**：不只分析当事双方，必须覆盖上下游关键玩家的受益/受损。比如谈云+AI绑定，就不能漏掉芯片上游（英伟达/ASML）和替代路径（开源生态）。
3. **反面力量与替代路径**：巨头闭环越强，反面力量（开源、监管、解耦需求）就越重要。必须分析"如果这个趋势持续，谁会反抗、怎么反抗"。
4. **历史类比的精确性**：用历史类比时，必须指出类比的**适用边界**——哪里像、哪里不像。不能简单说"和XX一样"。
5. **二阶效应**：不只分析直接影响，还要分析"因为A所以B，因为B所以C"的连锁反应。

=== 报告要求 ===
直接输出中文文章。

结构：
- 标题（有态度、有判断，注意引号配对完整）
- 副标题（核心论点一句话）
- 正文 5-7 个章节，每章有小标题
- 每个章节要展开论证：不只是说结论，要解释为什么、逻辑链条是什么、反面怎么看
- 善用对比（和竞争对手比、和历史事件比）和类比来帮助理解
- 来源标注用素材中提供的真实 URL
- 末尾附「核心判断」（4-5 条，**每条要有冲击力和明确立场，不要四平八稳的废话**）和「关注标的」（美股/港股代码 + 逻辑）

核心判断的写法示例（好 vs 差）：
- ❌ 差："1000亿美元支出承诺将加剧算力市场马太效应"（太平淡）
- ✅ 好："百亿投资换千亿云订单，本质是变相的算力垄断与云收入'左手倒右手'，技术主导权正不可逆地向云巨头倾斜"（有判断、有态度）

写作风格参考：Stratechery（Ben Thompson）、Money Stuff（Matt Levine）——有观点、有逻辑、不装腔作势。
"""


def write_report(client, topic_info, brain_dump_text, research_materials):
    """
    Stage 3b: Write the deep analysis report using reasoning model (R1).
    No tool use — pure text generation with deep thinking.
    """
    print(f"\n=== Stage 3b: Report Writing (model: {WRITER_MODEL}) ===")

    prompt = WRITER_PROMPT.format(
        brain_dump=brain_dump_text,
        research_materials=research_materials[:15000],
        current_time=format_date_cst(now_cst()),
    )

    try:
        response = client.chat.completions.create(
            model=WRITER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16384,
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
            max_tokens=16384,
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

第三步：检查逻辑质量
除了事实核查，还要检查：
- **不当类比**：历史类比是否准确？比如"A公司收购B，和C收购D一样"——但如果实际情况有重大差异（如D在被收后仍有重大突破），就应该指出类比不成立
- **遗漏关键玩家**：分析产业格局时是否遗漏了不可忽视的参与者（如谈AI芯片不提英伟达、谈云市场不提阿里云/华为云等）
- **财务逻辑漏洞**：大额承诺是否有可行性分析？钱从哪来？是否存在循环融资嫌疑？

第四步：输出核查结果
完成搜索验证后，输出 JSON：
{{
  "total_claims": 核查的事实条数,
  "verified": 搜索确认正确的条数,
  "unverifiable": 搜索找不到佐证但也没有反证的条数,
  "false_or_fabricated": 搜索结果与声明矛盾、或完全找不到任何相关信息的条数,
  "logic_issues": 逻辑问题的条数,
  "issues": [
    {{
      "claim": "有问题的声明原文",
      "search_result": "你搜到的实际信息是什么",
      "verdict": "false / fabricated / unverifiable / bad_analogy / missing_player / logic_gap",
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
    fixable_issues = [i for i in issues if i.get("verdict") in ("false", "fabricated", "bad_analogy", "missing_player", "logic_gap")]

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
            max_tokens=16384, temperature=0.1,
        )

        if fixed and len(fixed) > len(report_text) * 0.4:
            print(f"  Report corrected: {len(report_text)} → {len(fixed)} chars")
            return fixed, final_result
        else:
            print(f"  Correction failed, keeping original")

    return report_text, final_result


# ============ Stage 4: Format to JSON ============

FORMAT_PROMPT = """把以下深度分析报告转换为 JSON 格式。保留所有内容，只改变格式。

⚠️ 格式化时检查标点：
- 标题和副标题中的引号必须配对（有左引号必须有右引号）
- 中文使用中文标点（""、''），不要混用半角引号

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

    # ====== Stage 3a: Research Collection (chat model + tools) ======
    research_materials, sources = research_collect(client, topic_info, breaking_context)

    # ====== Stage 3b: Report Writing (reasoning model) ======
    report_text = write_report(client, topic_info, brain_dump_text, research_materials)

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
