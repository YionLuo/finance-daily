#!/usr/bin/env python3
"""
Deep Research Report Generator for Y Daily.

Six-stage pipeline with independent fact-checking:

  Stage 1 — Topic Selection: Pick the best deep-dive topic from breaking news
  Stage 2 — Research Planning: Activate domain knowledge + generate explicit key questions
  Stage 3 — Research: Question-guided tool-using research → structured evidence bank
  Stage 4 — Fact Check: Independent verification of claims (fresh context, adversarial)
  Stage 5 — Report Writing: Generate report from verified evidence only (no tools)
  Stage 6 — Format: Convert to structured JSON + validation

Outputs: Updates the `deepResearch` array in index.html
"""

import os
import sys
import json
import re
import time as _time
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

# ============ Constants & Config ============

MAX_RESEARCH_ENTRIES = 30
WEEKDAY_MAP = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# Auto-detect model names based on API endpoint
_base_url = os.environ.get("OPENAI_BASE_URL", "")
_is_openrouter = "openrouter" in _base_url

LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek/deepseek-v4-flash" if _is_openrouter else "deepseek-v4-flash")
WRITER_MODEL = os.environ.get("WRITER_MODEL", LLM_MODEL)  # defaults to flash; set env var for pro

# Tool budgets
MAX_RESEARCH_TOOL_CALLS = 18
MAX_RESEARCH_ROUNDS = 15
MAX_FACT_CHECK_TOOL_CALLS = 10
MAX_FACT_CHECK_ROUNDS = 8


# ============ Stage 1: Topic Selection ============

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
    cleaned = _strip_llm_wrapper(response)

    try:
        topic_info = json.loads(cleaned)
    except json.JSONDecodeError:
        fallback = (breaking_news[0].get("text", "市场动态")[:15] if breaking_news
                    else ai_breaking_news[0].get("text", "AI行业")[:15] if ai_breaking_news
                    else "全球市场综述")
        topic_info = {"topic": fallback, "topicReason": "自动选题", "angle": "综合分析"}

    # Collect seed article URLs for Stage 3
    all_breaking = breaking_news + ai_breaking_news
    seed_urls = []
    for item in all_breaking[:30]:
        url = item.get("url", "")
        if url and "news.google.com" not in url:
            seed_urls.append({"title": item.get("text", "")[:80], "url": url})

    topic_info["seed_urls"] = seed_urls[:8]

    print(f"  Topic: {topic_info.get('topic', '?')}")
    print(f"  Reason: {topic_info.get('topicReason', '')}")
    return topic_info


# ============ Stage 2: Research Planning ============

PLAN_RESEARCH_PROMPT = """你是一位在 AI 和科技金融领域有 15 年经验的资深分析师，服务对象是专业投资人。
当前时间：{current_time}

今天你要深度分析的主题是：「{topic}」
角度：{angle}
选题理由：{reason}

你的任务是产出一份**研究计划**，包含两部分：

═══ 第一部分：结构性认知（Brain Dump）═══
梳理你对这个话题的结构性认知——重点是分析框架和历史规律。

覆盖以下维度：
1. **行业结构与价值链**：竞争格局、关键玩家定位、各环节利润率
   ⚠️ 价值链必须覆盖完整层次——不能只分析"两端"而忽略"中间层"。
   例如：AI产业链不能只看"芯片"和"终端应用"，必须包含"平台/SaaS层"。
2. **历史投资案例**：类似事件驱动了什么股价反应？教科书案例？
3. **二阶/三阶效应**：间接影响和负反馈环路（如：大规模失业→消费下降→科技收入受损）
4. **常见误区**：大众/媒体在这个话题上容易犯什么错？

⚠️ 重要：你的训练数据截止到 2024 年底。不要写具体的产品版本号、融资金额、公司估值——这些会通过搜索获取。
聚焦结构性认知（竞争逻辑、商业模式、历史规律），标注不确定的为 [待验证]。

═══ 第二部分：关键问题清单 ═══
列出这份报告必须回答的 5-8 个关键问题。

问题类型：
- **事实验证型**：需要搜索确认的具体事实（如"X公司最新季度营收是多少？"）
- **财务穿透型**：涉及公司的哪条业务线受影响？影响多少？
- **竞争格局型**：有哪些竞争者/替代方案？各自定位？
- **反面论证型**：谁会反驳这个结论？有什么反面证据？
- **催化剂型**：未来什么事件能验证/推翻核心假设？

每个问题附带搜索建议（具体的搜索关键词）。

═══ 第三部分：分析陷阱提醒 ═══
列出这个话题特有的分析陷阱。
例如：
- "NVIDIA 是 fabless 公司（台积电代工），不要把 FCF 低归因于建厂 CAPEX"
- "同一篇文章的多个数据点不算独立验证"
- "yfinance 返回的是 TTM 数据，可能与最新季报有口径差异"

输出严格 JSON（不要 markdown 代码块）：
{{
  "brain_dump": "结构性认知文本（800-1200字）",
  "key_questions": [
    {{
      "id": "Q1",
      "question": "具体问题",
      "type": "事实验证型/财务穿透型/竞争格局型/反面论证型/催化剂型",
      "priority": "high/medium/low",
      "search_hints": ["搜索关键词1", "搜索关键词2"]
    }}
  ],
  "data_needs": ["需要获取的具体数据1", "需要获取的具体数据2"],
  "pitfalls": ["分析陷阱1", "分析陷阱2"]
}}
"""


def plan_research(client, topic_info):
    """Stage 2: Generate research plan with brain dump + key questions."""
    print("\n=== Stage 2: Research Planning ===")

    prompt = PLAN_RESEARCH_PROMPT.format(
        topic=topic_info.get("topic", ""),
        angle=topic_info.get("angle", ""),
        reason=topic_info.get("topicReason", ""),
        current_time=format_date_cst(now_cst()),
    )

    response = llm_chat_with_retry(
        client, [{"role": "user", "content": prompt}],
        max_tokens=4096, temperature=0.5,
    )

    cleaned = _strip_llm_wrapper(response)

    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError:
        print("  WARNING: Failed to parse research plan JSON, using fallback")
        plan = {
            "brain_dump": response[:1200],
            "key_questions": [
                {"id": "Q1", "question": "这个事件的核心事实是什么？", "type": "事实验证型", "priority": "high", "search_hints": [topic_info.get("topic", "") + " 2026"]},
                {"id": "Q2", "question": "对相关公司的财务影响？", "type": "财务穿透型", "priority": "high", "search_hints": [topic_info.get("topic", "") + " revenue earnings"]},
                {"id": "Q3", "question": "主要风险和反面论据？", "type": "反面论证型", "priority": "high", "search_hints": [topic_info.get("topic", "") + " risks bear case"]},
            ],
            "data_needs": [],
            "pitfalls": [],
        }

    questions = plan.get("key_questions", [])
    print(f"  Brain dump: {len(plan.get('brain_dump', ''))} chars")
    print(f"  Key questions: {len(questions)}")
    for q in questions:
        print(f"    {q.get('id', '?')} [{q.get('type', '?')}] {q.get('question', '')[:60]}")
    print(f"  Pitfalls: {len(plan.get('pitfalls', []))}")

    return plan


# ============ Stage 3: Research ============

RESEARCH_AGENT_PROMPT = """你是 Y Daily 的研究助理。当前时间：{current_time}

今天你要为「{topic}」收集投研级素材。

=== 初始知识框架 ===
{brain_dump}

=== 必须回答的关键问题 ===
{questions_text}

=== 今日相关新闻 ===
{breaking_context}

=== 可预抓取的种子文章 URL ===
{seed_urls_text}

---

你有三个工具：
- **web_search(query)**: 搜索 Google News，返回标题、摘要、URL
- **fetch_url_content(url)**: 读取 URL 全文（最多 10000 字）
- **fetch_financial_data(ticker)**: Yahoo Finance 实时财务数据

---

【数据铁律 — 违反即报废】
- 只使用工具返回的数据。绝不使用训练数据中的具体数字。
- 如果工具没有返回某个数字，不要猜。
- 你的训练数据截止 2024 年底。当前时间见上方。
- 不要写入任何"据报道XX投资了XX亿"这类从记忆中回忆的金额。

---

【研究策略】

你有 {max_tool_calls} 次工具调用预算。按以下优先级分配：

1. **种子文章**（如果有 URL）：先 fetch_url_content 读取 2-3 篇最相关的种子文章
2. **高优先问题**：针对每个 high 优先级问题，做 1-2 次 web_search
3. **财务数据**：涉及具体公司时，调用 fetch_financial_data
4. **一手信源**：至少 1 次搜索 SEC 文件 / 监管机构公告 / 公司 IR 页面
5. **反面观点**：至少 1 次搜索 bear case / risks / criticism
6. **竞争格局**：至少 1 次搜索 competitors / alternatives
7. **中优先问题**：用剩余预算

【分析陷阱提醒】
{pitfalls_text}

研究够了就停止调用工具。系统会自动整理你的研究成果。
"""

RESEARCH_COMPRESS_PROMPT = """请将以上研究过程中获得的所有关键信息整理为结构化研究摘要。

=== 必须回答的问题清单 ===
{questions_text}

=== 整理要求 ===

输出严格 JSON（不要 markdown 代码块）：
{{
  "claims": [
    {{
      "id": "c1",
      "text": "具体的、可验证的事实声明",
      "source_urls": ["https://..."],
      "source_names": ["Reuters", "Bloomberg"],
      "source_count": 1,
      "answers_questions": ["Q1"],
      "category": "fact/metric/analysis/projection/unverified"
    }}
  ],
  "financial_data": {{
    "tickers_fetched": ["NVDA"],
    "raw_data": "fetch_financial_data 返回的原始数据文本"
  }},
  "questions_coverage": {{
    "answered": {{"Q1": ["c1", "c2"]}},
    "unanswered": ["Q4", "Q5"]
  }},
  "source_independence_notes": "哪些来源可能不独立（如同一篇通稿的不同转载）",
  "info_gaps": ["搜不到的关键信息1", "搜不到的关键信息2"]
}}

规则：
1. 每条 claim 必须是具体的、可验证的声明，不是模糊总结
2. source_count 严格按独立来源计数——同一篇文章的多个数据点只算 1 个来源
3. 只保留工具返回的真实数据，绝不补充训练数据中的数字
4. 没搜到的信息放 info_gaps，不要编造
"""


def research(client, topic_info, plan, breaking_context):
    """
    Stage 3: Execute research plan via tool calls → compress to evidence bank.
    Phase A: Question-guided research with tools
    Phase B: Compress raw research to structured evidence
    """
    print(f"\n=== Stage 3: Research (max {MAX_RESEARCH_ROUNDS} rounds, {MAX_RESEARCH_TOOL_CALLS} tool calls) ===")

    raw_client = client  # Use the validated client from main()

    # Format questions for prompt
    questions = plan.get("key_questions", [])
    questions_text = ""
    if questions:
        lines = []
        for q in questions:
            qid = q.get('id', '?')
            qtext = q.get('question', '未知问题')
            hints = ", ".join(q.get("search_hints", [])[:3])
            lines.append(f"  {qid} [{q.get('type', '?')}] (优先级: {q.get('priority', 'medium')}) {qtext}")
            if hints:
                lines.append(f"     搜索建议: {hints}")
        questions_text = "\n".join(lines)
    else:
        questions_text = "（未生成问题清单，请自行判断研究重点）"

    # Format seed URLs
    seed_urls = topic_info.get("seed_urls", [])
    seed_urls_text = "\n".join(f"  - {s['title']}: {s['url']}" for s in seed_urls[:5]) if seed_urls else "（无种子 URL）"

    # Format pitfalls
    pitfalls = plan.get("pitfalls", [])
    pitfalls_text = "\n".join(f"  - {p}" for p in pitfalls) if pitfalls else "（无特殊提醒）"

    system_prompt = RESEARCH_AGENT_PROMPT.format(
        topic=topic_info.get("topic", ""),
        brain_dump=plan.get("brain_dump", "")[:1500],
        questions_text=questions_text,
        breaking_context=breaking_context[:5000],
        seed_urls_text=seed_urls_text,
        pitfalls_text=pitfalls_text,
        current_time=format_date_cst(now_cst()),
        max_tool_calls=MAX_RESEARCH_TOOL_CALLS,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"开始研究「{topic_info.get('topic', '')}」。按优先级使用工具，研究够了就停。"},
    ]

    tool_call_count = 0

    # ===== Phase A: Gather =====
    print("\n  --- Phase A: Gather ---")
    start_time = _time.time()

    for round_num in range(MAX_RESEARCH_ROUNDS):
        if tool_call_count >= MAX_RESEARCH_TOOL_CALLS:
            print(f"  Tool budget exhausted ({tool_call_count}/{MAX_RESEARCH_TOOL_CALLS})")
            break

        print(f"  Round {round_num + 1}/{MAX_RESEARCH_ROUNDS} (tools: {tool_call_count}/{MAX_RESEARCH_TOOL_CALLS})")

        try:
            response = raw_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=4096,
                timeout=180,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            break

        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            processed_ids = set()
            for tool_call in message.tool_calls:
                if tool_call_count >= MAX_RESEARCH_TOOL_CALLS:
                    # Backfill skipped tool calls to keep API happy
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "(工具预算已用完，跳过此调用)",
                    })
                    processed_ids.add(tool_call.id)
                    continue
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, ValueError):
                    func_args = {}
                print(f"    Tool: {func_name}({json.dumps(func_args, ensure_ascii=False)[:100]})")
                result = execute_tool_call(func_name, func_args)
                if len(result) > 10000:
                    result = result[:10000] + "\n...(截断)"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                processed_ids.add(tool_call.id)
                tool_call_count += 1

            # Budget warning
            remaining = MAX_RESEARCH_TOOL_CALLS - tool_call_count
            if 0 < remaining <= 3:
                messages.append({
                    "role": "user",
                    "content": f"你还剩 {remaining} 次工具调用。如果关键信息已够，可以停止。"
                })
        else:
            # Agent stopped calling tools
            print(f"  Agent stopped researching after {tool_call_count} tool calls")
            break

    elapsed = _time.time() - start_time
    print(f"  Phase A complete: {tool_call_count} tool calls, {elapsed:.0f}s")

    # ===== Phase B: Compress =====
    print("\n  --- Phase B: Compress ---")

    compress_prompt = RESEARCH_COMPRESS_PROMPT.format(
        questions_text=questions_text,
    )
    messages.append({"role": "user", "content": compress_prompt})

    evidence_bank = None

    # Attempt 1: Compress within full conversation context
    try:
        compress_resp = raw_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=6000,
            temperature=0.1,
            timeout=180,
        )
        compress_text = _strip_think(compress_resp.choices[0].message.content or "")
        cleaned_json = _strip_llm_wrapper(compress_text)
        evidence_bank = json.loads(cleaned_json)
        print(f"  Compression attempt 1 succeeded")
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Compression attempt 1 failed: {e}")

    # Attempt 2: If first attempt failed, try with a CLEAN context
    # Extract only the tool results (much smaller) and re-ask
    if evidence_bank is None:
        print("  Retrying compression with clean context...")
        tool_results_summary = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                content = msg.get("content", "")[:2000]
                tool_results_summary.append(content)
            elif hasattr(msg, 'content') and hasattr(msg, 'role'):
                # Skip assistant messages to reduce context
                pass

        clean_messages = [
            {"role": "user", "content": (
                f"以下是对「{topic_info.get('topic', '')}」的研究工具调用结果摘要。"
                f"请整理为结构化研究摘要。\n\n"
                f"=== 工具结果 ===\n"
                + "\n---\n".join(tool_results_summary[:15])
                + f"\n\n{compress_prompt}"
            )},
        ]
        try:
            compress_resp2 = raw_client.chat.completions.create(
                model=LLM_MODEL,
                messages=clean_messages,
                max_tokens=6000,
                temperature=0.1,
                timeout=180,
            )
            compress_text2 = _strip_think(compress_resp2.choices[0].message.content or "")
            cleaned_json2 = _strip_llm_wrapper(compress_text2)
            evidence_bank = json.loads(cleaned_json2)
            print(f"  Compression attempt 2 (clean context) succeeded")
        except (json.JSONDecodeError, Exception) as e2:
            print(f"  Compression attempt 2 failed: {e2}")

    # Attempt 3: Last resort — extract claims from raw text manually
    if evidence_bank is None:
        print("  WARNING: All compression attempts failed, extracting claims from raw text")
        # Gather all tool result text
        raw_texts = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                raw_texts.append(msg.get("content", "")[:1500])
        combined_raw = "\n".join(raw_texts)[:8000]

        evidence_bank = {
            "claims": [{"id": f"c{i+1}", "text": chunk.strip()[:500], "source_urls": [], "source_names": [], "source_count": 0, "answers_questions": [], "category": "unverified"} for i, chunk in enumerate(combined_raw.split("\n---\n")[:20]) if chunk.strip()],
            "financial_data": {},
            "questions_coverage": {"answered": {}, "unanswered": [q.get("id", "?") for q in questions]},
            "source_independence_notes": "",
            "info_gaps": ["研究摘要JSON解析失败，使用工具结果原文提取"],
        }

    claims = evidence_bank.get("claims", [])
    gaps = evidence_bank.get("info_gaps", [])
    coverage = evidence_bank.get("questions_coverage", {})
    print(f"  Evidence bank: {len(claims)} claims, {len(gaps)} info gaps")
    print(f"  Questions answered: {list(coverage.get('answered', {}).keys())}")
    print(f"  Questions unanswered: {coverage.get('unanswered', [])}")

    # Abort if zero sources
    if not claims:
        print("  ⚠️ Zero claims extracted — report would be pure hallucination")

    return evidence_bank


# ============ Stage 4: Fact Check ============

FACT_CHECK_PROMPT = """你是一位独立的事实核查编辑。你从未见过产生这些声明的研究过程——你只看到最终的声明列表。你的立场是：**假设每条声明都是错的，直到你验证它。**

你有 web_search 和 fetch_financial_data 工具（最多 {max_tool_calls} 次）。

=== 待核查声明 ===
{claims_text}

=== 财务数据参考 ===
{financial_data_text}

=== 核查流程 ===

1. **优先核查**：先核查具体金额、百分比、估值等数字声明
2. **交叉验证**：对每条关键声明做 1 次 web_search 验证
3. **训练数据泄漏检测**：
   - 如果一个投融资金额很"整"（如 $40B, $100B）且搜索完全没有相关文章 → training_data_leak
   - 如果一个"众所周知"的事件搜索返回 0 结果 → 可能根本没发生 → training_data_leak
   - ⚠️ 重要：如果你搜索后**找到了相关报道**（即使无法确认精确日期），这**不是** training_data_leak。
     Google News RSS 经常不返回精确发表时间，但文章确实存在。
     只有在搜索完全为零结果时才标记 training_data_leak。
   - 同样，如果 claim 中的事件在搜索结果中被多篇文章报道（标题匹配），即使日期不确定，也应标记为 verified 或 partially_correct。
4. **信源独立性审计**：
   - 检查 source_count：如果声明说有 3 个来源，但 URL 指向同一篇通稿的不同转载 → source_independence_error
5. **财务逻辑检查**：
   - fabless 公司（如 NVIDIA, AMD）不应有高 CAPEX → 如果报告把 FCF 低归因于 CAPEX → logic_gap
   - yfinance TTM 数据可能与最新季报不一致 → 标注 stale_data

完成核查后，输出严格 JSON（不要 markdown 代码块）：
{{
  "verdicts": [
    {{
      "claim_id": "c1",
      "verdict": "verified/refuted/training_data_leak/source_independence_error/stale_data/disputed/unverified/partially_correct/bad_analogy/logic_gap",
      "corrected_text": "修正后的文本（如不需修正则为 null）",
      "verification_notes": "简要说明你搜到了什么",
      "usable": true
    }}
  ],
  "data_quality_score": 0.75,
  "flagged_items": [
    {{
      "claim_id": "c1",
      "flag_type": "training_data_leak",
      "details": "具体说明",
      "action": "remove/correct/add_caveat"
    }}
  ]
}}

data_quality_score 计算：verified 占比 × 0.8 + partially_correct 占比 × 0.5。
如果有 training_data_leak，score 减 0.2。如果有 source_independence_error，score 减 0.1。
"""


def fact_check(client, evidence_bank):
    """
    Stage 4: Independent fact-checking of evidence bank claims.
    Fresh context — never sees research agent's reasoning.
    """
    print(f"\n=== Stage 4: Fact Check (max {MAX_FACT_CHECK_ROUNDS} rounds, {MAX_FACT_CHECK_TOOL_CALLS} tool calls) ===")

    raw_client = client  # Use the validated client from main()

    claims = evidence_bank.get("claims", [])
    if not claims:
        print("  No claims to check, skipping")
        return {"verdicts": [], "data_quality_score": 0.0, "flagged_items": []}

    # Format claims for the checker
    claims_lines = []
    for c in claims:
        sources = ", ".join(c.get("source_names", [])[:3]) or "无来源"
        claims_lines.append(f"  [{c.get('id', '?')}] ({c.get('category', '?')}) {c.get('text', '')}")
        claims_lines.append(f"    来源({c.get('source_count', 0)}个): {sources}")
        urls = c.get("source_urls", [])
        if urls:
            claims_lines.append(f"    URL: {urls[0][:80]}")
    claims_text = "\n".join(claims_lines)

    financial_data = evidence_bank.get("financial_data", {})
    financial_data_text = financial_data.get("raw_data", "（无财务数据）")[:3000]

    # Use only web_search and fetch_financial_data for fact-checking (no fetch_url_content — speed)
    fact_check_tools = [t for t in AGENT_TOOLS if t["function"]["name"] in ("web_search", "fetch_financial_data")]

    system_prompt = FACT_CHECK_PROMPT.format(
        claims_text=claims_text,
        financial_data_text=financial_data_text,
        max_tool_calls=MAX_FACT_CHECK_TOOL_CALLS,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "开始核查。优先验证具体金额和数字声明。"},
    ]

    tool_call_count = 0
    final_result = None

    budget_exhausted_sent = False

    for round_num in range(MAX_FACT_CHECK_ROUNDS):
        if tool_call_count >= MAX_FACT_CHECK_TOOL_CALLS and not budget_exhausted_sent:
            messages.append({"role": "user", "content": "工具预算用完。请立即输出最终核查结果 JSON。"})
            budget_exhausted_sent = True

        print(f"  Round {round_num + 1}/{MAX_FACT_CHECK_ROUNDS} (tools: {tool_call_count}/{MAX_FACT_CHECK_TOOL_CALLS})")

        try:
            response = raw_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=fact_check_tools if tool_call_count < MAX_FACT_CHECK_TOOL_CALLS else None,
                temperature=0.2,
                max_tokens=6000,
                timeout=180,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            break

        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call_count >= MAX_FACT_CHECK_TOOL_CALLS:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "(工具预算已用完，跳过此调用)",
                    })
                    continue
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}
                print(f"    Verify: {func_name}({json.dumps(func_args, ensure_ascii=False)[:80]})")
                result = execute_tool_call(func_name, func_args)
                if len(result) > 3000:
                    result = result[:3000] + "\n...(截断)"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                tool_call_count += 1
        else:
            # Parse final output
            text = _strip_think(message.content or "")
            cleaned = _strip_llm_wrapper(text)
            try:
                final_result = json.loads(cleaned)
            except json.JSONDecodeError:
                json_match = re.search(r'\{[\s\S]*\}', cleaned)
                if json_match:
                    try:
                        final_result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
            break

    if not final_result:
        print("  ⚠️ Fact check did not produce parseable result, marking all unverified")
        final_result = {
            "verdicts": [{"claim_id": c.get("id", "?"), "verdict": "unverified", "corrected_text": None, "verification_notes": "核查超时", "usable": True} for c in claims],
            "data_quality_score": 0.3,
            "flagged_items": [],
        }

    # Print summary
    verdicts = final_result.get("verdicts", [])
    score = final_result.get("data_quality_score", 0)
    flagged = final_result.get("flagged_items", [])

    verdict_counts = {}
    for v in verdicts:
        vtype = v.get("verdict", "unknown")
        verdict_counts[vtype] = verdict_counts.get(vtype, 0) + 1

    print(f"\n  Results: {len(verdicts)} verdicts, quality score: {score:.2f}")
    for vtype, count in sorted(verdict_counts.items()):
        icon = {"verified": "✅", "refuted": "❌", "training_data_leak": "🚫", "unverified": "❓"}.get(vtype, "⚠️")
        print(f"    {icon} {vtype}: {count}")
    if flagged:
        print(f"  Flagged items: {len(flagged)}")
        for f in flagged[:5]:
            print(f"    🚩 [{f.get('flag_type', '?')}] {f.get('details', '')[:80]}")

    return final_result


# ============ Stage 5: Report Writing ============

REPORT_WRITER_PROMPT = """你是 Y Daily 的首席分析师。当前时间：{current_time}
你已经完成了对「{topic}」的研究和事实核查。请基于以下**已验证的证据**写一份深度投研分析报告。

=== 分析框架 ===
{brain_dump}

=== 关键问题 ===
{questions_text}

=== 已验证的证据（只能使用这些） ===
{verified_claims_text}

=== 财务数据 ===
{financial_data_text}

=== 未能回答的问题 ===
{unanswered_text}

=== 信息空白 ===
{info_gaps_text}

=== 核查摘要 ===
数据质量评分: {quality_score}/1.0
{flagged_text}

---

【关键约束 — 每一条都必须遵守】

1. **数据纪律**：只使用"已验证的证据"中的数据。不在证据中的数字/事实一律不写。
   不确定就写"（未获取到具体数据）"。绝不使用训练数据中的数字。

2. **信源多元化铁律**：
   - ≥2 独立来源 = 【已验证】
   - 1 个来源 = 【部分验证】（即使该来源内部自洽）
   - 0 个来源 = 【未验证-推测】
   ⚠️ 同一篇文章的多个数据点只算 1 个来源。

3. **多空排版**：每个分析章节内，多方和空方观点用 ### 子标题分开。
   每个列表项独占一行，前后有空行：

   ### 多方观点

   - **论点A**：证据。【已验证】来源：URL

   - **论点B**：证据。【部分验证】来源：URL

   ### 空方观点

   - **论点C**：证据。来源：URL

4. **评级-信源一致性铁律**：
   🟢 高确定性：≥2 独立来源硬数据，无强反面证据
   🟡 中等确定性：1 来源 / 数据不完整 / 存在可信反面论点
   🔴 低确定性：纯推测 / 正反证据强度接近 / 关键变量未知
   - 1 个来源 → 最高 🟡
   - 未验证假设 → 只能 🔴
   - "需更多数据验证" → 不能给 🟢
   - 评级理由必须引用具体来源数量

5. **催化剂日历**：时间具体到"周"或更精确。禁止"下半年"、"近期"。
   | 时间 | 事件 | 验证/推翻假设 | 影响方向 |
   ⚠️ 至少 2 条必须有具体日期（从已验证证据中提取财报日、会议日等）。
   "时间不确定"最多出现 1 次。如果大部分催化剂都无法确定时间，说明研究不够充分——在表前注明"⚠️ 催化剂时间精度不足，需补充研究"。

6. **交易建议**：不写"买入/卖出/做多/做空"。
   用情景假设："若 [条件] 被验证，则 [标的] 面临 [方向] 压力/机会，关键信号是 [X]。"
   提及配对交易必须给出具体 ETF/个股代码，或明确写"未找到合适标的"。

7. **竞争格局**：如果证据中有竞争者信息，必须做对比。否则标注为信息空白。

8. **关键洞察展开**：有价值的宏观洞察（负反馈环路、结构性矛盾）必须用独立段落展开，
   列出环路节点、搜索历史弹性系数。无法量化则标注"（无历史数据支撑量化）"。

9. **未回答问题诚实说**：对 unanswered 的问题，在相关章节明确写"当前证据不足以回答此问题"。

---

【报告结构】（严格遵守）

# [标题]

## 核心问题
（1-2 句话）

## 关键假设与验证状态
| 假设 | 验证状态 | 证据来源 | 独立来源数 |

## [分析章节 1]
### 多方观点
### 空方观点

## [分析章节 2]
### 多方观点
### 空方观点

## 我们的判断
（每个分判断独立一行 + 🟢🟡🔴 + 理由 + 来源数量）

## 催化剂日历与待验证信号
（Markdown 表格，时间排序）

---
写作风格：Stratechery 结构感 + SemiAnalysis 数据穿透。克制。不确定就说不确定。
禁止废话："短期/中期/长期"、"存在不确定性"、"需要密切关注"。
目标：3500-5000 字。以 # 标题行开头，直接输出。
"""


def write_report(client, topic_info, plan, evidence_bank, fact_check_result):
    """
    Stage 5: Write report from verified evidence only. No tools.
    """
    print(f"\n=== Stage 5: Report Writing (model: {WRITER_MODEL}) ===")

    raw_client = client  # Use the validated client from main()

    # Build verified claims text
    verdicts = {v.get("claim_id"): v for v in fact_check_result.get("verdicts", [])}
    claims = evidence_bank.get("claims", [])

    verified_lines = []
    for c in claims:
        cid = c.get("id", "")
        verdict = verdicts.get(cid, {})
        usable = verdict.get("usable", True)
        if not usable:
            continue  # Skip refuted/leaked claims

        vtype = verdict.get("verdict", "unverified")
        text = verdict.get("corrected_text") or c.get("text", "")
        sources = ", ".join(c.get("source_names", [])[:3]) or "无来源"
        src_count = c.get("source_count", 0)
        icon = {"verified": "✅", "partially_correct": "⚡", "unverified": "❓"}.get(vtype, "⚠️")

        verified_lines.append(f"  {icon} [{cid}] {text}")
        verified_lines.append(f"    验证: {vtype} | 独立来源: {src_count} | 来源: {sources}")

    verified_claims_text = "\n".join(verified_lines) if verified_lines else "（无已验证证据）"

    if not verified_lines:
        print("  ⚠️ Zero usable claims — report will be based on minimal evidence")

    # Financial data
    fin_data = evidence_bank.get("financial_data", {})
    financial_data_text = fin_data.get("raw_data", "（无财务数据）")[:3000]

    # Questions
    questions = plan.get("key_questions", [])
    questions_text = "\n".join(f"  {q.get('id', '?')}. {q.get('question', '?')}" for q in questions) if questions else "（无问题清单）"

    coverage = evidence_bank.get("questions_coverage", {})
    unanswered = coverage.get("unanswered", [])
    unanswered_text = "\n".join(f"  - {qid}" for qid in unanswered) if unanswered else "（所有问题均已覆盖）"

    info_gaps = evidence_bank.get("info_gaps", [])
    info_gaps_text = "\n".join(f"  - {g}" for g in info_gaps) if info_gaps else "（无信息空白）"

    # Flagged items
    flagged = fact_check_result.get("flagged_items", [])
    flagged_text = "\n".join(f"  🚩 [{f.get('flag_type', '?')}] {f.get('details', '')}" for f in flagged) if flagged else "（无特殊标记）"

    quality_score = fact_check_result.get("data_quality_score", 0.5)

    prompt = REPORT_WRITER_PROMPT.format(
        topic=topic_info.get("topic", ""),
        brain_dump=plan.get("brain_dump", "")[:1200],
        questions_text=questions_text,
        verified_claims_text=verified_claims_text[:15000],
        financial_data_text=financial_data_text,
        unanswered_text=unanswered_text,
        info_gaps_text=info_gaps_text,
        quality_score=f"{quality_score:.2f}",
        flagged_text=flagged_text,
        current_time=format_date_cst(now_cst()),
    )

    # Try WRITER_MODEL first, fallback to LLM_MODEL if different
    models_to_try = [WRITER_MODEL]
    if LLM_MODEL != WRITER_MODEL:
        models_to_try.append(LLM_MODEL)

    for model in models_to_try:
        try:
            response = raw_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=0.5,
            )
            text = _strip_think(response.choices[0].message.content or "")
            if text and len(text) > 500:
                print(f"  Report written with {model}: {len(text)} chars")
                return text
            print(f"  {model} returned short output ({len(text)} chars), trying fallback")
        except Exception as e:
            print(f"  {model} failed: {e}")
            if model == LLM_MODEL:
                return f"# 报告生成失败\n\n错误：{e}\n\n话题：{topic_info.get('topic', '')}"
            continue

    return f"# 报告生成失败\n\n所有模型均失败。话题：{topic_info.get('topic', '')}"


# ============ Stage 6: Format to JSON ============

FORMAT_PROMPT = """把以下深度分析报告转换为 JSON 格式。保留所有内容，只改变格式。

⚠️ 格式化规则：
- 标题和副标题中的引号必须配对
- 中文使用中文标点（""、''），不要混用半角引号
- sections content 中的 HTML 必须完整闭合
- 输出必须是合法 JSON，不要输出 markdown 代码块标记

⚠️ keyTakeaways 禁止使用"建议买入/卖出/做多/做空/加仓/减仓"等任何交易指令。
改用情景假设句式："若[条件]被验证，则[标的]面临[方向]压力/机会，关键信号是[X]，确定性：🟡"。

=== 原始报告 ===
{report_text}

=== 输出 JSON ===
严格输出以下 JSON 结构（不要 markdown 代码块、不要任何前缀文字）：
{{
  "title": "报告标题",
  "subtitle": "副标题",
  "summary": "200字以内摘要",
  "tags": [
    {{"text": "标签名", "type": "up|down|warn"}}
  ],
  "keyTakeaways": ["核心判断1（情景假设句式）", "核心判断2", "核心判断3"],
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
sections content 用 HTML：<p>段落、<strong>加粗、<ul><li>列表、<table>表格、<h3>子标题。
"""


def format_to_json(client, report_text):
    """Stage 6: Convert plain-text analysis into structured JSON + validate."""
    print("\n=== Stage 6: Format to JSON ===")
    print(f"  Report text length: {len(report_text)} chars")

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
            sections = result.get("sections", [])
            if sections and len(sections) >= 2:
                print(f"  ✅ JSON parsed: {len(sections)} sections")
                # Run validators
                _validate_report(result)
                return result
            else:
                print(f"  ⚠️ Only {len(sections)} section(s), retrying...")
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse failed: {e}")

    # ====== Fallback: Parse Markdown directly ======
    print("  ⚠️ LLM formatting failed, using Markdown fallback parser")
    return _fallback_md_to_json(report_text)


def _validate_report(result):
    """Run deterministic validation checks on the formatted report."""
    warnings = []
    sections = result.get("sections", [])

    # Check for crammed bull/bear lists
    for sec in sections:
        content = sec.get("content", "")
        # Count list items per <ul> block
        for ul_match in re.finditer(r'<ul>(.*?)</ul>', content, re.DOTALL):
            items = re.findall(r'<li>', ul_match.group(1))
            if len(items) > 6:
                warnings.append(f"  ⚠️ Section '{sec.get('title', '?')}': {len(items)} list items in one block (check formatting)")

    # Check for vague catalyst dates
    for sec in sections:
        content = sec.get("content", "")
        if "催化剂" in sec.get("title", "") or "催化剂" in content[:50]:
            vague_patterns = ["下半年", "中长期", "未来几个月", "近期"]
            for pat in vague_patterns:
                if pat in content:
                    warnings.append(f"  ⚠️ Vague catalyst date: '{pat}' found")

    # Check confidence rating consistency
    for sec in sections:
        content = sec.get("content", "")
        if "🟢" in content and ("单一来源" in content or "部分验证" in content):
            warnings.append(f"  ⚠️ Possible rating-source inconsistency in '{sec.get('title', '?')}'")

    for w in warnings:
        print(w)


def _fallback_md_to_json(report_text):
    """Fallback: parse Markdown report directly into JSON structure."""
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

    full_text = '\n'.join(all_content_text)
    summary_text = re.sub(r'[#*\-|]', '', full_text[:300]).strip()
    summary_text = re.sub(r'\s+', ' ', summary_text)[:200]

    key_takeaways = []
    for sec_title, sec_content in md_sections:
        if any(kw in sec_title for kw in ['判断', '摘要', 'Takeaway']):
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

    print(f"  Fallback: {len(sections)} sections, {len(tags)} tags")
    return result


# ============ Helpers ============

def _strip_llm_wrapper(text):
    """Strip markdown code blocks and <think> tags from LLM output, extract JSON."""
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE).strip()
    cleaned = _strip_think(cleaned)
    # Try to find JSON boundaries with string-aware brace matching
    first_brace = cleaned.find('{')
    if first_brace > 0:
        cleaned = cleaned[first_brace:]
    elif first_brace < 0:
        return cleaned
    depth = 0
    last_brace = -1
    i = 0
    while i < len(cleaned):
        c = cleaned[i]
        if c == '"':
            # Skip string contents (handles escaped quotes)
            i += 1
            while i < len(cleaned):
                if cleaned[i] == '\\':
                    i += 2
                    continue
                if cleaned[i] == '"':
                    break
                i += 1
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                last_brace = i
                break
        i += 1
    if last_brace > 0:
        cleaned = cleaned[:last_brace + 1]
    return cleaned


def _strip_think(text):
    """Remove <think>...</think> blocks from DeepSeek R1 output."""
    think_end = text.find("</think>")
    if think_end != -1:
        return text[think_end + len("</think>"):].strip()
    # Strip unclosed <think> block (truncated output)
    think_start = text.find("<think>")
    if think_start != -1:
        before = text[:think_start].strip()
        if before:
            return before  # JSON came before <think>
        return ""  # Only <think> with no closing — nothing usable
    return text


def _clean_llm_json(response):
    """Clean LLM response and extract JSON (alias for format_to_json compatibility)."""
    return _strip_llm_wrapper(response)


def _md_to_html(text):
    """Basic Markdown to HTML conversion for fallback."""
    lines = text.split('\n')
    html_parts = []
    in_list = False
    list_type = 'ul'  # track whether we're in <ul> or <ol>
    in_table = False
    table_rows = []

    def _close_list():
        nonlocal in_list, list_type
        if in_list:
            html_parts.append(f'</{list_type}>')
            in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            _close_list()
            if in_table and table_rows:
                html_parts.append(_build_table(table_rows))
                table_rows = []
                in_table = False
            html_parts.append('')
            continue

        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            table_rows.append(stripped)
            continue
        elif in_table and table_rows:
            html_parts.append(_build_table(table_rows))
            table_rows = []
            in_table = False

        if stripped.startswith('### '):
            _close_list()
            html_parts.append(f'<h3>{_inline_md(stripped[4:])}</h3>')
        elif stripped.startswith('## '):
            _close_list()
            html_parts.append(f'<h3>{_inline_md(stripped[3:])}</h3>')
        elif stripped.startswith('# '):
            _close_list()
            continue
        elif stripped.startswith('- ') or stripped.startswith('* '):
            if in_list and list_type != 'ul':
                _close_list()
            if not in_list:
                html_parts.append('<ul>')
                in_list = True
                list_type = 'ul'
            html_parts.append(f'<li>{_inline_md(stripped[2:])}</li>')
        elif re.match(r'^\d+\.\s', stripped):
            content = re.sub(r'^\d+\.\s', '', stripped)
            if in_list and list_type != 'ol':
                _close_list()
            if not in_list:
                html_parts.append('<ol>')
                in_list = True
                list_type = 'ol'
            html_parts.append(f'<li>{_inline_md(content)}</li>')
        elif stripped == '---' or stripped == '***':
            continue
        else:
            _close_list()
            html_parts.append(f'<p>{_inline_md(stripped)}</p>')

    _close_list()
    if in_table and table_rows:
        html_parts.append(_build_table(table_rows))

    return '\n'.join(html_parts)


def _inline_md(text):
    """Convert inline markdown to HTML."""
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
    for row in rows[2:]:
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
    """Split markdown text into sections by ## headers."""
    lines = text.split('\n')
    sections = []
    current_title = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        is_section = False
        if re.match(r'^#{1,2}\s+(?:\d+[\.\、]?\s*)?', stripped):
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

    if not sections:
        sections = [("分析报告", text)]

    return sections


def _extract_title_from_md(text):
    """Extract the first # heading as title."""
    for line in text.split('\n')[:10]:
        stripped = line.strip()
        if stripped.startswith('# ') and not stripped.startswith('## '):
            return stripped[2:].strip()
    return None


def _extract_tickers_from_text(text):
    """Extract stock ticker symbols from text."""
    tickers = set()
    for m in re.finditer(r'\(([A-Z]{1,5})\)', text):
        t = m.group(1)
        if len(t) >= 2 and t not in {'AI', 'US', 'EU', 'UK', 'HK', 'CN', 'GDP', 'CPI', 'PPI',
                                       'IEA', 'IPO', 'ETF', 'CEO', 'CFO', 'CTO', 'PPA', 'IRR',
                                       'THE', 'FOR', 'AND', 'BUT', 'NOT', 'ARE', 'WAS', 'HAS',
                                       'PCB', 'UAE', 'IMF', 'SPR', 'LNG', 'PPA', 'TCO', 'API'}:
            tickers.add(t)
    return list(tickers)[:15]


def _extract_tags_from_text(text, tickers):
    """Extract tags from report text."""
    tags = []
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

    seen = set()
    unique_tags = []
    for tag in tags:
        key = tag["text"]
        if key not in seen:
            seen.add(key)
            unique_tags.append(tag)

    if len(unique_tags) < 3 and tickers:
        for t in tickers[:3]:
            if t not in seen:
                unique_tags.append({"text": t, "type": "warn"})
                seen.add(t)

    return unique_tags[:8]


# ============ Main ============

def main():
    print("=" * 60)
    print("Y Daily — Deep Research Report Generator (6-Stage Pipeline)")
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
        print("\nABORT: No breaking news available. No report today.")
        sys.exit(0)

    # Create LLM client
    client = create_llm_client(required=True)

    # Get recent topics for dedup
    existing_topics = [r.get("topic", "") for r in deep_research[:7] if r.get("topic")]

    # ====== Stage 1: Topic Selection ======
    topic_info = select_topic(client, breaking_news, ai_breaking_news, existing_topics)

    # ====== Stage 2: Research Planning ======
    plan = plan_research(client, topic_info)

    # Build breaking news context for Stage 3
    breaking_lines = []
    for item in (breaking_news + ai_breaking_news)[:30]:
        breaking_lines.append(f"[{item.get('time', '')}] {item.get('text', '')}")
    breaking_context = "\n".join(breaking_lines)

    # ====== Stage 3: Research ======
    evidence_bank = research(client, topic_info, plan, breaking_context)

    # Check if we have enough to continue
    claims = evidence_bank.get("claims", [])
    if not claims:
        print("\nABORT: Zero claims from research. Refusing to hallucinate a report.")
        sys.exit(0)

    # ====== Stage 4: Fact Check ======
    try:
        fact_check_result = fact_check(client, evidence_bank)
    except Exception as e:
        print(f"\n⚠️ Fact check failed ({e}), proceeding with unverified evidence")
        fact_check_result = {
            "verdicts": [{"claim_id": c.get("id", "?"), "verdict": "unverified", "corrected_text": None, "verification_notes": "核查失败", "usable": True} for c in claims],
            "data_quality_score": 0.3,
            "flagged_items": [{"claim_id": "", "flag_type": "warning", "details": f"事实核查失败: {e}", "action": "add_caveat"}],
        }

    # Check quality score and usable claims
    quality_score = fact_check_result.get("data_quality_score", 0)
    verdicts = fact_check_result.get("verdicts", [])
    usable_count = sum(1 for v in verdicts if v.get("usable", True))

    if usable_count == 0 and len(verdicts) > 0:
        # All claims marked unusable — but pipeline should still produce a report
        # with honest "no verified evidence" framing, rather than aborting entirely.
        # Mark claims as usable=True but keep their verdicts (unverified/training_data_leak)
        # so the writer can see the verification status and be appropriately cautious.
        print(f"  ⚠️ All {len(verdicts)} claims marked unusable — overriding to allow cautious report")
        for v in verdicts:
            v["usable"] = True
        fact_check_result["data_quality_score"] = max(quality_score, 0.15)
        quality_score = fact_check_result["data_quality_score"]

    # ====== Stage 5: Write Report ======
    report_text = write_report(client, topic_info, plan, evidence_bank, fact_check_result)

    # ====== Stage 6: Format to JSON ======
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
        "sources": report.get("sources", report.get("sourceIndices", [])),
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
    print(f"  Data quality: {quality_score:.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
