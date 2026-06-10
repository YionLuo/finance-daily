#!/usr/bin/env python3
"""
Deep Research Report Generator for Y Daily.

Six-stage investment memo pipeline with independent fact-checking:

  Stage 1 — Editor Gate: Score candidates and skip low-value days
  Stage 2 — Research Contract: Define thesis, anti-thesis, evidence needs, stop rules
  Stage 3 — Research: Contract-guided tool research → evidence cards
  Stage 4 — Fact Check: Independent verification of claims (fresh context, adversarial)
  Stage 5 — Memo Writing: Financial mapping + red team + investment judgment
  Stage 6 — Format: Convert memo to structured JSON + validation

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
    python_to_js_array, DEFAULT_LLM_MODEL,
)
from news_fetcher import (
    fetch_topic_articles, articles_to_context,
    AGENT_TOOLS, execute_tool_call,
)

# ============ Constants & Config ============

MAX_RESEARCH_ENTRIES = 30
WEEKDAY_MAP = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

LLM_MODEL = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
WRITER_MODEL = os.environ.get("WRITER_MODEL", LLM_MODEL)

# Tool budgets
MAX_RESEARCH_TOOL_CALLS = 18
MAX_RESEARCH_ROUNDS = 15
MAX_FACT_CHECK_TOOL_CALLS = 10
MAX_FACT_CHECK_ROUNDS = 8

WATCHLIST_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "skill-金融资讯日报",
    "references",
    "watchlist.md",
)

# Quality gates. A run exits cleanly without writing a report when these are not met.
MIN_TOPIC_SCORE = 0.62
MIN_DATA_QUALITY_SCORE = 0.35
MIN_USABLE_CLAIMS = 3
MIN_ANSWERED_QUESTIONS = 2
MIN_QUESTION_COVERAGE = 0.35


def load_watchlist_context(max_chars=5000):
    """Load the user's watchlist and focus areas for topic scoring and mapping."""
    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            return f.read()[:max_chars]
    except OSError:
        return "（未找到股票池配置，按通用 AI/科技/金融市场主题评估）"


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _topic_total_score(topic_info):
    scores = topic_info.get("scores", {})
    if isinstance(scores, dict):
        return _as_float(scores.get("total"), _as_float(topic_info.get("totalScore"), 0.0))
    return _as_float(topic_info.get("totalScore"), 0.0)


def _should_skip_topic(topic_info):
    """Return (should_skip, reason) for the editor gate."""
    if topic_info.get("shouldSkip"):
        return True, topic_info.get("skipReason") or "Editor 判定今日没有足够投资研究价值"

    total = _topic_total_score(topic_info)
    if total < MIN_TOPIC_SCORE:
        return True, f"选题总分 {total:.2f} 低于门槛 {MIN_TOPIC_SCORE:.2f}"

    scores = topic_info.get("scores", {})
    if isinstance(scores, dict):
        repetition = _as_float(scores.get("repetitionPenalty"), 0.0)
        if repetition >= 0.8:
            return True, "近期重复度过高，跳过"

    return False, ""


def _research_coverage_stats(plan, evidence_bank):
    """Return question coverage stats used by the post-research quality gate."""
    questions = plan.get("key_questions", [])
    total_questions = len(questions)
    coverage = evidence_bank.get("questions_coverage", {})
    answered = coverage.get("answered", {}) if isinstance(coverage, dict) else {}
    answered_ids = [qid for qid, claim_ids in answered.items() if claim_ids]
    answered_count = len(set(answered_ids))
    coverage_ratio = answered_count / total_questions if total_questions else 1.0
    return {
        "total_questions": total_questions,
        "answered_count": answered_count,
        "coverage_ratio": coverage_ratio,
    }


def _should_skip_after_research(plan, evidence_bank):
    """Return (should_skip, reason) when research coverage is too thin for a memo."""
    stats = _research_coverage_stats(plan, evidence_bank)
    total = stats["total_questions"]
    answered = stats["answered_count"]
    ratio = stats["coverage_ratio"]

    if total == 0:
        return True, "研究计划没有关键问题，跳过以避免无约束写作"
    if answered < MIN_ANSWERED_QUESTIONS:
        return True, f"关键问题覆盖不足：仅回答 {answered}/{total}，至少需要 {MIN_ANSWERED_QUESTIONS} 个"
    if ratio < MIN_QUESTION_COVERAGE:
        return True, f"关键问题覆盖率 {ratio:.0%} 低于门槛 {MIN_QUESTION_COVERAGE:.0%}"
    return False, ""


# ============ Stage 1: Topic Selection ============

TOPIC_SELECTION_PROMPT = """你是 Y Daily 的 Editor，不是日报写手。你的职责是判断今天是否存在值得写成投资判断 memo 的研究机会。

目标读者：关注美股/港股 AI、科技、互联网、半导体、宏观流动性的专业投资人。
核心原则：宁缺毋滥。若今天没有足够强的投资判断增量，应该跳过，不要硬写长文。

用户股票池与关注领域：
{watchlist_context}

近期报告主题（用于去重）：
{existing_topics}

今日金融快讯：
{finance_news}

今日 AI/科技快讯：
{ai_news}

请先生成 2-5 个候选 TopicCandidate，再逐项评分，最后选择一个或决定跳过。

评分维度（0-1 分）：
- tickerRelevance：是否能直接映射到用户股票池/关注资产的业务线、估值变量或风险偏好。
- informationDelta：今天的信息是否改变原有判断，而不是普通新闻复述。
- evidenceVerifiability：是否可能用公开报道、公司 IR、监管文件、财务数据验证。
- catalystClarity：未来 1-8 周是否有可观察的验证/推翻信号。
- bearCaseStrength：是否存在足够强的反方论证，避免单边叙事。
- repetitionPenalty：近期重复度，0=完全不重复，1=高度重复。

total 评分公式：
0.25*tickerRelevance + 0.25*informationDelta + 0.20*evidenceVerifiability + 0.15*catalystClarity + 0.10*bearCaseStrength - 0.15*repetitionPenalty

跳过规则：
- selected.total < 0.62，跳过
- 与近期报告主题高度重复，跳过
- 无法回答“读者读完应改变什么投资判断”，跳过
- 只是市场行情、产品发布、融资传闻、媒体观点复述，跳过

输出严格 JSON（不要 markdown 代码块）：
{{
  "shouldSkip": false,
  "skipReason": null,
  "candidates": [
    {{
      "topic": "专题名称（15字以内）",
      "topicReason": "为什么它可能值得写（2句话以内）",
      "angle": "投资判断切入角度",
      "whyNow": "为什么今天必须写",
      "beliefUpdate": "读者读完后应该更新的判断",
      "linkedTickers": ["NVDA", "MSFT"],
      "scores": {{
        "tickerRelevance": 0.8,
        "informationDelta": 0.7,
        "evidenceVerifiability": 0.8,
        "catalystClarity": 0.6,
        "bearCaseStrength": 0.6,
        "repetitionPenalty": 0.0,
        "total": 0.71
      }},
      "skipReason": null
    }}
  ],
  "selected": {{
    "topic": "专题名称（15字以内）",
    "topicReason": "最终选择理由",
    "angle": "投资判断切入角度",
    "whyNow": "为什么今天必须写",
    "beliefUpdate": "读者读完后应该更新的判断",
    "linkedTickers": ["NVDA", "MSFT"],
    "scores": {{
      "tickerRelevance": 0.8,
      "informationDelta": 0.7,
      "evidenceVerifiability": 0.8,
      "catalystClarity": 0.6,
      "bearCaseStrength": 0.6,
      "repetitionPenalty": 0.0,
      "total": 0.71
    }},
    "skipReason": null
  }}
}}
"""


def select_topic(client, breaking_news, ai_breaking_news, existing_topics, watchlist_context):
    """Stage 1: Editor gate, candidate scoring, and optional skip."""
    print("\n=== Stage 1: Topic Selection ===")

    topics_str = "\n".join(f"- {t}" for t in existing_topics) if existing_topics else "（无历史报告）"

    finance_lines = []
    for item in breaking_news[:20]:
        finance_lines.append(f"[{item.get('time', '')}] [{item.get('tagText', '')}] {item.get('text', '')}")

    ai_lines = []
    for item in ai_breaking_news[:20]:
        ai_lines.append(f"[{item.get('time', '')}] [{item.get('tagText', '')}] {item.get('text', '')}")

    prompt = TOPIC_SELECTION_PROMPT.format(
        watchlist_context=watchlist_context,
        existing_topics=topics_str,
        finance_news="\n".join(finance_lines) or "（无金融快讯）",
        ai_news="\n".join(ai_lines) or "（无AI快讯）",
    )

    response = llm_chat_with_retry(client, [{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.3)
    cleaned = _strip_llm_wrapper(response)

    try:
        selection = json.loads(cleaned)
    except json.JSONDecodeError:
        print("  WARNING: Failed to parse topic selection JSON, skipping to avoid low-value report")
        selection = {
            "shouldSkip": True,
            "skipReason": "选题评分 JSON 解析失败",
            "candidates": [],
            "selected": {
                "topic": "无合格选题",
                "topicReason": "自动跳过",
                "angle": "",
                "whyNow": "",
                "beliefUpdate": "",
                "linkedTickers": [],
                "scores": {"total": 0.0},
                "skipReason": "选题评分 JSON 解析失败",
            },
        }

    selected = selection.get("selected") or {}
    if not selected and "topic" in selection:
        selected = selection

    topic_info = dict(selected)
    topic_info["shouldSkip"] = bool(selection.get("shouldSkip", selected.get("shouldSkip", False)))
    topic_info["skipReason"] = selection.get("skipReason") or selected.get("skipReason")
    topic_info["candidates"] = selection.get("candidates", [])
    if not isinstance(topic_info.get("linkedTickers"), list):
        topic_info["linkedTickers"] = []
    if not isinstance(topic_info.get("scores"), dict):
        topic_info["scores"] = {"total": _as_float(topic_info.get("totalScore"), 0.0)}
    topic_info["totalScore"] = _topic_total_score(topic_info)

    # Collect seed article URLs for Stage 3
    all_breaking = breaking_news + ai_breaking_news
    seed_urls = []
    for item in all_breaking[:30]:
        url = item.get("url", "")
        if url and "news.google.com" not in url:
            seed_urls.append({"title": item.get("text", "")[:80], "url": url})

    topic_info["seed_urls"] = seed_urls[:8]

    print(f"  Topic: {topic_info.get('topic', '?')}")
    print(f"  Score: {topic_info.get('totalScore', 0):.2f}")
    print(f"  Linked tickers: {', '.join(topic_info.get('linkedTickers', [])) or 'none'}")
    print(f"  Reason: {topic_info.get('topicReason', '')}")
    if topic_info.get("shouldSkip") or topic_info.get("skipReason"):
        print(f"  Skip reason: {topic_info.get('skipReason', '')}")
    return topic_info


# ============ Stage 2: Research Planning ============

PLAN_RESEARCH_PROMPT = """你是一位在 AI 和科技金融领域有 15 年经验的资深研究负责人，服务对象是专业投资人。
当前时间：{current_time}

今天你要深度分析的主题是：「{topic}」
角度：{angle}
选题理由：{reason}
为什么今天写：{why_now}
预期判断更新：{belief_update}
关联标的：{linked_tickers}

用户股票池与关注领域：
{watchlist_context}

你的任务是产出一份**Research Contract + 研究计划**。Research Contract 是后续 agent 的约束：如果无法满足，宁可跳过，也不要写一篇貌似完整但没有判断增量的报告。

═══ 第一部分：Research Contract ═══
必须明确：
1. **coreQuestion**：这篇 memo 要回答的唯一核心投资问题。
2. **thesis**：当前最值得检验的正向假设。
3. **antiThesis**：最强反向假设，不要稻草人。
4. **beliefUpdate**：读者读完后应更新的判断。
5. **requiredEvidence**：必须拿到哪些证据才值得写。
6. **stopConditions**：出现哪些情况就应该停止写作或降级为低置信度。

═══ 第二部分：结构性认知（Brain Dump）═══
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

═══ 第三部分：关键问题清单 ═══
列出这份报告必须回答的 5-8 个关键问题。

问题类型：
- **事实验证型**：需要搜索确认的具体事实（如"X公司最新季度营收是多少？"）
- **财务穿透型**：涉及公司的哪条业务线、财务变量、时间窗口受影响？
- **竞争格局型**：有哪些竞争者/替代方案？各自定位？
- **反面论证型**：谁会反驳这个结论？有什么反面证据？
- **催化剂型**：未来什么事件能验证/推翻核心假设？

每个问题附带搜索建议（具体的搜索关键词）。

═══ 第四部分：分析陷阱提醒 ═══
列出这个话题特有的分析陷阱。
例如：
- "NVIDIA 是 fabless 公司（台积电代工），不要把 FCF 低归因于建厂 CAPEX"
- "同一篇文章的多个数据点不算独立验证"
- "yfinance 返回的是 TTM 数据，可能与最新季报有口径差异"

输出严格 JSON（不要 markdown 代码块）：
{{
  "research_contract": {{
    "coreQuestion": "唯一核心投资问题",
    "thesis": "正向假设",
    "antiThesis": "最强反向假设",
    "beliefUpdate": "读者应更新的判断",
    "requiredEvidence": ["必须拿到的证据1", "必须拿到的证据2"],
    "stopConditions": ["停止/跳过条件1", "停止/跳过条件2"]
  }},
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


def plan_research(client, topic_info, watchlist_context):
    """Stage 2: Generate Research Contract, brain dump, and key questions."""
    print("\n=== Stage 2: Research Planning ===")

    prompt = PLAN_RESEARCH_PROMPT.format(
        topic=topic_info.get("topic", ""),
        angle=topic_info.get("angle", ""),
        reason=topic_info.get("topicReason", ""),
        why_now=topic_info.get("whyNow", ""),
        belief_update=topic_info.get("beliefUpdate", ""),
        linked_tickers=", ".join(topic_info.get("linkedTickers", [])) or "（无直接标的）",
        watchlist_context=watchlist_context[:4000],
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
            "research_contract": {
                "coreQuestion": f"{topic_info.get('topic', '')} 是否改变用户股票池的投资判断？",
                "thesis": topic_info.get("beliefUpdate", "") or "该事件可能改变相关标的的风险收益结构。",
                "antiThesis": "该事件只是短期新闻，不改变基本面或估值变量。",
                "beliefUpdate": topic_info.get("beliefUpdate", ""),
                "requiredEvidence": ["核心事实", "相关标的财务/业务映射", "反方证据", "未来催化剂"],
                "stopConditions": ["无法获得可验证证据", "无法映射到具体标的或财务变量"],
            },
            "brain_dump": response[:1200],
            "key_questions": [
                {"id": "Q1", "question": "这个事件的核心事实是什么？", "type": "事实验证型", "priority": "high", "search_hints": [topic_info.get("topic", "") + " 2026"]},
                {"id": "Q2", "question": "对相关公司的财务影响？", "type": "财务穿透型", "priority": "high", "search_hints": [topic_info.get("topic", "") + " revenue earnings"]},
                {"id": "Q3", "question": "主要风险和反面论据？", "type": "反面论证型", "priority": "high", "search_hints": [topic_info.get("topic", "") + " risks bear case"]},
            ],
            "data_needs": [],
            "pitfalls": [],
        }

    contract = plan.get("research_contract", {})
    questions = plan.get("key_questions", [])
    print(f"  Core question: {contract.get('coreQuestion', '')[:100]}")
    print(f"  Thesis: {contract.get('thesis', '')[:100]}")
    print(f"  Anti-thesis: {contract.get('antiThesis', '')[:100]}")
    print(f"  Brain dump: {len(plan.get('brain_dump', ''))} chars")
    print(f"  Key questions: {len(questions)}")
    for q in questions:
        print(f"    {q.get('id', '?')} [{q.get('type', '?')}] {q.get('question', '')[:60]}")
    print(f"  Pitfalls: {len(plan.get('pitfalls', []))}")

    return plan


# ============ Stage 3: Research ============

RESEARCH_AGENT_PROMPT = """你是 Y Daily 的研究助理。当前时间：{current_time}

今天你要为「{topic}」收集投研级素材。

=== Research Contract（必须服从） ===
{research_contract_text}

=== 关联标的 ===
{linked_tickers_text}

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
2. **Contract 必需证据**：优先满足 requiredEvidence；若满足不了，明确记录 info_gaps
3. **高优先问题**：针对每个 high 优先级问题，做 1-2 次 web_search
4. **财务映射**：对关联标的调用 fetch_financial_data，并搜索公司 IR/earnings/10-Q/10-K/公告
5. **一手信源**：至少 1 次搜索 SEC 文件 / 监管机构公告 / 公司 IR 页面
6. **Red Team**：至少 1 次搜索 bear case / risks / criticism / alternative explanation
7. **催化剂**：搜索未来 1-8 周可验证事件，如财报、监管截止日、产品发布、会议
8. **竞争格局**：至少 1 次搜索 competitors / alternatives

【分析陷阱提醒】
{pitfalls_text}

研究够了就停止调用工具。系统会自动整理你的研究成果。不要为了填满预算而搜索。
"""

RESEARCH_COMPRESS_PROMPT = """请将以上研究过程中获得的所有关键信息整理为结构化研究摘要。

=== 必须回答的问题清单 ===
{questions_text}

=== 整理要求 ===

输出严格 JSON（不要 markdown 代码块）：
{{
  "evidence_cards": [
    {{
      "id": "e1",
      "claim": "具体的、可验证的证据卡片",
      "sourceIds": ["c1"],
      "sourceType": "official/filing/company_ir/media/financial_data/analyst/other",
      "confidence": "high/medium/low",
      "supports": ["thesis", "antiThesis", "tickerImpact:NVDA", "catalyst"],
      "affectedTickers": ["NVDA"],
      "caveats": ["限制或口径说明"]
    }}
  ],
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
  "ticker_impact_map": [
    {{
      "ticker": "NVDA",
      "businessLine": "数据中心 GPU",
      "financialVariable": "收入增速/毛利率/资本开支/估值倍数",
      "direction": "positive/negative/mixed/unclear",
      "timeWindow": "1-8周/季度/年度",
      "evidenceIds": ["e1", "e2"],
      "confidence": "high/medium/low"
    }}
  ],
  "red_team": [
    {{
      "argument": "最强反方论点",
      "evidenceIds": ["e3"],
      "wouldInvalidateThesisIf": "什么信号出现会推翻 thesis"
    }}
  ],
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
5. 每个 ticker_impact_map.evidenceIds 必须能在 evidence_cards 中找到
6. 单一来源证据的 confidence 最高只能是 medium；0 来源只能是 low
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

    contract = plan.get("research_contract", {})
    research_contract_text = json.dumps(contract, ensure_ascii=False, indent=2) if contract else "（无 Research Contract）"
    linked_tickers = topic_info.get("linkedTickers", [])
    linked_tickers_text = ", ".join(linked_tickers) if linked_tickers else "（无直接标的，需验证是否值得继续）"

    system_prompt = RESEARCH_AGENT_PROMPT.format(
        topic=topic_info.get("topic", ""),
        research_contract_text=research_contract_text[:3000],
        linked_tickers_text=linked_tickers_text,
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

    evidence_bank = _normalize_evidence_bank(evidence_bank)

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
你要基于以下**已核查证据**写一份投资判断 memo，而不是新闻综述或行业科普。

你同时承担三个角色：
- Financial Mapper：把事件映射到关联标的、业务线、财务变量、时间窗口和证据 ID。
- Red Team：提出最强反方与推翻条件。
- Writer：把判断压缩成专业投资人愿意读的 memo。

=== Topic ===
{topic}

=== Research Contract ===
{research_contract_text}

=== 结构性框架 ===
{brain_dump}

=== 关键问题 ===
{questions_text}

=== Evidence Cards（优先使用，只能使用这些事实） ===
{evidence_cards_text}

=== 已核查 Claims（辅助溯源） ===
{verified_claims_text}

=== 初始标的映射（可修正，但证据 ID 必须存在） ===
{ticker_impact_map_text}

=== Red Team 素材 ===
{red_team_text}

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

【关键约束】

1. 只使用 Evidence Cards、已核查 Claims 和财务数据中的信息。不要写训练数据里的具体数字。
2. 不输出买入/卖出/做多/做空/加仓/减仓等交易指令。
3. 所有判断必须写成情景句式：若 [条件] 被验证，则 [标的] 面临 [方向] 压力/机会，关键信号是 [X]。
4. 单一来源证据最高只能给 🟡；未验证或证据冲突只能给 🔴。
5. 每个 Watchlist Impact 必须包含 Evidence IDs，例如：证据：e1, e3。
6. 催化剂必须尽量具体到日期、周或财报/监管节点；如果证据不足，写“未获取到具体时间”并降置信度。
7. 未回答问题要明确列入 What We Still Don't Know，不能用套话掩盖。
8. 禁止废话："短期/中期/长期"、"存在不确定性"、"需要密切关注"。

【输出结构】（严格遵守）

# [标题]

## Bottom Line
用 2-4 句话给出核心判断、置信度和最重要的验证信号。

## What Changed
列出今天的新信息如何改变原有判断。每条附证据 ID。

## Belief Update
明确读者读完后应该从什么判断更新到什么判断。

## Watchlist Impact
用表格：
| 标的 | 业务线/变量 | 方向 | 时间窗口 | 情景句式判断 | 置信度 | 证据 |

## Bull/Base/Bear Scenarios
用表格：
| 情景 | 条件 | 影响 | 关键验证信号 | 置信度 |

## Disconfirming Evidence
列出最强反方论点、证据 ID、以及什么信号会推翻核心 thesis。

## Catalyst Calendar
用表格：
| 时间 | 事件/信号 | 验证/推翻什么 | 关联标的 | 证据 |

## What We Still Don't Know
列出还没有足够证据回答的问题，以及为什么这会影响判断。

写作风格：克制、密度高、面向投资判断。目标 1800-3000 字。以 # 标题行开头，直接输出。
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

    # Evidence cards are the primary building blocks for the investment memo.
    evidence_card_lines = []
    for card in evidence_bank.get("evidence_cards", []):
        usable = card.get("usable", True)
        status = "usable" if usable else "not_usable"
        evidence_card_lines.append(
            f"  [{card.get('id', '?')}] ({status}, {card.get('confidence', 'low')}, {card.get('sourceType', 'other')}) "
            f"{card.get('claim', '')}"
        )
        evidence_card_lines.append(
            f"    sourceIds: {', '.join(card.get('sourceIds', [])) or 'none'} | "
            f"supports: {', '.join(card.get('supports', [])) or 'none'} | "
            f"tickers: {', '.join(card.get('affectedTickers', [])) or 'none'}"
        )
        caveats = card.get("caveats", [])
        if caveats:
            evidence_card_lines.append(f"    caveats: {'; '.join(caveats[:3])}")
    evidence_cards_text = "\n".join(evidence_card_lines) if evidence_card_lines else "（无 Evidence Cards）"

    contract = plan.get("research_contract", {})
    research_contract_text = json.dumps(contract, ensure_ascii=False, indent=2) if contract else "（无 Research Contract）"
    ticker_impact_map_text = json.dumps(
        evidence_bank.get("ticker_impact_map", []),
        ensure_ascii=False,
        indent=2,
    )[:5000] or "[]"
    red_team_text = json.dumps(
        evidence_bank.get("red_team", []),
        ensure_ascii=False,
        indent=2,
    )[:4000] or "[]"

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
        research_contract_text=research_contract_text[:3000],
        brain_dump=plan.get("brain_dump", "")[:1200],
        questions_text=questions_text,
        evidence_cards_text=evidence_cards_text[:12000],
        verified_claims_text=verified_claims_text[:15000],
        ticker_impact_map_text=ticker_impact_map_text,
        red_team_text=red_team_text,
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

FORMAT_PROMPT = """把以下投资判断 memo 转换为 JSON 格式。保留所有内容，只改变格式。

⚠️ 格式化规则：
- 标题和副标题中的引号必须配对
- 中文使用中文标点（""、''），不要混用半角引号
- sections content 中的 HTML 必须完整闭合
- 输出必须是合法 JSON，不要输出 markdown 代码块标记
- 新增字段必须尽量从对应章节抽取；抽不到时用空数组、空字符串或 null，不要编造。

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
  "whyNow": "为什么今天值得写",
  "bottomLine": "Bottom Line 章节的核心判断",
  "beliefUpdate": "Belief Update 章节的判断更新",
  "confidence": "high|medium|low",
  "tags": [
    {{"text": "标签名", "type": "up|down|warn"}}
  ],
  "keyTakeaways": ["核心判断1（情景假设句式）", "核心判断2", "核心判断3"],
  "relatedTickers": ["AAPL", "NVDA"],
  "tickerImpacts": [
    {{
      "ticker": "NVDA",
      "businessLine": "业务线/变量",
      "direction": "positive|negative|mixed|unclear",
      "timeWindow": "时间窗口",
      "judgment": "若[条件]被验证，则[标的]面临[方向]压力/机会，关键信号是[X]",
      "confidence": "high|medium|low",
      "evidenceIds": ["e1", "e3"]
    }}
  ],
  "scenarios": {{
    "bull": {{"conditions": "条件", "impact": "影响", "signals": "验证信号", "confidence": "high|medium|low"}},
    "base": {{"conditions": "条件", "impact": "影响", "signals": "验证信号", "confidence": "high|medium|low"}},
    "bear": {{"conditions": "条件", "impact": "影响", "signals": "验证信号", "confidence": "high|medium|low"}}
  }},
  "catalysts": [
    {{
      "time": "时间",
      "event": "事件/信号",
      "tests": "验证/推翻什么",
      "tickers": ["NVDA"],
      "evidenceIds": ["e1"]
    }}
  ],
  "redTeam": [
    {{
      "argument": "最强反方论点",
      "evidenceIds": ["e2"],
      "wouldInvalidateThesisIf": "推翻条件"
    }}
  ],
  "unknowns": ["尚未回答的问题1", "尚未回答的问题2"],
  "qualityScore": null,
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
        "whyNow": "",
        "bottomLine": _extract_section_excerpt(md_sections, "Bottom Line"),
        "beliefUpdate": _extract_section_excerpt(md_sections, "Belief Update"),
        "confidence": _extract_confidence_from_text(report_text),
        "tags": tags,
        "keyTakeaways": key_takeaways[:5],
        "relatedTickers": tickers,
        "tickerImpacts": [],
        "scenarios": {},
        "catalysts": [],
        "redTeam": [],
        "unknowns": _extract_section_bullets(md_sections, "What We Still"),
        "qualityScore": None,
        "sections": sections,
        "sourceIndices": [],
    }

    print(f"  Fallback: {len(sections)} sections, {len(tags)} tags")
    return result


# ============ Helpers ============

def _normalize_evidence_bank(evidence_bank):
    """Normalize optional v2 evidence structures while preserving old claim-based flow."""
    if not isinstance(evidence_bank, dict):
        return {
            "evidence_cards": [],
            "claims": [],
            "financial_data": {},
            "ticker_impact_map": [],
            "red_team": [],
            "questions_coverage": {"answered": {}, "unanswered": []},
            "source_independence_notes": "",
            "info_gaps": ["研究摘要不是合法对象"],
        }

    claims = evidence_bank.get("claims", [])
    if not isinstance(claims, list):
        claims = []
    for i, claim in enumerate(claims, 1):
        if not isinstance(claim, dict):
            claims[i - 1] = {"id": f"c{i}", "text": str(claim), "source_count": 0}
            claim = claims[i - 1]
        claim.setdefault("id", f"c{i}")
        urls = claim.get("source_urls") if isinstance(claim.get("source_urls"), list) else []
        names = claim.get("source_names") if isinstance(claim.get("source_names"), list) else []
        if "source_count" not in claim:
            claim["source_count"] = max(len(set(urls)), len(set(names)))
    evidence_bank["claims"] = claims

    claim_by_id = {c.get("id"): c for c in claims}
    cards = evidence_bank.get("evidence_cards", [])
    if not isinstance(cards, list):
        cards = []
    if not cards:
        for claim in claims:
            cards.append({
                "id": "e" + re.sub(r"^\D+", "", str(claim.get("id", ""))) if claim.get("id") else f"e{len(cards) + 1}",
                "claim": claim.get("text", ""),
                "sourceIds": [claim.get("id", "")],
                "sourceType": "media",
                "confidence": "medium" if _as_float(claim.get("source_count"), 0) >= 1 else "low",
                "supports": claim.get("answers_questions", []),
                "affectedTickers": [],
                "caveats": [],
            })

    normalized_cards = []
    for i, card in enumerate(cards, 1):
        if not isinstance(card, dict):
            card = {"claim": str(card)}
        card.setdefault("id", f"e{i}")
        card.setdefault("claim", card.get("text", ""))
        source_ids = card.get("sourceIds", [])
        if not isinstance(source_ids, list):
            source_ids = [str(source_ids)] if source_ids else []
        card["sourceIds"] = [sid for sid in source_ids if sid]
        if not card["sourceIds"] and claims:
            card["sourceIds"] = [claims[min(i - 1, len(claims) - 1)].get("id", "")]
        source_count = 0
        for sid in card["sourceIds"]:
            source_count = max(source_count, _as_float(claim_by_id.get(sid, {}).get("source_count"), 0))
        if source_count == 0:
            card["confidence"] = "low"
        elif source_count < 2 and card.get("confidence") == "high":
            card["confidence"] = "medium"
        card.setdefault("sourceType", "other")
        if not isinstance(card.get("supports"), list):
            card["supports"] = []
        if not isinstance(card.get("affectedTickers"), list):
            card["affectedTickers"] = []
        if not isinstance(card.get("caveats"), list):
            card["caveats"] = []
        normalized_cards.append(card)
    evidence_bank["evidence_cards"] = normalized_cards

    if not isinstance(evidence_bank.get("ticker_impact_map"), list):
        evidence_bank["ticker_impact_map"] = []
    if not isinstance(evidence_bank.get("red_team"), list):
        evidence_bank["red_team"] = []
    if not isinstance(evidence_bank.get("info_gaps"), list):
        evidence_bank["info_gaps"] = []
    if not isinstance(evidence_bank.get("questions_coverage"), dict):
        evidence_bank["questions_coverage"] = {"answered": {}, "unanswered": []}
    if not isinstance(evidence_bank.get("financial_data"), dict):
        evidence_bank["financial_data"] = {}

    return evidence_bank


def _apply_fact_check_to_evidence_bank(evidence_bank, fact_check_result):
    """Attach fact-check status to evidence cards and enforce confidence ceilings."""
    verdicts = {v.get("claim_id"): v for v in fact_check_result.get("verdicts", [])}
    bad_verdicts = {"refuted", "training_data_leak", "source_independence_error", "bad_analogy", "logic_gap"}

    claims = {c.get("id"): c for c in evidence_bank.get("claims", [])}
    for card in evidence_bank.get("evidence_cards", []):
        source_ids = card.get("sourceIds", [])
        card_verdicts = [verdicts.get(sid, {}) for sid in source_ids if sid in verdicts]
        unusable = any((not v.get("usable", True)) or v.get("verdict") in bad_verdicts for v in card_verdicts)
        card["usable"] = not unusable
        if unusable:
            card["confidence"] = "low"

        max_source_count = max((_as_float(claims.get(sid, {}).get("source_count"), 0) for sid in source_ids), default=0)
        if max_source_count == 0:
            card["confidence"] = "low"
        elif max_source_count < 2 and card.get("confidence") == "high":
            card["confidence"] = "medium"

        if any(v.get("verdict") in {"unverified", "disputed", "stale_data"} for v in card_verdicts):
            if card.get("confidence") == "high":
                card["confidence"] = "medium"

    card_by_id = {c.get("id"): c for c in evidence_bank.get("evidence_cards", [])}
    for impact in evidence_bank.get("ticker_impact_map", []):
        if not isinstance(impact, dict):
            continue
        evidence_ids = impact.get("evidenceIds", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = [str(evidence_ids)] if evidence_ids else []
        impact["evidenceIds"] = evidence_ids
        usable_cards = [card_by_id.get(eid) for eid in evidence_ids if card_by_id.get(eid, {}).get("usable", True)]
        if not usable_cards:
            impact["confidence"] = "low"
        elif any(c.get("confidence") == "low" for c in usable_cards) and impact.get("confidence") == "high":
            impact["confidence"] = "medium"

    return evidence_bank


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


def _extract_section_excerpt(md_sections, title_keyword, max_chars=260):
    """Return a compact text excerpt from the first section matching a title keyword."""
    for sec_title, sec_content in md_sections:
        if title_keyword.lower() in sec_title.lower():
            text = re.sub(r'[#*\-|`]', '', sec_content).strip()
            text = re.sub(r'\s+', ' ', text)
            return text[:max_chars]
    return ""


def _extract_section_bullets(md_sections, title_keyword):
    """Extract simple bullet-like lines from a matching markdown section."""
    bullets = []
    for sec_title, sec_content in md_sections:
        if title_keyword.lower() not in sec_title.lower():
            continue
        for line in sec_content.split("\n"):
            stripped = line.strip()
            if re.match(r'^[-*]\s+', stripped):
                clean = re.sub(r'^[-*]\s+', '', stripped)
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
                if clean:
                    bullets.append(clean[:220])
        break
    return bullets[:8]


def _extract_confidence_from_text(text):
    """Best-effort confidence extraction for fallback formatting."""
    if "🟢" in text or re.search(r'\bhigh\b', text, re.IGNORECASE):
        return "high"
    if "🔴" in text or re.search(r'\blow\b', text, re.IGNORECASE):
        return "low"
    if "🟡" in text or re.search(r'\bmedium\b', text, re.IGNORECASE):
        return "medium"
    return "medium"


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
    print("Y Daily — Investment Memo Generator (6-Stage Pipeline)")
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
    watchlist_context = load_watchlist_context()

    # Get recent topics for dedup
    existing_topics = [r.get("topic", "") for r in deep_research[:7] if r.get("topic")]

    # ====== Stage 1: Topic Selection ======
    topic_info = select_topic(client, breaking_news, ai_breaking_news, existing_topics, watchlist_context)
    should_skip, skip_reason = _should_skip_topic(topic_info)
    if should_skip:
        print(f"\nSKIP: {skip_reason}")
        print("No deep research report written today.")
        sys.exit(0)

    # ====== Stage 2: Research Planning ======
    plan = plan_research(client, topic_info, watchlist_context)

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

    should_skip, skip_reason = _should_skip_after_research(plan, evidence_bank)
    if should_skip:
        stats = _research_coverage_stats(plan, evidence_bank)
        print(f"\nSKIP: {skip_reason}.")
        print(
            f"Research coverage: {stats['answered_count']}/"
            f"{stats['total_questions']} ({stats['coverage_ratio']:.0%})."
        )
        print("No deep research report written today.")
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
    usable_count = sum(
        1 for v in verdicts
        if v.get("usable", True) and v.get("verdict") in ("verified", "partially_correct")
    )
    evidence_bank = _apply_fact_check_to_evidence_bank(evidence_bank, fact_check_result)

    if usable_count < MIN_USABLE_CLAIMS:
        print(f"\nSKIP: only {usable_count} verified/partially-correct claims; need {MIN_USABLE_CLAIMS}.")
        print("No deep research report written today.")
        sys.exit(0)

    if quality_score < MIN_DATA_QUALITY_SCORE:
        print(f"\nSKIP: data quality score {quality_score:.2f} below {MIN_DATA_QUALITY_SCORE:.2f}.")
        print("No deep research report written today.")
        sys.exit(0)

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
        "topicScores": topic_info.get("scores", {}),
        "topicCandidates": topic_info.get("candidates", []),
        "whyNow": report.get("whyNow") or topic_info.get("whyNow", ""),
        "bottomLine": report.get("bottomLine", ""),
        "beliefUpdate": report.get("beliefUpdate") or topic_info.get("beliefUpdate", ""),
        "confidence": report.get("confidence", "medium"),
        "qualityScore": report.get("qualityScore") if report.get("qualityScore") is not None else quality_score,
        "readTime": f"{read_minutes}分钟",
        "generatedAt": format_date_cst(now),
        "keyTakeaways": report.get("keyTakeaways", []),
        "relatedTickers": report.get("relatedTickers", []) or topic_info.get("linkedTickers", []),
        "tickerImpacts": report.get("tickerImpacts", []) or evidence_bank.get("ticker_impact_map", []),
        "scenarios": report.get("scenarios", {}),
        "catalysts": report.get("catalysts", []),
        "redTeam": report.get("redTeam", []) or evidence_bank.get("red_team", []),
        "unknowns": report.get("unknowns", []) or evidence_bank.get("info_gaps", []),
        "researchContract": plan.get("research_contract", {}),
        "evidenceCards": evidence_bank.get("evidence_cards", []),
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
