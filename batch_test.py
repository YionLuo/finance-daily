#!/usr/bin/env python3
"""
Batch test: run the deep research pipeline N times and save raw report text for review.
Does NOT write to index.html — only saves to /tmp/reports/.
"""

import os, sys, json, re
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from utils import (
    read_html, extract_js_array, create_llm_client,
    now_cst, format_date_cst,
)
from update_deep_research import (
    select_topic, brain_dump, research_collect, write_report, fact_check, format_to_json,
)

os.makedirs("/tmp/reports", exist_ok=True)

html = read_html()
breaking_news = extract_js_array(html, "breakingNews")
ai_breaking_news = extract_js_array(html, "aiBreakingNews")
deep_research = extract_js_array(html, "deepResearch")
existing_topics = [r.get("topic", "") for r in deep_research[:7] if r.get("topic")]

client = create_llm_client(required=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3

for run in range(1, N + 1):
    print(f"\n{'='*60}")
    print(f"  RUN {run}/{N}")
    print(f"{'='*60}")

    # Stage 1
    topic_info = select_topic(client, breaking_news, ai_breaking_news, existing_topics)
    existing_topics.append(topic_info.get("topic", ""))  # avoid dups across runs

    # Stage 2
    brain_dump_text = brain_dump(client, topic_info)

    # Breaking context
    breaking_lines = []
    for item in (breaking_news + ai_breaking_news)[:30]:
        breaking_lines.append(f"[{item.get('time', '')}] {item.get('text', '')}")
    breaking_context = "\n".join(breaking_lines)

    # Stage 3a
    research_materials, sources = research_collect(client, topic_info, breaking_context)

    # Stage 3b
    report_text = write_report(client, topic_info, brain_dump_text, research_materials)

    # Stage 3.5
    report_text, fc_result = fact_check(client, report_text)

    # Stage 4
    report_json = format_to_json(client, report_text)

    # Save raw text
    outfile = f"/tmp/reports/report_{run}.txt"
    with open(outfile, "w") as f:
        f.write(f"=== TOPIC: {topic_info.get('topic', '?')} ===\n")
        f.write(f"=== REASON: {topic_info.get('topicReason', '')} ===\n")
        f.write(f"=== ANGLE: {topic_info.get('angle', '')} ===\n\n")
        f.write("--- BRAIN DUMP ---\n")
        f.write(brain_dump_text + "\n\n")
        f.write("--- RESEARCH MATERIALS ---\n")
        f.write(research_materials + "\n\n")
        f.write("--- RAW REPORT ---\n")
        f.write(report_text + "\n\n")
        if fc_result:
            f.write("--- FACT CHECK ---\n")
            f.write(json.dumps(fc_result, ensure_ascii=False, indent=2) + "\n\n")
        f.write("--- FORMATTED JSON ---\n")
        f.write(json.dumps(report_json, ensure_ascii=False, indent=2) + "\n")

    print(f"\n  Saved: {outfile} ({os.path.getsize(outfile)} bytes)")

print(f"\n{'='*60}")
print(f"All {N} reports saved to /tmp/reports/")
print(f"{'='*60}")
