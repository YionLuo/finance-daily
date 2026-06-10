import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import update_deep_research as deep


class DeepResearchAgentTests(unittest.TestCase):
    def test_low_score_topic_is_skipped(self):
        topic = {"scores": {"total": 0.4}, "shouldSkip": False}
        should_skip, reason = deep._should_skip_topic(topic)
        self.assertTrue(should_skip)
        self.assertIn("低于门槛", reason)

    def test_high_score_topic_can_continue(self):
        topic = {
            "scores": {"total": 0.72, "repetitionPenalty": 0.1},
            "shouldSkip": False,
        }
        should_skip, reason = deep._should_skip_topic(topic)
        self.assertFalse(should_skip)
        self.assertEqual(reason, "")

    def test_single_source_evidence_card_cannot_be_high_confidence(self):
        bank = {
            "claims": [
                {
                    "id": "c1",
                    "text": "A verifiable single-source claim.",
                    "source_urls": ["https://example.com/a"],
                }
            ],
            "evidence_cards": [
                {
                    "id": "e1",
                    "claim": "A verifiable single-source claim.",
                    "sourceIds": ["c1"],
                    "confidence": "high",
                }
            ],
        }
        normalized = deep._normalize_evidence_bank(bank)
        self.assertEqual(normalized["claims"][0]["source_count"], 1)
        self.assertEqual(normalized["evidence_cards"][0]["confidence"], "medium")

    def test_fact_check_marks_refuted_card_unusable(self):
        bank = deep._normalize_evidence_bank({
            "claims": [{"id": "c1", "text": "Claim", "source_count": 2}],
            "evidence_cards": [{"id": "e1", "claim": "Claim", "sourceIds": ["c1"], "confidence": "high"}],
            "ticker_impact_map": [{"ticker": "NVDA", "evidenceIds": ["e1"], "confidence": "high"}],
        })
        checked = deep._apply_fact_check_to_evidence_bank(bank, {
            "verdicts": [{"claim_id": "c1", "verdict": "refuted", "usable": False}]
        })
        self.assertFalse(checked["evidence_cards"][0]["usable"])
        self.assertEqual(checked["evidence_cards"][0]["confidence"], "low")
        self.assertEqual(checked["ticker_impact_map"][0]["confidence"], "low")

    def test_thin_research_coverage_is_skipped(self):
        plan = {
            "key_questions": [{"id": f"Q{i}", "question": f"Q{i}"} for i in range(1, 9)]
        }
        bank = {"questions_coverage": {"answered": {"Q4": ["c1"]}, "unanswered": []}}
        should_skip, reason = deep._should_skip_after_research(plan, bank)
        self.assertTrue(should_skip)
        self.assertIn("关键问题覆盖不足", reason)

    def test_sufficient_research_coverage_can_continue(self):
        plan = {
            "key_questions": [{"id": f"Q{i}", "question": f"Q{i}"} for i in range(1, 6)]
        }
        bank = {"questions_coverage": {"answered": {"Q1": ["c1"], "Q2": ["c2"]}, "unanswered": []}}
        should_skip, reason = deep._should_skip_after_research(plan, bank)
        self.assertFalse(should_skip)
        self.assertEqual(reason, "")

    def test_fallback_json_contains_memo_fields(self):
        report = """# Test Memo

## Bottom Line
若 X 被验证，则 NVDA 面临机会，关键信号是 Y。🟡

## Belief Update
从普通新闻更新为需要验证的基本面假设。

## What We Still Don't Know
- 是否能持续转化为收入。
"""
        result = deep._fallback_md_to_json(report)
        self.assertIn("bottomLine", result)
        self.assertIn("beliefUpdate", result)
        self.assertIn("unknowns", result)
        self.assertEqual(result["confidence"], "medium")


if __name__ == "__main__":
    unittest.main()
