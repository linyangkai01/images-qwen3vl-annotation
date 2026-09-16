from __future__ import annotations

import unittest

from traffic_cv_closed_loop.adapters.qwen import ThinkingJsonParser, _extract_json_from_text


class ThinkingJsonParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ThinkingJsonParser()

    def test_takes_last_balanced_json_after_thinking(self) -> None:
        text = (
            "思考：用户举例 {\"a\": 1} 只是示意，{示例对象: 已忽略}\n"
            "最终答案如下：\n"
            "{\"objects\": [], \"hazard_report\": {\"overall_risk\": \"none\"}}"
        )
        parsed = self.parser.parse(text)
        self.assertEqual(parsed["hazard_report"]["overall_risk"], "none")
        self.assertEqual(parsed["objects"], [])

    def test_strips_think_tags(self) -> None:
        text = " thinking内部推理 {\"noise\": true} <｜end▁of▁thinking｜>{\"objects\": [], \"hazard_report\": {}}"
        parsed = self.parser.parse(text)
        self.assertNotIn("noise", parsed)

    def test_strips_thinking_fence(self) -> None:
        text = "```thinking\nsome reasoning with {braces}\n```\n{\"objects\": [], \"hazard_report\": {}}"
        parsed = self.parser.parse(text)
        self.assertEqual(parsed["objects"], [])

    def test_repairs_trailing_commas(self) -> None:
        text = '{"objects": [{"label": "person", "bbox": [0, 0, 1, 1],},],}'
        parsed = self.parser.parse(text)
        self.assertEqual(parsed["objects"][0]["label"], "person")

    def test_repairs_single_quotes(self) -> None:
        text = "{'objects': [], 'hazard_report': {'overall_risk': 'none'}}"
        parsed = self.parser.parse(text)
        self.assertEqual(parsed["hazard_report"]["overall_risk"], "none")

    def test_repairs_python_literals(self) -> None:
        text = '{"flag": True, "missing": None, "value": False}'
        parsed = self.parser.parse(text)
        self.assertIs(parsed["flag"], True)
        self.assertIsNone(parsed["missing"])
        self.assertIs(parsed["value"], False)

    def test_repairs_truncated_json(self) -> None:
        text = '{"objects": [{"label": "person", "bbox": [0, 0, 1, 1]}'
        parsed = self.parser.parse(text)
        self.assertEqual(parsed["objects"][0]["label"], "person")

    def test_raises_when_unparseable(self) -> None:
        with self.assertRaises(ValueError):
            self.parser.parse("there is no json here at all")

    def test_extract_json_from_text_prefers_last(self) -> None:
        # Regression guard: the legacy helper used to return the FIRST fragment.
        text = 'prefix {"first": 1} middle {"second": 2} suffix'
        self.assertEqual(_extract_json_from_text(text), {"second": 2})

    def test_fenced_draft_does_not_override_later_final_answer(self) -> None:
        # Regression (F3): a fenced draft inside the reasoning must not beat the
        # real answer printed afterwards.
        text = '```json\n{"draft": 1, "objects": []}\n```\n最终答案：{"final": 2, "objects": []}'
        self.assertEqual(self.parser.parse(text), {"final": 2, "objects": []})

    def test_comment_stripping_does_not_corrupt_urls(self) -> None:
        # Regression (F4): the "//" comment stripper must skip string literals.
        text = "{'url': 'http://example.com/x', 'objects': []}"
        self.assertEqual(self.parser.parse(text)["url"], "http://example.com/x")

    def test_trailing_degenerate_span_does_not_override_real_answer(self) -> None:
        # Regression (N3): the F3 "keep the later span" rule must not let a
        # stray trailing ``{}`` / ``[]`` silently degrade a good answer to empty.
        self.assertEqual(
            self.parser.parse('```json\n{"real": 1}\n```\n说明 {}'), {"real": 1}
        )
        self.assertEqual(
            self.parser.parse('```json\n{"real": 1}\n```\n说明 []'), {"real": 1}
        )
        self.assertEqual(
            self.parser.parse('{"real": 1} then stray {}'), {"real": 1}
        )
        # A degenerate payload is still returned when it is the only candidate.
        self.assertEqual(self.parser.parse("{}"), {})
        self.assertEqual(self.parser.parse("[]"), [])
        # A non-empty object that merely *contains* an empty list is not
        # degenerate and must be returned as-is.
        self.assertEqual(self.parser.parse('{"objects": []}'), {"objects": []})


if __name__ == "__main__":
    unittest.main()
