import unittest

from app.gold_ai_trader import claude


class GoldAiClaudeJsonParserTests(unittest.TestCase):
    def test_parse_json_from_markdown_fence(self):
        raw = """```json
{"action":"skip","direction":null,"entry":null,"stop_loss":null,"take_profit":null,"confidence":37,"rationale":"too extended"}
```"""
        parsed = claude._parse_json(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "skip")
        self.assertEqual(parsed["confidence"], 37)

    def test_parse_json_with_leading_trailing_prose(self):
        raw = (
            "Reasoning: setup overextended.\n"
            '{"action":"skip","direction":null,"entry":null,"stop_loss":null,'
            '"take_profit":null,"confidence":28,"rationale":"outside zone"}\n'
            "End."
        )
        parsed = claude._parse_json(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "skip")
        self.assertEqual(parsed["confidence"], 28)

    def test_extract_balanced_json_object_handles_braces_in_strings(self):
        raw = (
            'text before {"action":"skip","rationale":"literal {brace} in text","confidence":12} '
            "text after"
        )
        obj = claude._extract_first_json_object(raw)
        self.assertIsNotNone(obj)
        self.assertIn('"literal {brace} in text"', obj)
        parsed = claude._parse_json(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["confidence"], 12)

    def test_strict_user_prompt_instruction(self):
        prompt = claude._json_only_user_prompt("CTX")
        self.assertIn("Return ONLY one valid JSON object", prompt)
        self.assertIn("No prose, no markdown, no code fences", prompt)
        self.assertTrue(prompt.endswith("CTX"))


if __name__ == "__main__":
    unittest.main()
