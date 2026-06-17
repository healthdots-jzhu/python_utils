import unittest

from prompt_validator import validate_prompt


class PromptValidatorTest(unittest.TestCase):
    def test_scores_clear_prompt_as_ready(self):
        result = validate_prompt(
            "Summarize the attached bug report into 5 bullets and include the likely root cause. "
            "Use the log file as context and keep the tone concise.",
            task_type="analysis",
        )

        self.assertTrue(result["ready_for_model"])
        self.assertGreaterEqual(result["score"], 70)
        self.assertTrue(result["strengths"])

    def test_flags_empty_prompt(self):
        result = validate_prompt("")

        self.assertFalse(result["ready_for_model"])
        self.assertIn("Prompt is empty.", result["issues"])
        self.assertTrue(result["blocking_issues"])

    def test_flags_underspecified_prompt(self):
        result = validate_prompt("Help with this")

        self.assertFalse(result["ready_for_model"])
        self.assertLess(result["score"], 70)
        self.assertTrue(any("context" in issue.lower() for issue in result["issues"]))
        self.assertTrue(result["suggested_questions"])


if __name__ == "__main__":
    unittest.main()