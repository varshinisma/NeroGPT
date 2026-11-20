import unittest


class PageSlots(unittest.TestCase):
    def test_the_report_slots_are_made_empty_before_the_first_event(self):
        with open("streamlit_app.py", encoding="utf-8") as handle:
            source = handle.read()
        made = source.index("for _ in range(PRE_SLOTS)")
        self.assertLess(made, source.index("for event in api_events("))      # an old answer's text cannot stay on screen under the new question
        self.assertIn("PRE_SLOTS = 14", source)


if __name__ == "__main__":
    unittest.main()
