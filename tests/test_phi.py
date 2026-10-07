"""Identifier scrubbing: identifiers go, clinical content and search quality stay."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phi  # noqa: E402
import qa_stream  # noqa: E402

PAIRS = [  # (question with identifiers, the same question as a clinician would type it without them)
    ("John, 5-year-old boy with autism, what is the evidence for risperidone?", "5-year-old boy with autism, what is the evidence for risperidone?"),
    ("Mrs. Sharma, 62, MRN 4482913, has multiple sclerosis: latest evidence on ocrelizumab", "62, has multiple sclerosis: latest evidence on ocrelizumab"),
    ("Priya's HbA1c is 8.2% on metformin 500 mg, how to intensify treatment of type 2 diabetes?", "HbA1c is 8.2% on metformin 500 mg, how to intensify treatment of type 2 diabetes?"),
    ("DOB 12/03/2016, phone +91 98765 43210: epilepsy in children, first-line drugs", "epilepsy in children, first-line drugs"),
]


class Removal(unittest.TestCase):
    def test_names_ids_contacts_removed(self):
        for text, gone in [("John has seizures", "John"), ("Mrs. Sharma's MRI shows lesions", "Sharma"), ("MRN 4482913 seizures", "4482913"),
                           ("email dr.rao@hospital.org about seizures", "hospital.org"), ("call 044-2345-6789 about seizures", "2345"),
                           ("DOB 12/03/2016 seizures", "2016"), ("lives at 12 Gandhi Road, Chennai 600004 seizures", "Gandhi")]:
            out = phi.scrub(text).text
            self.assertNotIn(gone, out, f"{text!r} -> {out!r}")

    def test_clinical_details_kept(self):
        q = "5-year-old male with HbA1c 7.2% and ALT 45 U/L on risperidone 0.5 mg, autism, last updated 2025"
        r = phi.scrub(q)
        self.assertEqual(r.text, q)
        self.assertFalse(r.changed)

    def test_eponymous_diseases_not_removed(self):
        for q in ["Parkinson's disease treatment", "Down syndrome and hearing loss", "What is the dose for Wilson disease", "Crohn's disease in children",
                  "Tourette syndrome treatment", "Glasgow Coma Scale 8 management"]:
            self.assertEqual(phi.scrub(q).text, q)

    def test_removed_values_are_not_stored(self):
        r = phi.scrub("John Smith MRN 4482913")
        self.assertTrue(all(isinstance(v, int) for v in r.removed.values()))
        self.assertNotIn("John", repr(r.removed))

    def test_idempotent(self):
        once = phi.scrub("John, 5-year-old boy with autism, MRN 12345, phone 044-2345-6789").text
        self.assertEqual(phi.scrub(once).text, once)


class SearchQualityNotHarmed(unittest.TestCase):
    def test_same_search_terms_with_and_without_identifiers(self):
        for with_ids, clean in PAIRS:
            scrubbed = phi.scrub(with_ids).text
            a, b = qa_stream.keywords(scrubbed)[0], qa_stream.keywords(clean)[0]
            # every clinical keyword the clean question yields must still be searched; the placeholder may add at most the word "patient"
            self.assertTrue(set(b) <= set(a + qa_stream.keywords(clean)[0]), f"{scrubbed!r}: {a} vs {b}")
            self.assertEqual([w for w in a if w != "patient"][:len(b)], b, f"{scrubbed!r}: {a} vs {b}")

    def test_core_stems_equal(self):
        for with_ids, clean in PAIRS:
            self.assertEqual(qa_stream.core_stems(phi.scrub(with_ids).text) - {"patie"}, qa_stream.core_stems(clean) - {"patie"})


if __name__ == "__main__":
    unittest.main()


class HistoryIsScrubbed(unittest.TestCase):
    def test_saved_conversation_contains_no_identifiers(self):
        import server
        msgs = server.scrub_messages([{"role": "user", "content": "John Smith, MRN 4482913, 5-year-old with autism: how to treat?"},
                                      {"role": "assistant", "content": "An earlier answer."}])
        self.assertNotIn("John", msgs[0]["content"])
        self.assertNotIn("4482913", msgs[0]["content"])
        self.assertIn("5-year-old with autism", msgs[0]["content"])
        self.assertEqual(msgs[1]["content"], "An earlier answer.")
