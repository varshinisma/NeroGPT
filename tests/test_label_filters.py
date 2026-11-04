import unittest

import landscape
import qa_stream


class LabelFilters(unittest.TestCase):
    def test_homeopathic_product_is_not_a_drug_for_a_clinical_question(self):
        r = {"generic": "Strychnos Nux-Vomica Seed (Nux Vomica 30X)", "indication_excerpt": "Use: temporary relief of insomnia"}
        self.assertEqual(landscape.unsuitable_label("What medication is used for insomnia in children?", r), "homeopathic product")

    def test_adult_only_label_is_dropped_for_a_child_question_only(self):
        r = {"generic": "Zolpidem", "indication_excerpt": "insomnia", "label_extras": {"pediatric_use": "Safety and effectiveness of zolpidem in pediatric patients have not been established."}}
        self.assertTrue(landscape.unsuitable_label("What medication is used for insomnia in children?", r))
        self.assertIsNone(landscape.unsuitable_label("What medication is used for insomnia in adults?", r))

    def test_a_label_with_a_paediatric_age_range_is_kept(self):
        r = {"generic": "Risperidone", "indication_excerpt": "irritability", "label_extras": {"pediatric_use": "Efficacy was established in children and adolescents ages 5 to 17 years."}}
        self.assertIsNone(landscape.unsuitable_label("How to treat autism in children", r))

    def test_drop_marks_the_record_not_relevant(self):
        r = {"generic": "Zolpidem", "model_relevant": True, "indication_excerpt": "insomnia", "label_extras": {"pediatric_use": "Safety and effectiveness in pediatric patients have not been established."}}
        self.assertEqual(landscape.drop_unsuitable_labels("insomnia in a 4 year old", [r]), ["Zolpidem"])
        self.assertFalse(r["model_relevant"])


class AgeRange(unittest.TestCase):
    RISP = {"generic": "Risperidone", "indication_excerpt": "irritability associated with autistic disorder (ages 5 to 17 years)", "dose_statement": "schizophrenia adolescents 13 to 17 years"}

    def test_a_label_for_older_children_is_kept_with_an_age_note_for_a_three_year_old(self):
        self.assertIsNone(landscape.unsuitable_label("Three year old child has autism. What could be the management?", self.RISP))
        self.assertIn("labelled for ages 5 to 17 years", landscape.age_note(self.RISP, 3.0))

    def test_the_same_label_is_kept_for_a_ten_year_old(self):
        self.assertIsNone(landscape.unsuitable_label("A 10 year old child has autism. What could be the management?", self.RISP))

    def test_no_stated_age_means_no_drop(self):
        self.assertIsNone(landscape.unsuitable_label("How to treat autism in children", self.RISP))


class EmptyParts(unittest.TestCase):
    def test_label_only_bullets_empty_footers_and_empty_headings_are_removed(self):
        text = ("### Work-up plan\n#### Diagnostic assessment\n- **Genetic evaluation**\n- **Testing**: Select tests (PMID 1234567).\n#### Differential diagnosis\n\n"
                "### Management plan\n- Therapy helps (PMID 1234567).\n- Pharmacologic management (if indicated)\nNot covered by the retrieved sources:\n")
        out = qa_stream.drop_empty_parts(text)
        self.assertNotIn("Genetic evaluation", out)
        self.assertNotIn("Differential diagnosis", out)
        self.assertNotIn("Pharmacologic management", out)
        self.assertNotIn("Not covered", out)
        self.assertIn("### Work-up plan", out)
        self.assertIn("Select tests (PMID 1234567)", out)


    def test_the_not_covered_note_is_never_printed(self):
        text = "### Management\n- Family support [12]. *(Not covered by the retrieved sources: family counselling.)*\n- Therapy helps (PMID 1234567).\n- Not covered by the retrieved sources: EEG, MRI.\n"
        out = qa_stream.drop_empty_parts(text)
        self.assertNotIn("Not covered", out)
        self.assertIn("Family support [12].", out)
        self.assertIn("Therapy helps", out)


if __name__ == "__main__":
    unittest.main()
