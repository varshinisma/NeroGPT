import json
import unittest
from unittest import mock

import drug_handoff
import qa_stream
import retrieval_routing as rr


def state_with(**over):
    state = qa_stream.new_state("How do you manage aggression in severe autism?")
    state["papers"]["1"] = {"title": "Risperidone and haloperidol for irritability", "abstract": "Risperidone and haloperidol reduce irritability. Clonidine helped sleep.", "retracted": False}
    state["web"].append({"title": "Medicines", "snippet": "Melatonin is used for sleep problems."})
    state.update(over)
    return state


def label(name):
    return {"generic": name, "indication_excerpt": f"{name} is indicated for irritability", "label_date": "2026-01-01", "url": "http://x"}


class Handoff(unittest.TestCase):
    def run_stage(self, state, answer, labels=None):
        searched = []

        def search(terms, names, limit):
            searched.append(list(names))
            return json.dumps({"results": [label(n) for n in names if n.lower() in (labels or {n.lower() for n in names})], "searched_terms": []})
        qa_stream.drug_handoff_stage(state, "How do you manage aggression in severe autism?", ask=lambda *a, **k: answer, search=search)
        return searched

    def test_only_names_a_source_writes_are_searched(self):
        state = state_with()
        searched = self.run_stage(state, {"drugs": [{"name": "haloperidol"}, {"name": "clonidine", "use": "symptom_or_comorbidity"}, {"name": "madeupmab"}, {"name": "melatonin"}]})
        self.assertEqual(searched, [["haloperidol", "clonidine", "melatonin"]])
        self.assertEqual(state["drug_handoff"]["kept"], ["haloperidol", "clonidine", "melatonin"])
        self.assertEqual(state["drug_handoff"]["labels_added"], 3)
        self.assertEqual({r["generic"] for r in state["regulatory"]}, {"haloperidol", "clonidine", "melatonin"})

    def test_labels_already_in_hand_and_named_drugs_are_not_searched_again(self):
        state = state_with()
        state["regulatory"].append(label("Risperidone"))
        searched = self.run_stage(state, {"drugs": [{"name": "Risperidone"}, {"name": "haloperidol"}]})
        self.assertEqual(searched, [["haloperidol"]])

    def test_six_names_to_a_search(self):
        state = state_with()
        state["papers"]["1"]["abstract"] = " ".join(f"drug{c}x" for c in "abcdefghi")
        searched = self.run_stage(state, {"drugs": [{"name": f"drug{c}x"} for c in "abcdefghi"]})
        self.assertEqual([len(s) for s in searched], [6, 3])

    def test_a_failure_never_stops_the_stage(self):
        state = state_with()

        def boom(*a, **k):
            raise RuntimeError("model down")
        qa_stream.drug_handoff_stage(state, "q", ask=boom)
        self.assertIn("error", state["drug_handoff"])

    def test_it_can_be_switched_off(self):
        state = state_with()
        with mock.patch.dict("os.environ", {"DRUG_HANDOFF": "off"}):
            self.assertEqual(self.run_stage(state, {"drugs": [{"name": "haloperidol"}]}), [])
        self.assertNotIn("drug_handoff", state)

    def test_no_sources_means_no_call(self):
        state = qa_stream.new_state("q")
        qa_stream.drug_handoff_stage(state, "q", ask=lambda *a, **k: 1 / 0)
        self.assertNotIn("drug_handoff", state)


class LabelsByNameKeepTheCondition(unittest.TestCase):
    """A label looked up by a drug's name still needs the condition, or its dosing text is not the one for the use the question is about."""

    def test_named_drug_searches_carry_the_conditions_terms(self):
        plan = {"drugs": ["drugone", "drugtwo"], "condition": "autism in children"}
        ops = qa_stream.named_drug_operations("dosage of drugone and drugtwo for autism in children", plan, [])
        self.assertEqual(ops[0].args["condition_terms"], ["autism", "children"])
        self.assertEqual(ops[0].args["drug_names"], ["drugone", "drugtwo"])

    def test_the_handoff_search_carries_them_too(self):
        state = state_with()
        state["entities"] = {"condition": "autism in children", "drugs": []}
        seen = []
        qa_stream.drug_handoff_stage(state, "q", ask=lambda *a, **k: {"drugs": [{"name": "haloperidol"}]}, search=lambda terms, names, n: seen.append(list(terms)) or json.dumps({"results": []}))
        self.assertEqual(seen, [["autism", "children"]])


class SymptomSearch(unittest.TestCase):
    PLAN = {"condition": "aggression in severe autism", "understanding": {"medicines_needed": True}}

    def test_one_web_search_for_the_conditions_symptoms(self):
        with mock.patch("landscape.SHOW_SYMPTOM_DRUGS", True):
            ops = qa_stream.symptom_drug_operations("q", self.PLAN, [])
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].target, "web")
        self.assertIn("aggression in severe autism", ops[0].query)

    def test_not_when_the_table_is_off_or_medicines_are_not_needed(self):
        with mock.patch("landscape.SHOW_SYMPTOM_DRUGS", False):
            self.assertEqual(qa_stream.symptom_drug_operations("q", self.PLAN, []), [])
        with mock.patch("landscape.SHOW_SYMPTOM_DRUGS", True):
            self.assertEqual(qa_stream.symptom_drug_operations("q", {**self.PLAN, "understanding": {"medicines_needed": False}}, []), [])

    def test_the_skill_allows_symptom_drugs_of_the_condition(self):
        text = open("skills/drug-intelligence/SKILL.md", encoding="utf-8").read()
        self.assertIn("treats_associated_symptom_or_comorbidity", text)
        self.assertNotIn("keep ONLY if the question names the symptom", text)


if __name__ == "__main__":
    unittest.main()
