import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import conversation_memory as m

MEM = [{"question": "How do you manage aggression in severe autism?", "condition": "aggression in severe autism", "drugs": ["risperidone", "aripiprazole (Abilify)"]},
       {"question": "What are the medications used to control tics in Tourette syndrome", "condition": "tics in Tourette syndrome", "drugs": ["haloperidol", "aripiprazole", "pimozide"]}]
SAYS = lambda text: (lambda instructions, prompt: text)
FOLLOW = '{"follow_up": true, "turns": "all", "ask": "Give the dosage information", "needs_drugs": true}'


class Resolve(unittest.TestCase):
    def test_the_above_drugs_become_a_complete_question(self):
        r = m.resolve("give me the dosage information of all the above drugs", MEM, SAYS(FOLLOW))
        self.assertTrue(r["follow_up"])
        for word in ("risperidone", "aripiprazole", "haloperidol", "pimozide", "aggression in severe autism", "tics in Tourette syndrome", "dosage"):
            self.assertIn(word, r["question"])
        self.assertEqual(r["question"].lower().count("aripiprazole"), 1)      # a drug in both answers is named once
        self.assertNotIn("Abilify", r["question"])

    def test_the_model_cannot_add_a_drug(self):
        r = m.resolve("dose of the above drugs", MEM, SAYS('{"follow_up": true, "turns": "all", "ask": "Give the dose", "needs_drugs": true, "drugs": ["invented"]}'))
        self.assertNotIn("invented", r["question"])

    def test_a_short_question_that_stands_alone_is_left_as_asked(self):
        r = m.resolve("What is the dose of risperidone for children?", MEM, SAYS('{"mode": "new"}'))
        self.assertEqual((r["follow_up"], r["mode"], r["question"]), (False, "new", "What is the dose of risperidone for children?"))

    def test_no_memory_or_model_failure_leaves_the_question_alone(self):
        self.assertFalse(m.resolve("dose of the above drugs", [], SAYS(FOLLOW))["follow_up"])
        self.assertFalse(m.resolve("dose of the above drugs", MEM, lambda i, p: 1 / 0)["follow_up"])
        self.assertFalse(m.resolve("dose of the above drugs", MEM, SAYS("not json"))["follow_up"])
        self.assertFalse(m.resolve("dose of the above drugs", MEM, SAYS('{"follow_up": false}'))["follow_up"])

    def test_one_earlier_question_only(self):
        r = m.resolve("what about the dose of those?", MEM, SAYS('{"follow_up": true, "turns": [2], "ask": "Give the dose", "needs_drugs": true}'))
        self.assertIn("pimozide", r["question"])
        self.assertNotIn("risperidone", r["question"])


class Storage(unittest.TestCase):
    def test_turn_from_the_final_state_and_round_trip(self):
        final = {"markdown": "## Clinical summary\nBottom line text.\n\n## Latest findings\nx", "state": {"entities": {"condition": "tics", "intent": "drug_treatment", "conditions": [{"name": "tics", "stems": ["tics"]}]},
                 "listed_drugs": ["Pimozide", "haloperidol (Haldol)", "pimozide"], "regulatory": [{"generic": "clonidine", "model_relevant": True}, {"generic": "other", "model_relevant": False}]}}
        turn = m.turn_from_final("q", "q2", final)
        self.assertEqual(turn["drugs"], ["Pimozide", "haloperidol", "clonidine"])
        self.assertEqual(turn["summary"], "Bottom line text.")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(m, "MEMORY_DIR", Path(d)):
            m.save("chat-1", [turn])
            self.assertEqual(m.load("chat-1")[0]["condition"], "tics")
            self.assertEqual(m.load("chat-2"), [])
            m.save("../evil", [turn])      # an identifier cannot leave the folder
            self.assertEqual([p.name for p in Path(d).iterdir() if p.name != "chat-1.json"], ["evil.json"])

    def test_old_conversations_are_rebuilt_from_their_reports(self):
        history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "## Treatment Drug Landscape\n\n| Drug | Indication |\n|---|---|\n| Risperidone (Brand) | x |\n\n## Next\n"}]
        turns = m.turns_from_history(history)
        self.assertEqual(turns[0]["drugs"], ["Risperidone"])
        self.assertEqual(turns[0]["question"], "q1")


class ApiUsesMemory(unittest.TestCase):
    def test_follow_up_is_rewritten_remembered_and_announced(self):
        import api
        asked = []

        def stream_answer(question):
            asked.append(question)
            yield {"type": "final", "markdown": "## Clinical summary\nS", "timings": {}, "state": {"entities": {"condition": "c"}, "listed_drugs": ["drugx"]}}

        pipeline = lambda: (stream_answer, lambda q, f: iter([{"type": "pdf", "name": "r.pdf"}]))
        with tempfile.TemporaryDirectory() as d, mock.patch.object(m, "MEMORY_DIR", Path(d)), mock.patch("server.save_conversation"), mock.patch("server.scrub_messages", side_effect=lambda x: x):
            m.save("chat-9", MEM)
            body = api.AskBody(question="give me the dosage information of all the above drugs", conversation_id="chat-9")
            out = [json.loads(x) for x in api.ask_events(body, pipeline, SAYS(FOLLOW))]
            self.assertIn("understood", [e["type"] for e in out])
            self.assertIn("pimozide", asked[0])
            self.assertEqual(len(m.load("chat-9")), 3)      # the new answer is remembered too
            self.assertEqual(m.load("chat-9")[-1]["drugs"], ["drugx"])


CONVERSATION = '{"mode": "conversation", "turns": "all", "ask": "Explain simply", "needs_drugs": false}'
FULL = [dict(t, markdown="Risperidone 0.5 mg daily for aggression. Pimozide 1 mg for tics.", kind="report") for t in MEM]


class Modes(unittest.TestCase):
    def test_conversation_mode_keeps_the_question_and_the_turns(self):
        r = m.resolve("explain that simply", MEM, SAYS(CONVERSATION))
        self.assertEqual((r["mode"], r["question"], r["turns"]), ("conversation", "explain that simply", [1, 2]))

    def test_every_message_after_the_first_is_read_by_the_model_not_by_a_word_list(self):
        calls = []
        q = "What are the first-line drugs for migraine prophylaxis in adults with comorbid depression and anxiety?"
        m.resolve(q, MEM, lambda i, p: calls.append(1) or CONVERSATION)
        self.assertEqual(calls, [1])      # the model decides; no length or wording rule does

    def test_context_holds_the_chosen_answers_within_the_budget(self):
        ctx = m.build_context(FULL, [1, 2], budget=100000)
        self.assertIn("EARLIER ANSWER 1", ctx)
        self.assertIn("Pimozide 1 mg", ctx)
        self.assertLessEqual(len(m.build_context(FULL, [1, 2], budget=1200)), 1500)

    def test_numbers_not_in_the_earlier_answers_are_found(self):
        self.assertEqual(m.unmatched_numbers("Give 0.5 mg, or 12.5 mg", "Risperidone 0.5 mg"), ["12.5"])


class OtherCondition(unittest.TestCase):
    def test_a_different_condition_is_not_pulled_back_to_the_earlier_one(self):
        mem = [{"question": "treatment of the first cancer", "condition": "the first cancer", "drugs": ["drugone", "drugtwo"], "markdown": "x", "kind": "report"}]
        reply = '{"mode": "research", "turns": "all", "ask": "Are they also used to treat the second condition", "needs_drugs": true, "names_own_condition": true}'
        r = m.resolve("are those drugs also used to treat the second condition?", mem, SAYS(reply))
        self.assertIn("drugone, drugtwo", r["question"])
        self.assertIn("second condition", r["question"])
        self.assertNotIn("first cancer", r["question"])


class QuickCheck(unittest.TestCase):
    import quick_check as qc
    MEM1 = [{"question": "treatment of the first condition", "condition": "first condition", "drugs": ["drugone", "drugtwo", "drugthree"], "markdown": "x", "kind": "report"}]
    REPLY = '{"mode": "research", "turns": "all", "ask": "Are they also used to treat autism", "needs_drugs": true, "names_own_condition": true, "own_condition": "autism"}'

    def lookups(self, labels=None, papers=None):
        labels, papers = labels or {}, papers or {}
        fda = lambda d: json.dumps({"results": [{"generic": d, "indication_excerpt": labels.get(d, "indicated for a different disease.")}]})
        pub = lambda d, c: json.dumps({"results": [{"title": t, "pmid": "123"} for t in papers.get(d, [])]})
        return fda, pub

    def test_no_label_and_no_paper_gives_a_plain_no_first(self):
        text = self.qc.run(["drugone", "drugtwo"], "autism", *self.lookups())
        self.assertTrue(text.startswith("**Quick answer: no."))
        self.assertIn("**drugone:** no FDA label names autism", text)
        self.assertTrue(text.rstrip().endswith("---"))

    def test_a_label_or_a_paper_is_reported_with_its_evidence(self):
        text = self.qc.run(["drugone", "drugtwo", "drugthree"], "autism", *self.lookups({"drugone": "Indicated for irritability associated with autistic disorder."}, {"drugtwo": ["drugtwo in children with autism"]}))
        self.assertIn("for 1 of these 3 drugs an FDA label names autism", text)
        self.assertIn("**drugone:** yes", text)
        self.assertIn("**drugtwo:** not FDA-labelled for autism; 1 paper(s)", text)
        self.assertIn("**drugthree:** no FDA label", text)

    def test_a_paper_that_does_not_mention_the_drug_is_not_counted(self):
        text = self.qc.run(["drugone"], "autism", *self.lookups(papers={"drugone": ["autism interventions in general"]}))
        self.assertIn("Quick answer: no.", text)

    def test_a_failed_lookup_is_said_not_turned_into_no(self):
        def boom(*a):
            raise RuntimeError("down")
        text = self.qc.run(["drugone"], "autism", boom, boom)
        self.assertIn("could not be checked", text)
        self.assertNotIn("Quick answer: no.", text)

    def test_request_only_when_the_person_names_another_condition(self):
        ok = m.resolve("are those drugs also used to treat autism?", self.MEM1, SAYS(self.REPLY))
        self.assertEqual(ok["quick_check"], {"drugs": ["drugone", "drugtwo", "drugthree"], "condition": "autism"})
        invented = self.REPLY.replace('"own_condition": "autism"', '"own_condition": "epilepsy"')
        self.assertIsNone(m.resolve("are those drugs also used to treat autism?", self.MEM1, SAYS(invented))["quick_check"])
        dosage = '{"mode": "research", "turns": "all", "ask": "Give the dosage information", "needs_drugs": true}'
        self.assertIsNone(m.resolve("give the dosage of the above drugs", self.MEM1, SAYS(dosage))["quick_check"])

    def test_the_api_sends_the_quick_answer_before_the_research(self):
        import api

        def stream_answer(q):
            yield {"type": "token", "text": "# Report"}
            yield {"type": "final", "markdown": "## Clinical summary\nS", "timings": {}, "state": {"entities": {"condition": "c"}, "listed_drugs": []}}

        pipeline = lambda: (stream_answer, lambda q, f: iter([{"type": "pdf", "name": "r.pdf"}]))
        with tempfile.TemporaryDirectory() as d, mock.patch.object(m, "MEMORY_DIR", Path(d)), mock.patch("server.save_conversation"), mock.patch("server.scrub_messages", side_effect=lambda x: x):
            m.save("chat-5", self.MEM1)
            body = api.AskBody(question="are those drugs also used to treat autism?", conversation_id="chat-5")
            out = [json.loads(x) for x in api.ask_events(body, pipeline, SAYS(self.REPLY), None, self.lookups())]
            types = [e["type"] for e in out]
            self.assertLess(types.index("preface"), types.index("token"))
            self.assertTrue(out[types.index("preface")]["text"].startswith("**Quick answer: no."))
            self.assertTrue(m.load("chat-5")[-1]["markdown"].startswith("**Quick answer"))


class OrchestratorAgent(unittest.TestCase):
    import orchestrator as orch

    def agent_that_calls(self, name, **kwargs):
        def run(system, prompt, tools):
            by_name = {f.__name__ if hasattr(f, "__name__") else f.name: f for f in tools}
            fn = by_name[name]
            getattr(fn, "entrypoint", fn)(**kwargs)
        return run

    def test_instructions_come_from_the_orchestrator_skill(self):
        text = self.orch.instructions()
        self.assertIn("Choose the skills that expand the question", text)
        self.assertIn("answer_from_conversation", text)

    def test_follow_up_research_is_composed_from_the_stored_drugs(self):
        r = self.orch.decide("give me the dosage of all the above drugs", MEM, self.agent_that_calls("research", kind="follow_up", ask="Give the dosage information", needs_drugs=True))
        self.assertEqual(r["mode"], "research")
        for word in ("pimozide", "risperidone", "tics in Tourette syndrome"):
            self.assertIn(word, r["question"])

    def test_one_decision_carries_the_plan(self):
        r = self.orch.decide("what about asthma then?", MEM, self.agent_that_calls("research", kind="new", intent="drug_treatment", condition="asthma", info_needed=["treatment_options"], medicines_needed=True, skills=["drug-intelligence"]))
        self.assertEqual(r["plan"]["condition"], "asthma")
        self.assertEqual(r["plan"]["intent"], "drug_treatment")
        self.assertTrue(r["plan"]["medicines_needed"])

    def test_a_follow_up_plan_gets_its_drugs_and_conditions_from_the_stored_answers_not_the_agent(self):
        r = self.orch.decide("give me the dosage of all the above drugs", MEM, self.agent_that_calls("research", kind="follow_up", ask="Give the dosage information", needs_drugs=True, intent="drug_treatment", drugs=["inventedmab"], info_needed=["dosing"]))
        self.assertIn("pimozide", r["plan"]["drugs"])
        self.assertNotIn("inventedmab", r["plan"]["drugs"])
        self.assertEqual(r["plan"]["info_needed"], ["dosing"])

    def test_a_focused_follow_up_is_marked_focused_and_keeps_only_the_drug_skill(self):
        r = self.orch.decide("give me the dosage of all the above drugs", MEM, self.agent_that_calls("research", kind="follow_up", scope="focused", ask="Give the dosage information", needs_drugs=True, intent="drug_treatment", info_needed=["dosing"], skills=["latest-evidence", "drug-intelligence"]))
        self.assertEqual(r["plan"]["focus"], ["drugs"])
        self.assertEqual(r["plan"]["skills"], ["drug-intelligence"])
        self.assertIn("pimozide", r["plan"]["drugs"])

    def test_a_dosing_follow_up_is_focused_even_when_the_agent_leaves_scope_empty(self):
        r = self.orch.decide("give me the dosage of all the above drugs", MEM, self.agent_that_calls("research", kind="follow_up", ask="Give the dosage information", needs_drugs=True, info_needed=["dosing"], skills=["latest-evidence", "drug-intelligence"]))
        self.assertEqual(r["plan"]["focus"], ["drugs"])

    def test_drugs_the_agent_wrote_into_ask_are_not_added_twice(self):
        r = self.orch.decide("dosages of the above drugs", MEM, self.agent_that_calls("research", kind="follow_up", ask="What are the main dosages for pimozide and risperidone", needs_drugs=True, info_needed=["dosing"]))
        self.assertEqual(r["question"].lower().count("pimozide"), 1)

    def test_a_full_scope_follow_up_or_one_that_changes_the_condition_is_not_focused(self):
        full = self.orch.decide("give me the dosage of all the above drugs", MEM, self.agent_that_calls("research", kind="follow_up", scope="full", ask="Give the dosage information", needs_drugs=True, info_needed=["dosing"]))
        self.assertNotIn("focus", full["plan"])
        other = self.orch.decide("dosage of those in adults", MEM, self.agent_that_calls("research", kind="follow_up", scope="focused", ask="Give the dosage", needs_drugs=True, own_condition="tics in adults", info_needed=["dosing"]))
        self.assertEqual(other["plan"]["focus"], ["drugs"])      # it points back at the earlier drugs: a follow-up even though it names another population
        new_subject = self.orch.decide("treatment of tics in adults", MEM, self.agent_that_calls("research", kind="follow_up", scope="focused", ask="Treatment", needs_drugs=False, own_condition="tics in adults", info_needed=["dosing"]))
        self.assertNotIn("focus", new_subject["plan"])      # it does not point back at the earlier drugs: the whole report

    def test_the_plan_alone_is_asked_when_no_routing_was_needed(self):
        plan = self.orch.plan_only("Treatment of migraine in adults", self.agent_that_calls("research", kind="new", intent="drug_treatment", condition="migraine"))
        self.assertEqual(plan["condition"], "migraine")
        self.assertIsNone(self.orch.plan_only("hello", lambda *a: None))

    def test_without_memory_the_agent_has_only_the_research_tool(self):
        seen = []
        self.orch.plan_only("Treatment of migraine", lambda system, prompt, tools: seen.append([getattr(t, "__name__", "") for t in tools]))
        self.assertEqual(seen, [["research"]])

    def test_a_follow_up_without_ask_is_still_rewritten_with_the_stored_drugs(self):
        r = self.orch.decide("for the drugs above given what are the doses??", MEM, self.agent_that_calls("research", kind="follow_up", needs_drugs=True, intent="drug_treatment", condition="above given doses", info_needed=["dosing"]))
        self.assertEqual((r["mode"], r["follow_up"]), ("research", True))
        for word in ("pimozide", "risperidone", "tics in Tourette syndrome"):
            self.assertIn(word, r["question"])
        self.assertNotEqual(r["plan"]["condition"], "above given doses")
        self.assertIn("pimozide", r["plan"]["drugs"])

    def test_drugs_the_agent_wrote_for_a_follow_up_are_dropped_when_they_are_not_needed(self):
        r = self.orch.decide("what are the side effects?", MEM, self.agent_that_calls("research", kind="follow_up", ask="Give the side effects", needs_drugs=False, drugs=["inventedmab"]))
        self.assertEqual(r["plan"]["drugs"], [])

    def test_conversation_choice(self):
        r = self.orch.decide("explain that simply", MEM, self.agent_that_calls("answer_from_conversation", turns="1", ask="Explain simply"))
        self.assertEqual((r["mode"], r["turns"]), ("conversation", [1]))

    def test_new_question_is_left_as_asked(self):
        r = self.orch.decide("what about asthma then?", MEM, self.agent_that_calls("research", kind="new"))
        self.assertEqual((r["mode"], r["question"]), ("new", "what about asthma then?"))

    def test_clarification(self):
        r = self.orch.decide("what about it?", MEM, self.agent_that_calls("ask_for_clarification", question="Which drug do you mean?"))
        self.assertEqual((r["mode"], r["clarification"]), ("clarify", "Which drug do you mean?"))

    def test_other_condition_triggers_the_quick_check_request(self):
        mem = [{"question": "q", "condition": "c", "drugs": ["drugone"], "markdown": "x", "kind": "report"}]
        r = self.orch.decide("are those also used to treat autism?", mem, self.agent_that_calls("research", kind="follow_up", ask="Are they also used to treat autism", needs_drugs=True, own_condition="autism"))
        self.assertEqual(r["quick_check"], {"drugs": ["drugone"], "condition": "autism"})

    def test_agent_failure_or_no_choice_falls_back_to_the_json_router(self):
        def boom(*a):
            raise RuntimeError("agent down")
        r = self.orch.decide("give me the dosage of all the above drugs", MEM, boom, SAYS(FOLLOW))
        self.assertEqual(r["mode"], "research")
        r = self.orch.decide("give me the dosage of all the above drugs", MEM, lambda *a: None, SAYS(FOLLOW))
        self.assertEqual(r["mode"], "research")

    def test_no_memory_costs_no_agent_call(self):
        calls = []
        self.orch.decide("explain that", [], lambda *a: calls.append(1))
        self.assertEqual(calls, [])

    def test_the_api_answers_a_clarification_without_research(self):
        import api
        researched = []
        pipeline = lambda: ((lambda q: researched.append(q) or iter([])), lambda q, f: iter([]))
        with tempfile.TemporaryDirectory() as d, mock.patch.object(m, "MEMORY_DIR", Path(d)), mock.patch("server.save_conversation"), mock.patch("server.scrub_messages", side_effect=lambda x: x):
            m.save("chat-3", MEM)
            body = api.AskBody(question="what about it?", conversation_id="chat-3")
            out = [json.loads(x) for x in api.ask_events(body, pipeline, None, None, None, self.agent_that_calls("ask_for_clarification", question="Which drug do you mean?"))]
        self.assertEqual(researched, [])
        self.assertEqual(out[-1]["markdown"], "Which drug do you mean?")


class ApiConversation(unittest.TestCase):
    def run_ask(self, chunks, question="explain that simply"):
        import api
        researched = []

        def stream_answer(q):
            researched.append(q)
            yield {"type": "final", "markdown": "## Clinical summary\nS", "timings": {}, "state": {"entities": {"condition": "c"}, "listed_drugs": []}}

        pipeline = lambda: (stream_answer, lambda q, f: iter([{"type": "pdf", "name": "r.pdf"}]))
        with tempfile.TemporaryDirectory() as d, mock.patch.object(m, "MEMORY_DIR", Path(d)), mock.patch("server.save_conversation"), mock.patch("server.scrub_messages", side_effect=lambda x: x):
            m.save("chat-7", FULL)
            body = api.AskBody(question=question, conversation_id="chat-7")
            out = [json.loads(x) for x in api.ask_events(body, pipeline, SAYS(CONVERSATION), lambda i, p: iter(chunks))]
            stored = m.load("chat-7")
        return out, researched, stored

    def test_answered_from_the_conversation_without_research(self):
        out, researched, stored = self.run_ask(["In short, ", "risperidone is used for aggression ", "at 0.5 mg."])
        types = [e["type"] for e in out]
        self.assertEqual(researched, [])
        self.assertEqual(types[:2], ["scrubbed", "understood"])
        self.assertEqual(types[-1], "final")
        self.assertEqual(out[-1]["mode"], "chat")
        self.assertNotIn("pdf", types)
        self.assertEqual("".join(e["text"] for e in out if e["type"] == "token"), out[-1]["markdown"])
        self.assertEqual(stored[-1]["kind"], "chat")

    def test_text_is_sent_while_it_is_written(self):
        out, _, _ = self.run_ask(["Alpha beta gamma delta ", "epsilon ", "zeta."])
        self.assertGreaterEqual(len([e for e in out if e["type"] == "token"]), 2)

    def test_an_unmatched_number_is_flagged(self):
        out, _, _ = self.run_ask(["Risperidone is given at 77 mg per day in all cases."])
        self.assertIn("77", out[-1]["markdown"].split("could not be checked")[-1])

    def test_missing_information_falls_back_to_research(self):
        out, researched, _ = self.run_ask(["NEEDS_", "RESEARCH"])
        self.assertEqual(len(researched), 1)
        self.assertEqual(out[-1]["type"], "pdf")
        self.assertNotIn("chat", [e.get("mode") for e in out])



class NamedDrugSearches(unittest.TestCase):
    def test_every_named_drug_gets_a_label_search_six_to_a_search(self):
        import qa_stream as q
        import retrieval_routing as rr
        question = "Give the dosage for risperidone, aripiprazole, haloperidol, pimozide, clonidine, guanfacine, topiramate, tetrabenazine used for tics?"
        plan = {"drugs": ["risperidone", "aripiprazole", "haloperidol", "pimozide", "clonidine", "guanfacine", "topiramate", "tetrabenazine", "notinquestion"]}
        ops = [rr.Operation(id="a", need_id="a", need="n", target="fda_label", args={"condition_terms": ["tics"], "drug_names": ["risperidone", "aripiprazole"]})]
        new = q.named_drug_operations(question, plan, ops)
        self.assertEqual([o.args["drug_names"] for o in new], [["haloperidol", "pimozide", "clonidine", "guanfacine", "topiramate", "tetrabenazine"]])
        self.assertEqual(q.named_drug_operations(question, {"drugs": ["risperidone"]}, ops), [])


if __name__ == "__main__":
    unittest.main()


class CaseMemory(unittest.TestCase):
    def test_a_turn_keeps_the_case_the_conclusions_and_where_they_came_from(self):
        final = {"markdown": "## Clinical summary\n\n**Bottom line** Stop the combined pill because of stroke risk with aura. Use a progestogen-only method instead.\n\n## Other\n",
                 "state": {"entities": {"condition": "x", "associated_findings": ["on a combined pill"], "main_problem": "aura"}, "patient_age_years": 34.0,
                           "understanding": {"case_facts": ["34-year-old woman", "aura"], "purpose": "decide"}}}
        turn = m.turn_from_final("Q1 text", "Q1 text", final)
        self.assertEqual(turn["case"]["facts"], ["34-year-old woman", "aura", "on a combined pill"])
        self.assertEqual(turn["case"]["age_years"], 34.0)
        self.assertTrue(turn["conclusions"] and "stroke risk" in turn["conclusions"][0])
        follow = dict(m.turn_from_final("Q2", "Q2", {"markdown": "x", "state": {"understanding": {"case_facts": ["for her"]}}}, follow_up=True))
        profile = m.case_profile([turn, follow])
        self.assertEqual([f["text"] for f in profile["facts"]], ["34-year-old woman", "aura", "on a combined pill"])      # a follow-up adds no facts of its own
        self.assertTrue(all(f["from"] == 1 for f in profile["facts"]))
        self.assertIn("question 1", m.resolve_payload("alternatives for her?", [turn, follow]))

    def test_the_composed_question_carries_the_case(self):
        turn = {"question": "Q1", "case": {"facts": ["34-year-old woman", "aura"]}, "drugs": [], "condition": "migraine", "kind": "report"}
        text = m._research_question("What are the alternatives to the pill for her", [turn], [1], {"needs_drugs": False}, "q")
        self.assertIn("34-year-old woman", text)


class AskedDrugs(unittest.TestCase):
    def test_a_follow_up_about_both_means_the_drugs_the_question_named(self):
        final = {"markdown": "## Clinical summary\n\nx\n", "state": {"entities": {"condition": "obesity", "drugs": ["semaglutide", "tirzepatide", "metformin"]}, "listed_drugs": ["semaglutide", "tirzepatide", "metformin", "orlistat"]}}
        turn = m.turn_from_final("Compare semaglutide and tirzepatide for weight loss", "Compare semaglutide and tirzepatide for weight loss", final)
        self.assertEqual(turn["asked_drugs"], ["semaglutide", "tirzepatide"])      # metformin and orlistat were listed by the report, not asked about
        self.assertEqual(m.subject_drugs(turn), ["semaglutide", "tirzepatide"])
        text = m._research_question("What are the contraindications of both", [turn], [1], {"needs_drugs": True}, "q")
        self.assertIn("semaglutide", text)
        self.assertNotIn("orlistat", text)
        self.assertIn("the drugs the question named: semaglutide, tirzepatide", m.resolve_payload("contraindications of both?", [turn]))

    def test_a_question_about_a_condition_uses_the_drugs_its_answer_listed(self):
        final = {"markdown": "x", "state": {"entities": {"condition": "tics", "drugs": []}, "listed_drugs": ["risperidone", "aripiprazole"]}}
        turn = m.turn_from_final("How to treat tics in children?", "How to treat tics in children?", final)
        self.assertEqual(turn["asked_drugs"], [])
        self.assertEqual(m.subject_drugs(turn), ["risperidone", "aripiprazole"])


class AttributesDecideTheAnswer(unittest.TestCase):
    def test_the_facts_the_agent_names_decide_the_part_and_replace_the_fixed_asks(self):
        import orchestrator as o
        plan = {"info_needed": ["safety", "dosing"]}      # the agent over-listed: the message asked only about contraindications
        self.assertEqual(o.focus_sections(plan, {"attributes": ["contraindications"]}, True, False), ["drug_facts"])
        self.assertEqual(o.focus_sections({"info_needed": ["efficacy"]}, {"attributes": "use in pregnancy, monitoring"}, True, False), ["drug_facts", "evidence"])
        self.assertEqual(o.focus_sections(plan, {}, True, False), ["safety", "drugs"])      # without attributes the earlier behaviour
        self.assertEqual(o.focus_sections({"info_needed": ["efficacy"]}, {"attributes": ["contraindications"]}, False, False), ["evidence"])      # no drugs in memory: nothing to build the table from
        self.assertEqual(o.clean_attributes("contraindications; what to monitor, "), ["contraindications", "what to monitor"])
