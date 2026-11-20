"""Follow-up or new question? The orchestrator agent chooses, and two checks of code stand beside its choice: a message that names its own subject and shares nothing with the earlier answers is a new question
(however the agent labelled it), and a message that points back but was called new is put to the agent once more. The cases are the ones that went wrong in saved chats."""
import asyncio
import unittest

import async_orchestrator
import conversation_memory as m
import orchestrator

PANDAS = {"question": "What is the current treatment of PANDAs?", "asked_as": "What is the current treatment of PANDAS?", "condition": "PANDAS", "drugs": ["penicillin V", "azithromycin", "ibuprofen"], "kind": "report"}
INSOMNIA = {"question": "What medication is used for insomnia in children?", "condition": "insomnia in children", "drugs": ["melatonin", "clonidine"], "kind": "report"}
MIGRAINE = {"question": "Migraine prophylaxis 2026 evidence", "condition": "migraine", "drugs": ["propranolol", "topiramate", "erenumab"], "kind": "report"}


def agent(*calls):
    """A scripted agent: each call of the agent takes the next (tool, kwargs)."""
    queue, seen = list(calls), []

    def run(system, prompt, tools):
        seen.append(prompt)
        name, kwargs = queue.pop(0) if len(queue) > 1 else queue[0]
        fn = {f.__name__ if hasattr(f, "__name__") else f.name: f for f in tools}[name]
        getattr(fn, "entrypoint", fn)(**kwargs)
    run.seen = seen
    return run


FOLLOW = ("research", dict(kind="follow_up", ask="Give the information asked", needs_drugs=False, intent="disease_research"))
NEW = ("research", dict(kind="new", intent="disease_research", condition="x"))


class Standalone(unittest.TestCase):
    def test_new_questions_that_were_merged_with_the_earlier_subject(self):
        for question, memory in (("What medication is used for insomnia in children?", [PANDAS]), ("Migraine prophylaxis 2026 evidence", [PANDAS, INSOMNIA]),
                                 ("What are the medications used to control tics in Tourette syndrome?", [PANDAS, INSOMNIA, MIGRAINE])):
            self.assertTrue(m.standalone(question, memory), question)

    def test_real_follow_ups_are_never_judged_standalone(self):
        for question in ("What are the side effects of the above given drugs?", "Give me the dosage information of the above given drugs!!", "dosage?", "What about in pregnancy?",
                         "Is azithromycin safe in pregnancy?", "How long is penicillin V given?", "prognosis", "What are the side effects and monitoring needed?", "And the doses for those in adults", "What are the treatment options?", "Compare them in a table"):
            self.assertFalse(m.standalone(question, [PANDAS]), question)

    def test_the_same_subject_again_is_not_standalone(self):
        self.assertFalse(m.standalone("What is the prognosis of PANDAS in adolescents?", [PANDAS]))      # shares the subject word
        self.assertFalse(m.standalone("Latest trials for insomnia medication in kids", [INSOMNIA]))

    def test_no_conversation_means_standalone(self):
        self.assertTrue(m.standalone("anything at all here", []))

    def test_refers_back(self):
        self.assertTrue(m.refers_back("Give the dosage of the above drugs"))
        self.assertTrue(m.refers_back("what about those in adults"))
        self.assertFalse(m.refers_back("Migraine prophylaxis 2026 evidence"))


class DecideWithTheCheck(unittest.TestCase):
    def test_a_new_question_called_a_follow_up_is_researched_as_new_and_unchanged(self):
        r = orchestrator.decide("Migraine prophylaxis 2026 evidence", [INSOMNIA], agent(FOLLOW))
        self.assertEqual((r["follow_up"], r["mode"], r["question"], r["turns"]), (False, "new", "Migraine prophylaxis 2026 evidence", []))
        self.assertNotIn("plan", r)      # the planner writes a fresh plan for it
        self.assertIn("standalone", r["routing_note"])

    def test_a_real_follow_up_stays_a_follow_up(self):
        r = orchestrator.decide("give me the side effects of the above drugs", [INSOMNIA], agent(("research", dict(kind="follow_up", ask="Give the side effects", needs_drugs=True))))
        self.assertEqual(r["mode"], "research")
        self.assertTrue(r["follow_up"])
        self.assertIn("melatonin", r["question"])

    def test_a_message_that_points_back_but_was_called_new_is_asked_again(self):
        run = agent(NEW, ("research", dict(kind="follow_up", ask="Give the dosage information", needs_drugs=True)))
        r = orchestrator.decide("Give the dosage information of the above drugs", [INSOMNIA], run)
        self.assertEqual(len(run.seen), 2)
        self.assertIn("CHECK: this message points back", run.seen[1])
        self.assertTrue(r["follow_up"])
        self.assertIn("clonidine", r["question"])

    def test_if_the_agent_still_says_new_the_question_is_new(self):
        run = agent(NEW)
        r = orchestrator.decide("What happens to those with epilepsy and melatonin", [INSOMNIA], run)
        self.assertEqual(len(run.seen), 2)
        self.assertEqual((r["follow_up"], r["mode"]), (False, "new"))

    def test_a_clearly_new_question_makes_no_routing_call(self):
        run = agent(NEW)
        r = orchestrator.decide("Migraine prophylaxis 2026 evidence", [INSOMNIA], run)
        self.assertEqual(len(run.seen), 0)      # decided by code: no wait before the research
        self.assertEqual((r["follow_up"], r["mode"]), (False, "new"))
        self.assertIn("no routing call", r["routing_note"])

    def test_instructions_on_the_answer_still_go_to_the_agent(self):
        for q in ("explain that simply", "summarise this in a table", "make a handout for parents", "tell me more about safety in kids", "thanks"):
            run = agent(("answer_from_conversation", dict(turns="1", ask=q)))
            orchestrator.decide(q, [INSOMNIA], run)
            self.assertEqual(len(run.seen), 1, q)

    def test_answering_from_the_conversation_is_not_overruled(self):
        r = orchestrator.decide("explain that simply", [INSOMNIA], agent(("answer_from_conversation", dict(turns="1", ask="Explain simply"))))
        self.assertEqual(r["mode"], "conversation")

    def test_the_async_twin_decides_the_same_way(self):
        async def arun(system, prompt, tools):
            name, kwargs = FOLLOW
            fn = {f.__name__ if hasattr(f, "__name__") else f.name: f for f in tools}[name]
            getattr(fn, "entrypoint", fn)(**kwargs)
        r = asyncio.run(async_orchestrator.adecide("Migraine prophylaxis 2026 evidence", [INSOMNIA], arun))
        self.assertEqual((r["follow_up"], r["mode"], r["question"]), (False, "new", "Migraine prophylaxis 2026 evidence"))

    def test_the_guide_tells_the_agent_that_new_is_the_default(self):
        text = orchestrator.instructions()
        self.assertIn("NEW QUESTION OR FOLLOW-UP?", text)
        self.assertIn("The default is a NEW question", text)


if __name__ == "__main__":
    unittest.main()
