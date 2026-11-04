import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import api
import conversation_memory


def events_of(lines):
    return [json.loads(x) for x in lines]


def fake_pipeline(fail=False):
    def stream_answer(question):
        yield {"type": "stage", "stage": "INITIAL"}
        yield {"type": "token", "text": "Hello " + question}
        if fail:
            raise RuntimeError("model down")
        yield {"type": "final", "markdown": "# Report", "timings": {}, "state": {"verification": {"status": "VERIFIED"}, "secret": "x" * 50}}

    def deliver_outputs(question, final):
        yield {"type": "figures", "items": []}
        yield {"type": "pdf", "name": "r.pdf"}
    return lambda: (stream_answer, deliver_outputs)


class AskEvents(unittest.TestCase):
    def run_ask(self, **kw):
        body = api.AskBody(question="  What treats it?  ", history=[{"role": "user", "content": "earlier"}], conversation_id="chat-1")
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(conversation_memory, "MEMORY_DIR", Path(folder)), \
                mock.patch("server.save_conversation") as save, mock.patch("server.scrub_messages", side_effect=lambda m: m):      # never the real memory folder
            out = events_of(api.ask_events(body, fake_pipeline(**kw)))
        return out, save

    def test_event_order_and_state_stays_on_the_server(self):
        out, save = self.run_ask()
        self.assertEqual([e["type"] for e in out], ["scrubbed", "stage", "token", "final", "figures", "pdf"])
        self.assertEqual(out[0]["text"], "What treats it?")
        final = out[3]
        self.assertNotIn("state", final)
        self.assertEqual(final["verification_status"], "VERIFIED")
        self.assertEqual(final["markdown"], "# Report")

    def test_conversation_saved_once_with_history(self):
        out, save = self.run_ask()
        save.assert_called_once()
        messages, answer, conversation_id = save.call_args[0]
        self.assertEqual(messages[0]["content"], "earlier")
        self.assertEqual(messages[-1], {"role": "user", "content": "What treats it?"})
        self.assertEqual((answer, conversation_id), ("# Report", "chat-1"))

    def test_pipeline_failure_becomes_an_error_event(self):
        out, save = self.run_ask(fail=True)
        self.assertEqual(out[-1]["type"], "error")
        self.assertIn("model down", out[-1]["message"])
        save.assert_not_called()


class Endpoints(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(api.app)

    def test_health(self):
        reply = self.client.get("/health")
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.json()["status"], "ok")

    def test_ask_streams_ndjson(self):
        with mock.patch("api.pipeline", fake_pipeline()), mock.patch("server.save_conversation"), mock.patch("server.scrub_messages", side_effect=lambda m: m):
            reply = self.client.post("/ask", json={"question": "hello"})
        self.assertEqual(reply.status_code, 200)
        self.assertIn("ndjson", reply.headers["content-type"])
        types = [json.loads(x)["type"] for x in reply.text.splitlines() if x]
        self.assertEqual(types[0], "scrubbed")
        self.assertEqual(types[-1], "pdf")

    def test_empty_question_rejected(self):
        self.assertEqual(self.client.post("/ask", json={"question": ""}).status_code, 422)

    def test_pdf_name_is_checked(self):
        self.assertEqual(self.client.get("/pdf/..%2F.env").status_code, 404)
        self.assertEqual(self.client.get("/pdf/missing.pdf").status_code, 404)
        self.assertEqual(self.client.get("/pdf/notpdf.txt").status_code, 400)

    def test_file_only_serves_figure_images(self):
        self.assertEqual(self.client.get("/file", params={"path": ".env"}).status_code, 404)
        self.assertEqual(self.client.get("/file", params={"path": "demo_output/figures/../../.env"}).status_code, 404)
        self.assertEqual(self.client.get("/file", params={"path": "demo_output/figures/none.png"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
