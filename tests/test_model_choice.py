import os
import sys
import types
import unittest
from unittest import mock


class Recorder:
    def __init__(self, **kw):
        self.kw = kw


class ModelChoice(unittest.TestCase):
    def build(self, model, temperature=0.0, **env):
        fake = types.ModuleType("agno.models.openai")
        fake.OpenAIChat = Recorder
        with mock.patch.dict(sys.modules, {"agno": types.ModuleType("agno"), "agno.models": types.ModuleType("agno.models"), "agno.models.openai": fake}), \
                mock.patch.dict(os.environ, {"MODEL_PROVIDER": "openai", "OPENAI_API_KEY": "x", "OPENAI_MODEL": model, **env}, clear=False):
            import server
            return server.build_model(max_tokens=500, temperature=temperature).kw

    def test_gpt_5_uses_max_completion_tokens_and_no_thinking_and_no_temperature(self):
        kw = self.build("gpt-5.1")
        self.assertEqual((kw["id"], kw["max_completion_tokens"], kw["reasoning_effort"]), ("gpt-5.1", 500, "none"))
        self.assertNotIn("temperature", kw)
        self.assertNotIn("max_tokens", kw)

    def test_temperature_is_sent_only_when_asked(self):
        self.assertEqual(self.build("gpt-5.1", OPENAI_GPT5_TEMPERATURE="on")["temperature"], 0.0)

    def test_gpt_4_1_mini_is_built_as_before(self):
        kw = self.build("gpt-4.1-mini")
        self.assertEqual((kw["max_tokens"], kw["temperature"]), (500, 0.0))
        self.assertNotIn("reasoning_effort", kw)


if __name__ == "__main__":
    unittest.main()
