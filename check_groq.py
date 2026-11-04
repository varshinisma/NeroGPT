"""Shows, for each Groq model, the tokens-per-minute limit your current key gets. Prints no key."""
import os, requests
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("GROQ_API_KEY", "").strip().strip('"')
for m in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"]:
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=30,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": m, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    h = r.headers
    print(m, r.status_code, "limit tokens/min:", h.get("x-ratelimit-limit-tokens"), "remaining:", h.get("x-ratelimit-remaining-tokens"), "reqs/day:", h.get("x-ratelimit-limit-requests"))
