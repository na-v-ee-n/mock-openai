"""
Client example using ONLY the Python standard library (no installs needed).

Make sure the mock server is running first:
    python mock_openai_server.py

Then run:
    python client_stdlib.py
"""

import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = "mock-key"  # the mock server accepts anything


def post(path, payload):
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def demo_chat():
    print("\n--- chat.completions ---")
    data = post(
        "/chat/completions",
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful tutor."},
                {"role": "user", "content": "Explain what an embedding is."},
            ],
        },
    )
    print("Reply :", data["choices"][0]["message"]["content"])
    print("Usage :", data["usage"])


def demo_embeddings():
    print("\n--- embeddings ---")
    data = post(
        "/embeddings",
        {"model": "text-embedding-3-small", "input": "hello world"},
    )
    vec = data["data"][0]["embedding"]
    print("Vector length:", len(vec))
    print("First 5 dims :", vec[:5])


def demo_stream():
    print("\n--- chat.completions (streaming) ---")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Stream me a sentence."}],
                "stream": True,
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content", "")
            print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    demo_chat()
    demo_embeddings()
    demo_stream()
