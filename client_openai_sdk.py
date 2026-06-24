"""
Client example using the official `openai` SDK pointed at the mock server.

This is the SAME code you'd use against real OpenAI. To switch to the real
API later, just change base_url back to the default and use your real key.

Setup (one-time):
    python -m pip install openai

Run (with the mock server running):
    python client_openai_sdk.py
"""

from openai import OpenAI

# The only difference from real OpenAI usage is base_url pointing to localhost.
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="mock-key",  # any string works for the mock
)


def demo_chat():
    print("\n--- chat.completions ---")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful tutor."},
            {"role": "user", "content": "Explain tokens in one sentence."},
        ],
    )
    print("Reply :", resp.choices[0].message.content)
    print("Usage :", resp.usage)


def demo_stream():
    print("\n--- chat.completions (streaming) ---")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Stream a short greeting."}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()


def demo_embeddings():
    print("\n--- embeddings ---")
    resp = client.embeddings.create(
        model="text-embedding-3-small", input="hello world"
    )
    print("Vector length:", len(resp.data[0].embedding))


if __name__ == "__main__":
    demo_chat()
    demo_stream()
    demo_embeddings()
