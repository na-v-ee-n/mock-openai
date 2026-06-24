"""
Smoke-test every mock endpoint using only the Python standard library.

Run with the server already running:
    python test_all_endpoints.py

Exit code: 0 if all assertions pass, 1 if any fail.
"""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000/v1"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer mock-key"}

_passed = 0
_failed = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=HDR,
                                 method=method)
    with urllib.request.urlopen(req) as r:
        ctype = r.headers.get("Content-Type", "")
        raw = r.read()
        return raw if "json" not in ctype else json.loads(raw)


def call_expect_404(method, path, payload=None):
    """Assert the server returns HTTP 404 for a missing resource."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=HDR,
                                 method=method)
    try:
        urllib.request.urlopen(req)
        return None  # unexpectedly succeeded
    except urllib.error.HTTPError as exc:
        return exc.code


def check(label, condition, detail=""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {label}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_core():
    section("Core endpoints")

    resp = call("POST", "/chat/completions",
                {"model": "gpt-4o-mini",
                 "messages": [{"role": "user", "content": "hi"}]})
    check("chat.completions returns object=chat.completion",
          resp.get("object") == "chat.completion")
    check("chat.completions has choices list",
          isinstance(resp.get("choices"), list) and len(resp["choices"]) > 0)
    check("chat.completions has usage block",
          isinstance(resp.get("usage"), dict))
    check("chat.completions has id starting with chatcmpl",
          str(resp.get("id", "")).startswith("chatcmpl"))

    resp = call("POST", "/chat/completions",
                {"model": "gpt-4o-mini",
                 "messages": [{"role": "user", "content": "weather?"}],
                 "tools": [{"type": "function",
                            "function": {"name": "get_weather",
                                         "parameters": {}}}]})
    check("chat+tools finish_reason=tool_calls",
          resp["choices"][0].get("finish_reason") == "tool_calls")
    check("chat+tools message has tool_calls",
          isinstance(resp["choices"][0]["message"].get("tool_calls"), list))

    resp = call("POST", "/completions",
                {"model": "gpt-3.5-turbo", "prompt": "Once upon a time"})
    check("completions returns object=text_completion",
          resp.get("object") == "text_completion")

    resp = call("POST", "/embeddings",
                {"model": "text-embedding-3-small", "input": "hello"})
    check("embeddings returns object=list",
          resp.get("object") == "list")
    check("embeddings data[0] has 1536-dim vector",
          len(resp["data"][0]["embedding"]) == 1536)

    resp = call("GET", "/models")
    check("models.list returns object=list",
          resp.get("object") == "list")
    check("models.list has at least one model",
          isinstance(resp.get("data"), list) and len(resp["data"]) > 0)

    resp = call("GET", "/models/gpt-4o")
    check("models.retrieve returns id=gpt-4o",
          resp.get("id") == "gpt-4o")


def test_multimodal():
    section("Multimodal / audio / moderation")

    resp = call("POST", "/images/generations",
                {"model": "dall-e-3", "prompt": "a cat", "n": 1})
    check("images.generations has data list",
          isinstance(resp.get("data"), list) and len(resp["data"]) == 1)
    check("images.generations data[0] has url",
          "url" in resp["data"][0])

    resp = call("POST", "/audio/speech",
                {"model": "tts-1", "input": "hello", "voice": "alloy"})
    check("audio.speech returns bytes",
          isinstance(resp, bytes) and len(resp) > 0)

    resp = call("POST", "/audio/transcriptions", {"model": "whisper-1"})
    check("audio.transcriptions has text field",
          "text" in resp)

    resp = call("POST", "/audio/translations", {"model": "whisper-1"})
    check("audio.translations has text field",
          "text" in resp)

    resp = call("POST", "/moderations", {"input": "test"})
    check("moderations has results list",
          isinstance(resp.get("results"), list))
    check("moderations result has flagged=False",
          resp["results"][0].get("flagged") is False)


def test_files():
    section("Files")

    f = call("POST", "/files", {})
    check("files.create returns object=file",
          f.get("object") == "file")
    check("files.create id starts with file",
          str(f.get("id", "")).startswith("file"))
    fid = f["id"]

    lst = call("GET", "/files")
    check("files.list object=list",
          lst.get("object") == "list")
    check("files.list contains created file",
          any(x["id"] == fid for x in lst["data"]))

    got = call("GET", f"/files/{fid}")
    check("files.retrieve returns same id",
          got.get("id") == fid)

    content = call("GET", f"/files/{fid}/content")
    check("files.content returns bytes",
          isinstance(content, bytes) and len(content) > 0)

    deleted = call("DELETE", f"/files/{fid}")
    check("files.delete returns deleted=True",
          deleted.get("deleted") is True)

    code = call_expect_404("GET", f"/files/{fid}")
    check("files.retrieve after delete returns 404",
          code == 404, f"got HTTP {code}")


def test_fine_tuning():
    section("Fine-tuning")

    ft = call("POST", "/fine_tuning/jobs",
              {"model": "gpt-4o-mini", "training_file": "file-abc"})
    check("fine_tuning.create object=fine_tuning.job",
          ft.get("object") == "fine_tuning.job")
    ftid = ft["id"]

    got = call("GET", f"/fine_tuning/jobs/{ftid}")
    check("fine_tuning.retrieve returns same id",
          got.get("id") == ftid)

    events = call("GET", f"/fine_tuning/jobs/{ftid}/events")
    check("fine_tuning.events object=list",
          events.get("object") == "list")

    cancelled = call("POST", f"/fine_tuning/jobs/{ftid}/cancel")
    check("fine_tuning.cancel status=cancelled",
          cancelled.get("status") == "cancelled")

    lst = call("GET", "/fine_tuning/jobs")
    check("fine_tuning.list object=list",
          lst.get("object") == "list")

    code = call_expect_404("GET", "/fine_tuning/jobs/ftjob-doesnotexist")
    check("fine_tuning.retrieve non-existent returns 404",
          code == 404, f"got HTTP {code}")


def test_batches():
    section("Batches")

    b = call("POST", "/batches",
             {"input_file_id": "file-abc",
              "endpoint": "/v1/chat/completions",
              "completion_window": "24h"})
    check("batches.create object=batch",
          b.get("object") == "batch")
    bid = b["id"]

    got = call("GET", f"/batches/{bid}")
    check("batches.retrieve returns same id",
          got.get("id") == bid)

    lst = call("GET", "/batches")
    check("batches.list object=list",
          lst.get("object") == "list")

    cancelled = call("POST", f"/batches/{bid}/cancel")
    check("batches.cancel status=cancelling",
          cancelled.get("status") == "cancelling")

    code = call_expect_404("GET", "/batches/batch-doesnotexist")
    check("batches.retrieve non-existent returns 404",
          code == 404, f"got HTTP {code}")


def test_vector_stores():
    section("Vector stores")

    vs = call("POST", "/vector_stores", {"name": "kb"})
    check("vector_stores.create object=vector_store",
          vs.get("object") == "vector_store")
    vsid = vs["id"]

    lst = call("GET", "/vector_stores")
    check("vector_stores.list object=list",
          lst.get("object") == "list")
    check("vector_stores.list contains created store",
          any(x["id"] == vsid for x in lst["data"]))

    got = call("GET", f"/vector_stores/{vsid}")
    check("vector_stores.retrieve returns same id",
          got.get("id") == vsid)

    deleted = call("DELETE", f"/vector_stores/{vsid}")
    check("vector_stores.delete returns deleted=True",
          deleted.get("deleted") is True)

    code = call_expect_404("GET", f"/vector_stores/{vsid}")
    check("vector_stores.retrieve after delete returns 404",
          code == 404, f"got HTTP {code}")


def test_assistants():
    section("Assistants")

    a = call("POST", "/assistants",
             {"model": "gpt-4o-mini", "name": "Tutor"})
    check("assistants.create object=assistant",
          a.get("object") == "assistant")
    check("assistants.create name=Tutor",
          a.get("name") == "Tutor")
    aid = a["id"]

    lst = call("GET", "/assistants")
    check("assistants.list object=list",
          lst.get("object") == "list")

    got = call("GET", f"/assistants/{aid}")
    check("assistants.retrieve returns same id",
          got.get("id") == aid)

    patched = call("PATCH", f"/assistants/{aid}", {"name": "Senior Tutor"})
    check("assistants.patch updates name",
          patched.get("name") == "Senior Tutor")

    deleted = call("DELETE", f"/assistants/{aid}")
    check("assistants.delete returns deleted=True",
          deleted.get("deleted") is True)

    code = call_expect_404("GET", f"/assistants/{aid}")
    check("assistants.retrieve after delete returns 404",
          code == 404, f"got HTTP {code}")


def test_threads_messages_runs():
    section("Threads / Messages / Runs")

    t = call("POST", "/threads", {})
    check("threads.create object=thread",
          t.get("object") == "thread")
    tid = t["id"]

    got = call("GET", f"/threads/{tid}")
    check("threads.retrieve returns same id",
          got.get("id") == tid)

    patched = call("PATCH", f"/threads/{tid}",
                   {"metadata": {"tag": "test"}})
    check("threads.patch updates metadata",
          patched.get("metadata", {}).get("tag") == "test")

    msg = call("POST", f"/threads/{tid}/messages",
               {"role": "user", "content": "Explain RAG"})
    check("messages.create object=thread.message",
          msg.get("object") == "thread.message")
    check("messages.create thread_id matches",
          msg.get("thread_id") == tid)

    a = call("POST", "/assistants", {"model": "gpt-4o-mini"})
    run = call("POST", f"/threads/{tid}/runs",
               {"assistant_id": a["id"]})
    check("runs.create object=thread.run",
          run.get("object") == "thread.run")
    check("runs.create status=completed",
          run.get("status") == "completed")
    rid = run["id"]

    msgs = call("GET", f"/threads/{tid}/messages")
    check("messages.list after run has >= 2 messages",
          len(msgs.get("data", [])) >= 2)

    runs_lst = call("GET", f"/threads/{tid}/runs")
    check("runs.list object=list",
          runs_lst.get("object") == "list")

    got_run = call("GET", f"/threads/{tid}/runs/{rid}")
    check("runs.retrieve returns same id",
          got_run.get("id") == rid)

    deleted = call("DELETE", f"/threads/{tid}")
    check("threads.delete returns deleted=True",
          deleted.get("deleted") is True)

    code = call_expect_404("GET", f"/threads/{tid}")
    check("threads.retrieve after delete returns 404",
          code == 404, f"got HTTP {code}")


def test_responses_api():
    section("Responses API")

    resp = call("POST", "/responses",
                {"model": "gpt-4o-mini", "input": "Say hello"})
    check("responses.create object=response",
          resp.get("object") == "response")
    check("responses.create status=completed",
          resp.get("status") == "completed")
    check("responses.create has output list",
          isinstance(resp.get("output"), list) and len(resp["output"]) > 0)
    check("responses.create has usage block",
          isinstance(resp.get("usage"), dict))


def test_health():
    section("Health / meta")

    req = urllib.request.Request("http://127.0.0.1:8000/health",
                                 headers=HDR, method="GET")
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    check("GET /health returns status=ok",
          data.get("status") == "ok")

    code = call_expect_404("GET", "/nonexistent-endpoint-xyz")
    check("unknown endpoint returns 404",
          code == 404, f"got HTTP {code}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 56)
    print("  Mock OpenAI server — endpoint smoke tests")
    print("=" * 56)

    test_core()
    test_multimodal()
    test_files()
    test_fine_tuning()
    test_batches()
    test_vector_stores()
    test_assistants()
    test_threads_messages_runs()
    test_responses_api()
    test_health()

    print("\n" + "=" * 56)
    print(f"  Results: {_passed} passed, {_failed} failed")
    print("=" * 56)

    sys.exit(0 if _failed == 0 else 1)
