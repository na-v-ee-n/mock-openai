"""
Comprehensive Mock OpenAI API server (zero dependencies, stdlib only).

Mimics the real OpenAI HTTP API across (almost) every endpoint so you can
learn and build entirely offline. Every response matches OpenAI's JSON schema
(ids, object types, usage, etc.) but the content is dummy data.

Point any OpenAI client at  http://localhost:8000/v1  with any api key.

Run:
    python mock_openai_server.py
    python mock_openai_server.py --port 8080

Supported (see ROUTES at the bottom for the full list):
  Models, Chat Completions (+tools +streaming), Completions, Embeddings,
  Images (generations/edits/variations), Audio (speech/transcriptions/
  translations), Moderations, Files, Fine-tuning, Batches, Vector Stores,
  Assistants / Threads / Messages / Runs, and the Responses API.
"""

import argparse
import base64
import json
import random
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ===========================================================================
# In-memory stores so create/list/retrieve/delete stay consistent per session.
# ===========================================================================

STORE = {
    "files": {},
    "fine_tuning_jobs": {},
    "batches": {},
    "assistants": {},
    "threads": {},
    "messages": {},        # message_id -> message (each carries thread_id)
    "runs": {},            # run_id -> run (each carries thread_id)
    "vector_stores": {},
}
STORE_LOCK = threading.Lock()

# Tunable behaviour, all set from CLI flags in main(). Defaults keep the server
# deterministic so the existing test suite passes unchanged.
CONFIG = {
    "chat_delay": 0.15,          # seconds of fake latency for non-stream chat
    "error_rate": 0.0,           # 0.0-1.0 chance of injecting a 500 error
    "rate_limit": 0,             # max requests/min per client (0 = disabled)
    "max_context_tokens": 0,     # reject chat over this many prompt tokens (0 = off)
    "realistic_stream": False,   # variable chunk size + jittery latency
}

# Per-client request timestamps for rate limiting: {client_ip: [epoch, ...]}
RATE_LOG = {}
RATE_LOG_LOCK = threading.Lock()

# Sentinel: returned by handlers that already wrote their own HTTP response
# (e.g. streaming SSE). Distinct from None, which signals "resource not found".
_STREAMED = object()


class APIError(Exception):
    """Raised by handlers to emit an OpenAI-style error with a chosen status."""

    def __init__(self, message, status=400, err_type="invalid_request_error",
                 code=None, headers=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.err_type = err_type
        self.code = code
        self.headers = headers or {}


def now() -> int:
    return int(time.time())


def rid(prefix: str) -> str:
    return f"{prefix}-" + uuid.uuid4().hex[:24]


# ===========================================================================
# Dummy "intelligence".
# ===========================================================================

_CANNED_REPLIES = [
    "This is a mock response from your local OpenAI-compatible server. "
    "It mirrors the real API's structure so your code runs unchanged.",
    "Sure! Here is a dummy answer. Swap base_url to the real OpenAI endpoint "
    "later and the same code will keep working.",
    "Hello from the mock server. This is a placeholder reply with realistic "
    "formatting and a valid response schema.",
    "Mock mode active. The response shape matches OpenAI so you can practice "
    "parsing it safely offline.",
]


def _estimate_tokens(text) -> int:
    if not isinstance(text, str):
        text = json.dumps(text)
    return max(1, len(text) // 4)


def _last_user_message(messages):
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):  # multimodal content parts
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c
    return ""


def _make_reply(messages):
    user_text = _last_user_message(messages)
    base = random.choice(_CANNED_REPLIES)
    if user_text:
        snippet = str(user_text).strip().replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:80] + "..."
        return f'{base}\n\n(You said: "{snippet}")'
    return base


def _embedding_vector(text, dims=1536):
    rnd = random.Random(hash(str(text)) & 0xFFFFFFFF)
    return [round(rnd.uniform(-1, 1), 6) for _ in range(dims)]


# A 1x1 transparent PNG, base64 (used for image responses).
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# ===========================================================================
# Payload builders (schema-faithful to OpenAI).
# ===========================================================================

def chat_completion_payload(model, messages, tools=None, tool_choice=None):
    prompt_tokens = sum(_estimate_tokens(m.get("content", "")) for m in messages)

    # If the caller passed tools and didn't disable them, simulate a tool call.
    if tools and tool_choice != "none":
        fn = tools[0].get("function", {})
        fname = fn.get("name", "my_function")
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": rid("call"),
                    "type": "function",
                    "function": {
                        "name": fname,
                        "arguments": json.dumps({"mock_arg": "mock_value"}),
                    },
                }
            ],
        }
        finish = "tool_calls"
        completion_tokens = 12
    else:
        reply = _make_reply(messages)
        message = {"role": "assistant", "content": reply, "refusal": None}
        finish = "stop"
        completion_tokens = _estimate_tokens(reply)

    return {
        "id": rid("chatcmpl"),
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "system_fingerprint": "fp_mock",
        "choices": [
            {"index": 0, "message": message, "logprobs": None,
             "finish_reason": finish}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chat_stream_chunks(model, messages):
    reply = _make_reply(messages)
    cid = rid("chatcmpl")
    created = now()

    def frame(delta, finish_reason=None):
        return {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model, "system_fingerprint": "fp_mock",
            "choices": [{"index": 0, "delta": delta,
                         "finish_reason": finish_reason}],
        }

    yield frame({"role": "assistant", "content": ""})
    words = reply.split(" ")
    if CONFIG["realistic_stream"]:
        # Real OpenAI emits variable-sized chunks (1-3 tokens), not 1 word each.
        i = 0
        while i < len(words):
            take = random.randint(1, 3)
            chunk = " ".join(words[i:i + take]) + " "
            yield frame({"content": chunk})
            i += take
    else:
        for word in words:
            yield frame({"content": word + " "})
    yield frame({}, finish_reason="stop")


def completion_payload(model, prompt):
    text = "\n" + random.choice(_CANNED_REPLIES)
    pt = _estimate_tokens(prompt)
    ct = _estimate_tokens(text)
    return {
        "id": rid("cmpl"), "object": "text_completion", "created": now(),
        "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None,
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                  "total_tokens": pt + ct},
    }


def embeddings_payload(model, inputs):
    if isinstance(inputs, str):
        inputs = [inputs]
    data, total = [], 0
    for i, text in enumerate(inputs):
        total += _estimate_tokens(text)
        data.append({"object": "embedding", "index": i,
                     "embedding": _embedding_vector(text)})
    return {"object": "list", "data": data, "model": model,
            "usage": {"prompt_tokens": total, "total_tokens": total}}


def models_payload():
    ids = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
           "text-embedding-3-small", "text-embedding-3-large",
           "dall-e-3", "tts-1", "whisper-1", "omni-moderation-latest"]
    return {"object": "list",
            "data": [{"id": m, "object": "model", "created": now(),
                      "owned_by": "mock"} for m in ids]}


def model_payload(model_id):
    return {"id": model_id, "object": "model", "created": now(),
            "owned_by": "mock"}


def images_payload(n=1, b64=False):
    data = []
    for _ in range(int(n or 1)):
        if b64:
            data.append({"b64_json": _TINY_PNG_B64,
                         "revised_prompt": "A mock generated image."})
        else:
            data.append({"url": "https://example.com/mock-image.png",
                         "revised_prompt": "A mock generated image."})
    return {"created": now(), "data": data}


def transcription_payload():
    return {"text": "This is a mock transcription of the supplied audio."}


def translation_payload():
    return {"text": "This is a mock English translation of the audio."}


def moderation_payload(model):
    cats = ["hate", "hate/threatening", "harassment", "harassment/threatening",
            "self-harm", "self-harm/intent", "self-harm/instructions",
            "sexual", "sexual/minors", "violence", "violence/graphic"]
    return {
        "id": rid("modr"), "model": model,
        "results": [{
            "flagged": False,
            "categories": {c: False for c in cats},
            "category_scores": {c: round(random.uniform(0, 0.01), 6)
                                for c in cats},
        }],
    }


# ===========================================================================
# Stateful object builders (files, fine-tuning, batches, assistants, ...).
# ===========================================================================

def file_payload(filename="mock.jsonl", purpose="fine-tune", nbytes=120):
    obj = {"id": rid("file"), "object": "file", "bytes": nbytes,
           "created_at": now(), "filename": filename, "purpose": purpose,
           "status": "processed", "status_details": None}
    with STORE_LOCK:
        STORE["files"][obj["id"]] = obj
    return obj


def list_payload(items):
    items = list(items)
    return {"object": "list", "data": items,
            "first_id": items[0]["id"] if items else None,
            "last_id": items[-1]["id"] if items else None,
            "has_more": False}


def deleted_payload(obj_id, object_type):
    return {"id": obj_id, "object": object_type, "deleted": True}


def fine_tuning_job_payload(model="gpt-4o-mini", training_file="file-mock"):
    obj = {"id": rid("ftjob"), "object": "fine_tuning.job", "created_at": now(),
           "finished_at": None, "model": model,
           "fine_tuned_model": None, "organization_id": "org-mock",
           "status": "queued", "training_file": training_file,
           "validation_file": None, "result_files": [], "trained_tokens": None,
           "error": None, "hyperparameters": {"n_epochs": "auto"},
           "seed": 0}
    with STORE_LOCK:
        STORE["fine_tuning_jobs"][obj["id"]] = obj
    return obj


def batch_payload(endpoint="/v1/chat/completions", input_file_id="file-mock"):
    obj = {"id": rid("batch"), "object": "batch", "endpoint": endpoint,
           "errors": None, "input_file_id": input_file_id,
           "completion_window": "24h", "status": "validating",
           "output_file_id": None, "error_file_id": None,
           "created_at": now(), "in_progress_at": None, "expires_at": now() + 86400,
           "finalizing_at": None, "completed_at": None, "failed_at": None,
           "expired_at": None, "cancelling_at": None, "cancelled_at": None,
           "request_counts": {"total": 0, "completed": 0, "failed": 0},
           "metadata": None}
    with STORE_LOCK:
        STORE["batches"][obj["id"]] = obj
    return obj


def vector_store_payload(name="Mock Vector Store"):
    obj = {"id": rid("vs"), "object": "vector_store", "created_at": now(),
           "name": name, "usage_bytes": 0, "file_counts":
           {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0,
            "total": 0}, "status": "completed", "expires_after": None,
           "expires_at": None, "last_active_at": now(), "metadata": {}}
    with STORE_LOCK:
        STORE["vector_stores"][obj["id"]] = obj
    return obj


def assistant_payload(model="gpt-4o-mini", name=None, instructions=None,
                      tools=None):
    obj = {"id": rid("asst"), "object": "assistant", "created_at": now(),
           "name": name, "description": None, "model": model,
           "instructions": instructions, "tools": tools or [],
           "tool_resources": {}, "metadata": {}, "temperature": 1.0,
           "top_p": 1.0, "response_format": "auto"}
    with STORE_LOCK:
        STORE["assistants"][obj["id"]] = obj
    return obj


def thread_payload():
    obj = {"id": rid("thread"), "object": "thread", "created_at": now(),
           "tool_resources": {}, "metadata": {}}
    with STORE_LOCK:
        STORE["threads"][obj["id"]] = obj
    return obj


def message_payload(thread_id, role="user", content=""):
    if isinstance(content, str):
        content = [{"type": "text",
                    "text": {"value": content, "annotations": []}}]
    obj = {"id": rid("msg"), "object": "thread.message", "created_at": now(),
           "thread_id": thread_id, "role": role, "content": content,
           "assistant_id": None, "run_id": None, "attachments": [],
           "metadata": {}}
    with STORE_LOCK:
        STORE["messages"][obj["id"]] = obj
    return obj


def run_payload(thread_id, assistant_id, status="completed"):
    obj = {"id": rid("run"), "object": "thread.run", "created_at": now(),
           "thread_id": thread_id, "assistant_id": assistant_id,
           "status": status, "started_at": now(), "expires_at": None,
           "cancelled_at": None, "failed_at": None, "completed_at": now(),
           "last_error": None, "model": "gpt-4o-mini", "instructions": None,
           "tools": [], "metadata": {}, "usage":
           {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
           "temperature": 1.0, "top_p": 1.0, "response_format": "auto"}
    with STORE_LOCK:
        STORE["runs"][obj["id"]] = obj
    return obj


def responses_payload(model, input_text):
    text = _make_reply([{"role": "user", "content": input_text}])
    return {
        "id": rid("resp"), "object": "response", "created_at": now(),
        "status": "completed", "model": model, "error": None,
        "output": [{
            "type": "message", "id": rid("msg"), "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text,
                         "annotations": []}],
        }],
        "usage": {"input_tokens": _estimate_tokens(input_text),
                  "output_tokens": _estimate_tokens(text),
                  "total_tokens": _estimate_tokens(input_text) +
                  _estimate_tokens(text)},
        "metadata": {},
    }


# ===========================================================================
# Route handlers. Each takes (handler, body, **path_params) and returns either
# a dict (sent as JSON), or a tuple (raw_bytes, content_type) for binary.
# `handler` is the MockOpenAIHandler instance (used only for streaming).
# ===========================================================================

def h_models(h, body):
    return models_payload()


def h_model_get(h, body, model):
    return model_payload(model)


def h_chat(h, body):
    model = body.get("model", "gpt-4o-mini")
    messages = body.get("messages", [])

    limit = CONFIG["max_context_tokens"]
    if limit:
        prompt_tokens = sum(_estimate_tokens(m.get("content", ""))
                            for m in messages)
        if prompt_tokens > limit:
            raise APIError(
                f"This model's maximum context length is {limit} tokens, "
                f"however your messages resulted in {prompt_tokens} tokens. "
                f"Please reduce the length of the messages.",
                status=400, err_type="invalid_request_error",
                code="context_length_exceeded",
            )

    if body.get("stream"):
        h.stream_sse(chat_stream_chunks(model, messages))
        return _STREAMED
    time.sleep(CONFIG["chat_delay"])
    return chat_completion_payload(model, messages, body.get("tools"),
                                   body.get("tool_choice"))


def h_completions(h, body):
    return completion_payload(body.get("model", "gpt-3.5-turbo"),
                              body.get("prompt", ""))


def h_embeddings(h, body):
    return embeddings_payload(body.get("model", "text-embedding-3-small"),
                              body.get("input", ""))


def h_images_gen(h, body):
    return images_payload(body.get("n", 1),
                          b64=body.get("response_format") == "b64_json")


def h_images_edit(h, body):
    return images_payload(1, b64=False)


def h_images_var(h, body):
    return images_payload(1, b64=False)


def h_audio_speech(h, body):
    # Real API returns binary audio; return dummy bytes with audio mimetype.
    return (b"MOCK_AUDIO_BYTES_" + uuid.uuid4().hex.encode(), "audio/mpeg")


def h_audio_transcribe(h, body):
    return transcription_payload()


def h_audio_translate(h, body):
    return translation_payload()


def h_moderations(h, body):
    return moderation_payload(body.get("model", "omni-moderation-latest"))


# ---- Files ----
def h_files_create(h, body):
    return file_payload()


def h_files_list(h, body):
    with STORE_LOCK:
        items = list(STORE["files"].values())
    return list_payload(items)


def h_files_get(h, body, file_id):
    with STORE_LOCK:
        obj = STORE["files"].get(file_id)
    return obj  # None -> 404


def h_files_delete(h, body, file_id):
    with STORE_LOCK:
        STORE["files"].pop(file_id, None)
    return deleted_payload(file_id, "file")


def h_files_content(h, body, file_id):
    return (b"mock file content\n", "application/octet-stream")


# ---- Fine-tuning ----
def h_ft_create(h, body):
    return fine_tuning_job_payload(body.get("model", "gpt-4o-mini"),
                                   body.get("training_file", "file-mock"))


def h_ft_list(h, body):
    with STORE_LOCK:
        items = list(STORE["fine_tuning_jobs"].values())
    return list_payload(items)


def h_ft_get(h, body, job_id):
    with STORE_LOCK:
        obj = STORE["fine_tuning_jobs"].get(job_id)
    return obj  # None -> 404


def h_ft_cancel(h, body, job_id):
    with STORE_LOCK:
        job = STORE["fine_tuning_jobs"].get(job_id)
    if job is None:
        return None  # 404
    job["status"] = "cancelled"
    return job


def h_ft_events(h, body, job_id):
    return list_payload([{
        "id": rid("ftevent"), "object": "fine_tuning.job.event",
        "created_at": now(), "level": "info",
        "message": "Mock fine-tuning event.", "type": "message"}])


# ---- Batches ----
def h_batch_create(h, body):
    return batch_payload(body.get("endpoint", "/v1/chat/completions"),
                         body.get("input_file_id", "file-mock"))


def h_batch_list(h, body):
    with STORE_LOCK:
        items = list(STORE["batches"].values())
    return list_payload(items)


def h_batch_get(h, body, batch_id):
    with STORE_LOCK:
        obj = STORE["batches"].get(batch_id)
    return obj  # None -> 404


def h_batch_cancel(h, body, batch_id):
    with STORE_LOCK:
        b = STORE["batches"].get(batch_id)
    if b is None:
        return None  # 404
    b["status"] = "cancelling"
    return b


# ---- Vector stores ----
def h_vs_create(h, body):
    return vector_store_payload(body.get("name", "Mock Vector Store"))


def h_vs_list(h, body):
    with STORE_LOCK:
        items = list(STORE["vector_stores"].values())
    return list_payload(items)


def h_vs_get(h, body, vs_id):
    with STORE_LOCK:
        obj = STORE["vector_stores"].get(vs_id)
    return obj  # None -> 404


def h_vs_delete(h, body, vs_id):
    with STORE_LOCK:
        STORE["vector_stores"].pop(vs_id, None)
    return deleted_payload(vs_id, "vector_store.deleted")


# ---- Assistants ----
def h_asst_create(h, body):
    return assistant_payload(body.get("model", "gpt-4o-mini"),
                             body.get("name"), body.get("instructions"),
                             body.get("tools"))


def h_asst_list(h, body):
    with STORE_LOCK:
        items = list(STORE["assistants"].values())
    return list_payload(items)


def h_asst_get(h, body, asst_id):
    with STORE_LOCK:
        obj = STORE["assistants"].get(asst_id)
    return obj  # None -> 404


def h_asst_patch(h, body, asst_id):
    with STORE_LOCK:
        obj = STORE["assistants"].get(asst_id)
    if obj is None:
        return None  # 404
    for field in ("name", "description", "model", "instructions", "tools",
                  "metadata", "temperature", "top_p", "response_format"):
        if field in body:
            obj[field] = body[field]
    return obj


def h_asst_delete(h, body, asst_id):
    with STORE_LOCK:
        STORE["assistants"].pop(asst_id, None)
    return deleted_payload(asst_id, "assistant.deleted")


# ---- Threads / Messages / Runs ----
def h_thread_create(h, body):
    return thread_payload()


def h_thread_get(h, body, thread_id):
    with STORE_LOCK:
        obj = STORE["threads"].get(thread_id)
    return obj  # None -> 404


def h_thread_patch(h, body, thread_id):
    with STORE_LOCK:
        obj = STORE["threads"].get(thread_id)
    if obj is None:
        return None  # 404
    for field in ("tool_resources", "metadata"):
        if field in body:
            obj[field] = body[field]
    return obj


def h_thread_delete(h, body, thread_id):
    with STORE_LOCK:
        STORE["threads"].pop(thread_id, None)
    return deleted_payload(thread_id, "thread.deleted")


def h_msg_create(h, body, thread_id):
    return message_payload(thread_id, body.get("role", "user"),
                           body.get("content", ""))


def h_msg_list(h, body, thread_id):
    with STORE_LOCK:
        msgs = [m for m in STORE["messages"].values()
                if m["thread_id"] == thread_id]
    return list_payload(msgs)


def h_run_create(h, body, thread_id):
    # Simulate the assistant adding a reply message to the thread.
    asst_id = body.get("assistant_id", rid("asst"))
    message_payload(thread_id, "assistant",
                    _make_reply([{"role": "user",
                                  "content": "continue"}]))
    return run_payload(thread_id, asst_id)


def h_run_list(h, body, thread_id):
    with STORE_LOCK:
        runs = [r for r in STORE["runs"].values() if r["thread_id"] == thread_id]
    return list_payload(runs)


def h_run_get(h, body, thread_id, run_id):
    with STORE_LOCK:
        obj = STORE["runs"].get(run_id)
    return obj  # None -> 404


# ---- Responses API ----
def h_responses(h, body):
    inp = body.get("input", "")
    if isinstance(inp, list):
        inp = " ".join(str(p.get("content", p)) for p in inp
                       if isinstance(p, (dict, str)))
    return responses_payload(body.get("model", "gpt-4o-mini"), inp)


# ===========================================================================
# Route table: (METHOD, compiled regex on path, handler).
# ===========================================================================

def _rx(pattern):
    return re.compile("^" + pattern + "/?$")

ROUTES = [
    ("GET", _rx(r"/v1/models"), h_models),
    ("GET", _rx(r"/v1/models/(?P<model>[^/]+)"), h_model_get),
    ("POST", _rx(r"/v1/chat/completions"), h_chat),
    ("POST", _rx(r"/v1/completions"), h_completions),
    ("POST", _rx(r"/v1/embeddings"), h_embeddings),
    ("POST", _rx(r"/v1/images/generations"), h_images_gen),
    ("POST", _rx(r"/v1/images/edits"), h_images_edit),
    ("POST", _rx(r"/v1/images/variations"), h_images_var),
    ("POST", _rx(r"/v1/audio/speech"), h_audio_speech),
    ("POST", _rx(r"/v1/audio/transcriptions"), h_audio_transcribe),
    ("POST", _rx(r"/v1/audio/translations"), h_audio_translate),
    ("POST", _rx(r"/v1/moderations"), h_moderations),
    # Files
    ("POST", _rx(r"/v1/files"), h_files_create),
    ("GET", _rx(r"/v1/files"), h_files_list),
    ("GET", _rx(r"/v1/files/(?P<file_id>[^/]+)/content"), h_files_content),
    ("GET", _rx(r"/v1/files/(?P<file_id>[^/]+)"), h_files_get),
    ("DELETE", _rx(r"/v1/files/(?P<file_id>[^/]+)"), h_files_delete),
    # Fine-tuning
    ("POST", _rx(r"/v1/fine_tuning/jobs"), h_ft_create),
    ("GET", _rx(r"/v1/fine_tuning/jobs"), h_ft_list),
    ("GET", _rx(r"/v1/fine_tuning/jobs/(?P<job_id>[^/]+)/events"), h_ft_events),
    ("POST", _rx(r"/v1/fine_tuning/jobs/(?P<job_id>[^/]+)/cancel"), h_ft_cancel),
    ("GET", _rx(r"/v1/fine_tuning/jobs/(?P<job_id>[^/]+)"), h_ft_get),
    # Batches
    ("POST", _rx(r"/v1/batches"), h_batch_create),
    ("GET", _rx(r"/v1/batches"), h_batch_list),
    ("POST", _rx(r"/v1/batches/(?P<batch_id>[^/]+)/cancel"), h_batch_cancel),
    ("GET", _rx(r"/v1/batches/(?P<batch_id>[^/]+)"), h_batch_get),
    # Vector stores
    ("POST", _rx(r"/v1/vector_stores"), h_vs_create),
    ("GET", _rx(r"/v1/vector_stores"), h_vs_list),
    ("GET", _rx(r"/v1/vector_stores/(?P<vs_id>[^/]+)"), h_vs_get),
    ("DELETE", _rx(r"/v1/vector_stores/(?P<vs_id>[^/]+)"), h_vs_delete),
    # Assistants
    ("POST", _rx(r"/v1/assistants"), h_asst_create),
    ("GET", _rx(r"/v1/assistants"), h_asst_list),
    ("GET", _rx(r"/v1/assistants/(?P<asst_id>[^/]+)"), h_asst_get),
    ("PATCH", _rx(r"/v1/assistants/(?P<asst_id>[^/]+)"), h_asst_patch),
    ("DELETE", _rx(r"/v1/assistants/(?P<asst_id>[^/]+)"), h_asst_delete),
    # Threads / messages / runs
    ("POST", _rx(r"/v1/threads"), h_thread_create),
    ("GET", _rx(r"/v1/threads/(?P<thread_id>[^/]+)"), h_thread_get),
    ("PATCH", _rx(r"/v1/threads/(?P<thread_id>[^/]+)"), h_thread_patch),
    ("DELETE", _rx(r"/v1/threads/(?P<thread_id>[^/]+)"), h_thread_delete),
    ("POST", _rx(r"/v1/threads/(?P<thread_id>[^/]+)/messages"), h_msg_create),
    ("GET", _rx(r"/v1/threads/(?P<thread_id>[^/]+)/messages"), h_msg_list),
    ("POST", _rx(r"/v1/threads/(?P<thread_id>[^/]+)/runs"), h_run_create),
    ("GET", _rx(r"/v1/threads/(?P<thread_id>[^/]+)/runs"), h_run_list),
    ("GET", _rx(r"/v1/threads/(?P<thread_id>[^/]+)/runs/(?P<run_id>[^/]+)"),
     h_run_get),
    # Responses API
    ("POST", _rx(r"/v1/responses"), h_responses),
]


# ===========================================================================
# HTTP handler.
# ===========================================================================

class MockOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "MockOpenAI/2.0"

    # -- logging --
    def log_message(self, fmt, *args):  # args[0]=code, args[1]=message
        print(f"[{self.log_date_time_string()}] {self.command} {self.path} {args[0]}")

    # -- CORS helpers --
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods",
                         "GET, POST, DELETE, PATCH, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, Authorization, OpenAI-Beta")

    # -- response helpers --
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, data, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, message, status=400, err_type="invalid_request_error",
                    code=None, extra_headers=None):
        body = json.dumps({"error": {"message": message, "type": err_type,
                                     "param": None, "code": code}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, str(v))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    # -- simulated failure modes (opt-in via CLI flags) --
    def _check_rate_limit(self):
        limit = CONFIG["rate_limit"]
        if not limit:
            return
        client = self.client_address[0]
        cutoff = time.time() - 60
        with RATE_LOG_LOCK:
            hits = [t for t in RATE_LOG.get(client, []) if t > cutoff]
            if len(hits) >= limit:
                retry = max(1, int(60 - (time.time() - hits[0])))
                RATE_LOG[client] = hits
                raise APIError(
                    f"Rate limit reached for requests. Limit: {limit} / min. "
                    f"Please try again in {retry}s.",
                    status=429, err_type="rate_limit_error",
                    code="rate_limit_exceeded",
                    headers={"Retry-After": retry},
                )
            hits.append(time.time())
            RATE_LOG[client] = hits

    def _maybe_inject_error(self):
        rate = CONFIG["error_rate"]
        if rate and random.random() < rate:
            raise APIError(
                "The server had an error while processing your request. "
                "Sorry about that! (simulated)",
                status=500, err_type="server_error",
                code="internal_error",
            )

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        ctype = self.headers.get("Content-Type", "")
        if "application/json" in ctype and raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        # multipart/form-data or other: we don't parse it, return marker dict.
        return {} if raw == b"" else {"_raw": True}

    def stream_sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors_headers()
        self.end_headers()
        realistic = CONFIG["realistic_stream"]
        try:
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                # Real streams have jittery inter-chunk latency, not a fixed tick.
                time.sleep(random.uniform(0.01, 0.2) if realistic else 0.04)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- dispatch --
    def _dispatch(self, method):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/health") and method == "GET":
            self._send_json({"status": "ok", "service": "mock-openai"})
            return

        body = self._read_body()
        if body is None:
            self._send_error("Invalid JSON in request body.")
            return

        for m, rx, func in ROUTES:
            if m != method:
                continue
            match = rx.match(path)
            if match:
                try:
                    # Opt-in failure simulations run before the handler.
                    self._check_rate_limit()
                    self._maybe_inject_error()
                    result = func(self, body, **match.groupdict())
                except APIError as exc:
                    self._send_error(exc.message, status=exc.status,
                                     err_type=exc.err_type, code=exc.code,
                                     extra_headers=exc.headers)
                    return
                if result is _STREAMED:
                    return  # handler already wrote the full response
                if result is None:
                    # Handler explicitly returned None -> resource not found
                    resource = path.rsplit("/", 1)[-1]
                    self._send_error(
                        f"No such resource: {resource}",
                        status=404, err_type="not_found",
                    )
                    return
                if isinstance(result, tuple):  # binary (data, content_type)
                    self._send_binary(result[0], result[1])
                else:
                    self._send_json(result)
                return

        self._send_error(f"Unknown endpoint: {method} {path}", status=404,
                         err_type="not_found")

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Mock OpenAI-compatible API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--delay", type=float, default=0.15, metavar="SECONDS",
        help="Simulated latency for non-streaming chat responses (default: 0.15s)",
    )
    parser.add_argument(
        "--error-rate", type=float, default=0.0, metavar="0.0-1.0",
        help="Probability of injecting a 500 server error (default: 0 = off)",
    )
    parser.add_argument(
        "--rate-limit", type=int, default=0, metavar="N",
        help="Max requests/min per client before 429 (default: 0 = off)",
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=0, metavar="N",
        help="Reject chat requests over N prompt tokens with 400 (default: 0 = off)",
    )
    parser.add_argument(
        "--realistic-stream", action="store_true",
        help="Variable chunk sizes and jittery latency in streamed responses",
    )
    args = parser.parse_args()
    CONFIG["chat_delay"] = args.delay
    CONFIG["error_rate"] = args.error_rate
    CONFIG["rate_limit"] = args.rate_limit
    CONFIG["max_context_tokens"] = args.max_context_tokens
    CONFIG["realistic_stream"] = args.realistic_stream

    server = ThreadingHTTPServer((args.host, args.port), MockOpenAIHandler)
    base = f"http://{args.host}:{args.port}/v1"
    _loopback = {"127.0.0.1", "localhost", "::1"}
    _network_exposed = args.host not in _loopback

    print("=" * 64)
    print("  Mock OpenAI server v2 (full API surface, no internet needed)")
    print(f"  Base URL : {base}")
    print("  API key  : anything works (e.g. 'mock-key')")
    print(f"  Routes   : {len(ROUTES)} endpoints registered")
    print(f"  Delay    : {args.delay}s (chat non-stream)")
    _sims = []
    if args.error_rate:
        _sims.append(f"error-rate={args.error_rate}")
    if args.rate_limit:
        _sims.append(f"rate-limit={args.rate_limit}/min")
    if args.max_context_tokens:
        _sims.append(f"max-context={args.max_context_tokens}tok")
    if args.realistic_stream:
        _sims.append("realistic-stream")
    print(f"  Sims     : {', '.join(_sims) if _sims else 'none (clean mode)'}")
    if _network_exposed:
        print("  " + "!" * 60)
        print(f"  WARNING: Listening on {args.host} — server is reachable")
        print("  from other machines on your network. This server has NO")
        print("  authentication. Use only in trusted environments.")
        print("  " + "!" * 60)
    print("  Stop with Ctrl+C")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down mock server.")
        server.shutdown()


if __name__ == "__main__":
    main()

