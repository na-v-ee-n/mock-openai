# Mock OpenAI Server

A **zero-dependency**, OpenAI-compatible API server for learning GenAI **offline**,
behind a corporate firewall, with **no API key and no internet** required.

Your code is written exactly as it would be for the real OpenAI API. To switch to
the real service later, you only change the `base_url` and use your real key.

## Files

| File | Purpose |
|------|---------|
| `mock_openai_server.py` | The fake OpenAI server (stdlib only, no installs) |
| `client_stdlib.py` | Example client using only Python standard library |
| `client_openai_sdk.py` | Example using the official `openai` SDK |
| `test_all_endpoints.py` | Smoke-test suite with assertions; exits 0 on pass |
| `requirements.txt` | Optional — only needed for `client_openai_sdk.py` |

## Quick start

1. Start the server (leave this terminal running):

   ```powershell
   python mock_openai_server.py                # default port 8000
   python mock_openai_server.py --port 8080    # custom port
   python mock_openai_server.py --delay 0      # zero latency
   ```

   It listens on `http://127.0.0.1:8000/v1`.

   **CLI flags**

   | Flag | Default | Description |
   |------|---------|-------------|
   | `--host` | `127.0.0.1` | Bind address |
   | `--port` | `8000` | TCP port |
   | `--delay` | `0.15` | Simulated chat latency in seconds |
   | `--error-rate` | `0.0` | Probability (0.0–1.0) of injecting a `500` error |
   | `--rate-limit` | `0` | Max requests/min per client before `429` (0 = off) |
   | `--max-context-tokens` | `0` | Reject chat over N prompt tokens with `400` (0 = off) |
   | `--realistic-stream` | off | Variable chunk sizes + jittery streaming latency |

   ### Practising real-world failure handling

   By default the server runs in clean, deterministic mode. The flags above let
   you simulate the failure modes you **will** hit against the real API, so you
   can practise handling them offline:

   ```powershell
   # Force retry/backoff logic: 1 in 5 requests fails with 500
   python mock_openai_server.py --error-rate 0.2

   # Practise 429 handling + Retry-After header (3 req/min per client)
   python mock_openai_server.py --rate-limit 3

   # Trigger context_length_exceeded (400) for long prompts
   python mock_openai_server.py --max-context-tokens 100

   # See realistic streaming chunking and latency
   python mock_openai_server.py --realistic-stream
   ```

   - **429** responses include a `Retry-After` header, like the real API.
   - **400** context errors use `code: "context_length_exceeded"`.
   - **500** errors use `type: "server_error"` — write retries with backoff.

2. In a second terminal, run a client:

   ```powershell
   python client_stdlib.py
   ```

   Or, with the official SDK (PyPI is reachable on your network):

   ```powershell
   python -m pip install openai
   python client_openai_sdk.py
   ```

## Supported endpoints (full OpenAI surface)

**Core**
- `GET  /v1/models`, `GET /v1/models/{id}`
- `POST /v1/chat/completions`  (supports `"stream": true` and `"tools"` -> tool_calls)
- `POST /v1/completions`
- `POST /v1/embeddings`

**Multimodal**
- `POST /v1/images/generations`, `/v1/images/edits`, `/v1/images/variations`
- `POST /v1/audio/speech` (returns binary), `/v1/audio/transcriptions`, `/v1/audio/translations`
- `POST /v1/moderations`

**Stateful resources** (create / list / retrieve / delete kept in memory)
- `Files`        : `POST/GET /v1/files`, `GET/DELETE /v1/files/{id}`, `GET /v1/files/{id}/content`
- `Fine-tuning`  : `POST/GET /v1/fine_tuning/jobs`, `GET /v1/fine_tuning/jobs/{id}[/events]`, `POST .../cancel`
- `Batches`      : `POST/GET /v1/batches`, `GET /v1/batches/{id}`, `POST .../cancel`
- `Vector stores`: `POST/GET /v1/vector_stores`, `GET/DELETE /v1/vector_stores/{id}`

**Assistants API**
- `Assistants` : `POST/GET /v1/assistants`, `GET/PATCH/DELETE /v1/assistants/{id}`
- `Threads`    : `POST /v1/threads`, `GET/PATCH/DELETE /v1/threads/{id}`
- `Messages`   : `POST/GET /v1/threads/{id}/messages`
- `Runs`       : `POST/GET /v1/threads/{id}/runs`, `GET .../runs/{run_id}`

**Responses API**
- `POST /v1/responses`

Run `python test_all_endpoints.py` (with the server running) to verify them all.
The script prints `[PASS]` / `[FAIL]` per assertion and exits with code `0` on
success or `1` on any failure — compatible with CI pipelines.

## Switching to real OpenAI later

When you're on an unrestricted network (e.g. a phone hotspot), change:

```python
client = OpenAI(
    base_url="https://api.openai.com/v1",  # real endpoint
    api_key="sk-...your-real-key...",
)
```

Everything else in your code stays the same.

## Security


| Property | Detail |
|---|---|
| **No real credentials** | The only "key" in the code is the literal string `"mock-key"`. The server accepts any string — it never validates or stores credentials. |
| **No outbound connections** | The server is purely inbound. It never calls OpenAI, sends telemetry, or makes any external network request. |
| **No persistent storage** | All state lives in an in-process Python dict. Nothing is written to disk. Everything is wiped when the server stops. |
| **Localhost by default** | The default bind address is `127.0.0.1` — unreachable from other machines on your network. |
| **Dummy data only** | All responses are synthetic. No user input is logged, stored, or forwarded anywhere. |
| **No sensitive source content** | The source files contain no API keys, passwords, internal URLs, or personal information. |

### Network exposure warning

If you start the server with `--host 0.0.0.0` (to share it on a LAN), the
startup banner will print a visible warning:

```
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  WARNING: Listening on 0.0.0.0 — server is reachable
  from other machines on your network. This server has NO
  authentication. Use only in trusted environments.
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

**Never expose the server on an untrusted or public network.** Because it
accepts any API key, anyone who can reach the port can use it freely.

### Before committing / publishing

- Do **not** paste a real `sk-...` OpenAI key anywhere in the source files.
- The `.gitignore` already excludes virtual environments, caches, and IDE
  folders. Check `git status` before your first push.

## What you CAN and CANNOT learn here

This mock is excellent for learning the **API protocol** — how to call endpoints,
shape requests, parse responses, stream, and handle errors. It is **not** a real
model, so it cannot teach you anything about model *behaviour*.

If you are a junior dev practising, be clear about the boundary:

### You CAN learn (transferable to the real API)

- **Request/response shapes** — exact JSON for every endpoint
- **SDK and HTTP client usage** — same code works against real OpenAI
- **Streaming (SSE)** — parsing `data:` chunks and `[DONE]`
- **Error handling** — `429` + `Retry-After`, `500` retries, `400` context limits
  (enable with `--rate-limit`, `--error-rate`, `--max-context-tokens`)
- **Resource lifecycles** — create / list / retrieve / patch / delete flows
- **Tool/function-call plumbing** — the `tool_calls` response structure

### You CANNOT learn (needs the real API)

| Skill | Why the mock can't teach it |
|---|---|
| **Prompt engineering** | Replies are canned text. Changing your prompt does not change the quality or correctness of the answer. |
| **Model selection** | `gpt-4o` vs `gpt-3.5` vs `o1` return the same dummy content here. You won't feel their real trade-offs. |
| **Temperature / sampling** | These params are accepted but ignored — output never actually varies by them. |
| **Real tool-call reasoning** | The mock always returns a fixed fake tool call; it does not *decide* when or how to call a function. |
| **Token/cost optimisation** | Token counts are rough estimates and free. You get no real feedback on cost or context budgeting. |
| **Embeddings quality** | Vectors are random-but-deterministic noise. Similarity search "works" mechanically but is semantically meaningless. |
| **Hallucination & safety behaviour** | There is no real model, so you can't study refusals, jailbreaks, or factual errors. |
| **Real latency & throughput** | Simulated delays are approximations, not production performance characteristics. |

### Recommended path for juniors

1. **Build against the mock first** — get your client code, error handling, and
   plumbing correct without spending a cent.
2. **Turn on the failure flags** (`--error-rate`, `--rate-limit`,
   `--max-context-tokens`) and make your code survive them.
3. **Then switch to the real API** with a small budget (a few dollars) to learn
   prompting, model selection, and actual output quality — the things only a
   real model can teach.

Treat this server as a **flight simulator**: perfect for learning the controls
and emergency procedures, but you still have to fly the real plane eventually.

## Customizing the fake responses

Edit `_CANNED_REPLIES` and `_make_reply()` in `mock_openai_server.py` to change
what the assistant "says". You can make it echo, template, or rule-match inputs
to simulate different behaviors while you learn.
