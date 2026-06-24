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

## Customizing the fake responses

Edit `_CANNED_REPLIES` and `_make_reply()` in `mock_openai_server.py` to change
what the assistant "says". You can make it echo, template, or rule-match inputs
to simulate different behaviors while you learn.


##Test
A zero-dependency, OpenAI-compatible HTTP API server for learning GenAI offline or behind a corporate firewall. No API key, no internet, no installs — just Python stdlib.

Point any OpenAI client at http://localhost:8000/v1 with any API key. When you're ready for the real service, change base_url and your code stays the same.

Features:
- Full OpenAI API surface: chat, completions, embeddings, images, audio, moderation, files, fine-tuning, batches, vector stores, assistants, threads, runs, responses
- Streaming SSE support for chat completions
- Stateful in-memory CRUD (create/list/retrieve/delete)
- CORS headers for browser-based clients
- PATCH support for assistants and threads
- Configurable simulated latency (--delay flag)
- Thread-safe concurrent request handling
- Comprehensive test suite with assertions
- Security: no outbound connections, no persistent storage, no credential handling