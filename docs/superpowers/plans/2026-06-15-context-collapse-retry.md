# Context Collapse Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn llama.cpp context overflow into a visible Context Collapse event, then trim oldest chat history and retry once.

**Architecture:** Keep llama.cpp client behavior simple: it raises the HTTP 400 from llama.cpp. The UI service layer catches the overflow, emits a `context_collapse` NDJSON event, trims oldest non-system messages, and retries once. The browser renders the collapse boundary and continues streaming the retried response.

**Tech Stack:** Python FastAPI streaming NDJSON, httpx, vanilla JS, CSS.

---

### Task 1: Backend Collapse Event And Retry

**Files:**
- Modify: `open_cortex/ui/app.py`
- Test: `tests/ui/test_app.py`

- [ ] Write a failing test that monkeypatches `stream_chat_events` to raise an HTTP 400 overflow on the first call and stream a normal response on the second call.
- [ ] Implement `_is_context_overflow`, `_trim_for_context_collapse`, and retry-once logic in `_stream_events`.
- [ ] Verify the stream contains `request_started`, `context_collapse`, then retried response events.

### Task 2: Frontend Collapse Boundary

**Files:**
- Modify: `open_cortex/ui/assets/open_cortex.js`
- Modify: `open_cortex/ui/assets/open_cortex.css`

- [ ] Add `applyContextCollapse(event)` to mark old messages as `historical-context`, insert `active-context-boundary`, and switch the right panel to collapse visuals.
- [ ] Ensure the next retried stream reuses the same assistant bubble.
- [ ] Run `mise exec -- node --check open_cortex/ui/assets/open_cortex.js`.

### Task 3: Verification

**Files:**
- Test: all tests

- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run python -m py_compile open_cortex/ui/app.py open_cortex/runtime/client.py`.
- [ ] Restart `uv run python app.py` and manually trigger a long-context continuation.
