from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from open_cortex.runtime.client import stream_chat_events
from open_cortex.runtime.events import RuntimeEvent
from open_cortex.runtime.messages import ChatMessage
from open_cortex.runtime.metrics import RuntimeSnapshot


ASSET_DIR = Path(__file__).with_name("assets")
CSS_FILE = ASSET_DIR / "open_cortex.css"
JS_FILE = ASSET_DIR / "open_cortex.js"
MAX_COLLAPSE_RETAINED_CHARS = 3000
COLLAPSED_MESSAGE_CHARS = 1200
DEFAULT_CONTEXT_SIZE = 2048


def _backend_mode() -> str:
    explicit = os.getenv("OPEN_CORTEX_BACKEND")
    if explicit:
        return explicit
    if os.getenv("SPACE_ID"):
        return "simulated"
    return "llama_cpp"


def _simulator_delay_seconds() -> float:
    return float(os.getenv("OPEN_CORTEX_SIMULATOR_DELAY_SECONDS", "0.025"))


def _asset_version() -> int:
    return int(max(CSS_FILE.stat().st_mtime, JS_FILE.stat().st_mtime))


def _memory_blocks() -> str:
    return "\n".join(
        '<i class="filled"></i>' if index < 1 else "<i></i>"
        for index in range(12)
    )


def _token_particles() -> str:
    return "\n".join('<i class="particle"></i>' for _ in range(9))


def _latency_trace() -> str:
    return "\n".join("<i></i>" for _ in range(8))


def render_index() -> str:
    asset_version = _asset_version()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenCortex Runtime Observatory</title>
  <link rel="stylesheet" href="/assets/open_cortex.css?v={asset_version}">
</head>
<body>
<main class="app phase-idle" id="app">
  <header class="topbar">
    <div class="brand">
      <div class="mark"></div>
      <div class="brand-name">OpenCortex</div>
    </div>
    <div class="top-actions">
      <div class="runtime-connection"><span class="live-dot"></span><span>llama.cpp · local</span></div>
    </div>
  </header>

  <div class="workspace">
    <section class="surface conversation">
      <div class="section-head">
        <div class="panel-title">Conversation</div>
        <div class="scenario" id="scenario-label">Live local chat</div>
      </div>

      <div class="messages" id="messages"></div>

      <div class="composer-wrap">
        <div class="composer">
          <textarea id="prompt" placeholder="Ask the local model, then watch inference state change..."></textarea>
          <div class="composer-actions">
            <span class="hint">↵ SEND · SHIFT + ↵ NEW LINE</span>
            <button class="send" id="send" type="button"><span>Send</span><span>↗</span></button>
          </div>
        </div>
      </div>
    </section>

    <div class="resizer" id="resizer" role="separator" aria-label="Resize conversation panel" aria-orientation="vertical">
      <span class="resize-grip"></span>
      <button class="drawer-toggle" id="drawer-toggle" type="button" aria-label="Show conversation">›</button>
    </div>

    <section class="surface observatory">
      <div class="observatory-head">
        <div class="panel-title">Runtime Observatory</div>
        <div class="model-state">
          <div class="model-name">Qwen2.5 1.5B · Q4_K_M</div>
          <div class="phase-pill" id="phase">IDLE</div>
          <div class="controls">
            <select id="experiment" aria-label="Runtime experiment">
              <option value="live">Live detected</option>
              <option value="long-context">Sim · Long context stress</option>
              <option value="memory-pressure">Sim · Memory pressure</option>
              <option value="slow-decode">Sim · Slow decode</option>
              <option value="context-collapse">Sim · Context collapse</option>
            </select>
            <button class="run-experiment" id="run-experiment" type="button">Run experiment</button>
          </div>
        </div>
      </div>

      <div class="observatory-stage">
        <div class="runtime-event" id="runtime-event">READY</div>
        <span class="conduit c1"></span>
        <span class="conduit c2"></span>
        <span class="conduit c3"></span>
        <span class="conduit c4"></span>

        <article class="organ memory active" id="memory-organ">
          <div class="organ-head">
            <div class="organ-name">Working Memory</div>
            <span class="organ-state" id="memory-state">Memory quiet</span>
          </div>
          <div class="metric-row">
            <div><span class="metric-value" id="kv">—</span><span class="metric-unit">%</span></div>
          </div>
          <div class="memory-blocks" id="memory-blocks">
            {_memory_blocks()}
          </div>
        </article>

        <article class="organ context active" id="context-organ">
          <div class="organ-head">
            <div class="organ-name">Context Window</div>
            <span class="organ-state" id="context-state">Context resting</span>
          </div>
          <div class="metric-row">
            <div><span class="metric-value" id="context-used">0</span><span class="metric-unit" id="context-unit">/ —</span></div>
          </div>
          <div class="context-vessel">
            <div class="context-fill" id="context-fill"></div>
            <div class="memory-tape" aria-hidden="true">
              <i class="memory-segment old"></i><i class="memory-segment old"></i>
              <i class="memory-segment"></i><i class="memory-segment"></i>
              <i class="memory-segment"></i><i class="memory-segment"></i>
              <i class="memory-segment recent"></i><i class="memory-segment recent"></i>
            </div>
            <div class="context-ticks"></div>
            <span class="tape-direction">→</span>
          </div>
        </article>

        <article class="organ tokens active" id="tokens-organ">
          <div class="organ-head">
            <div class="organ-name">Token Stream</div>
            <span class="organ-state" id="token-state">Stream dormant</span>
          </div>
          <div class="metric-row">
            <div><span class="metric-value" id="tps">0.0</span><span class="metric-unit">tok/s</span></div>
          </div>
          <div class="token-river">
            {_token_particles()}
          </div>
        </article>

        <article class="organ health active" id="engine-organ">
          <div class="organ-head">
            <div class="organ-name">Engine State</div>
            <span class="organ-state" id="health-state">Engine healthy</span>
          </div>
          <div class="metric-row">
            <div><span class="metric-value" id="ttft">—</span><span class="metric-unit">ms</span></div>
          </div>
          <div class="latency-trace">{_latency_trace()}</div>
        </article>

        <div class="core-wrap">
          <div class="core-rings"></div>
          <div class="core"></div>
        </div>
        <div class="core-copy">
          <div class="core-label">Cortex Core</div>
          <div class="core-state" id="core-state">Ready for input</div>
          <div class="core-detail" id="core-detail">ENGINE IDLE · SLOT AVAILABLE</div>
        </div>
      </div>

      <div class="evidence">
        <div class="evidence-cell">
          <div class="evidence-head"><span class="measured-dot"></span><span>Measured live by llama.cpp</span></div>
        </div>
        <div class="evidence-cell">
          <div class="evidence-label">Prompt eval</div>
          <div class="evidence-value" id="prompt-eval">measuring</div>
        </div>
        <div class="evidence-cell">
          <div class="evidence-label" id="evidence-3-label">KV evidence</div>
          <div class="evidence-value" id="kv-retained">unavailable</div>
        </div>
      </div>
    </section>
  </div>
</main>
<script type="module" src="/assets/open_cortex.js?v={asset_version}"></script>
</body>
</html>
"""


def _snapshot_to_payload(snapshot: RuntimeSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "prompt_tps": snapshot.prompt_tps,
        "decode_tps": snapshot.decode_tps,
        "requests_processing": snapshot.requests_processing,
        "requests_deferred": snapshot.requests_deferred,
        "active_slots": snapshot.active_slots,
        "slot_context_tokens": list(snapshot.slot_context_tokens),
        "slot_context_size": snapshot.slot_context_size,
    }


def event_to_payload(event: RuntimeEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "text_delta": event.text_delta,
        "ttft_ms": event.ttft_ms,
        "snapshot": _snapshot_to_payload(event.snapshot),
        "generated_tokens": event.generated_tokens,
        "elapsed_ms": event.elapsed_ms,
        "live_tps": event.live_tps,
        "repetition_detected": event.repetition_detected,
        "context_tokens": event.context_tokens,
        "context_size": event.context_size,
        "working_memory_percent": event.working_memory_percent,
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": event.completion_tokens,
        "prompt_tps": event.prompt_tps,
        "decode_tps": event.decode_tps,
    }


def _http_error_text(exc: httpx.HTTPStatusError) -> str:
    try:
        return exc.response.text
    except httpx.ResponseNotRead:
        try:
            exc.response.read()
        except (httpx.HTTPError, httpx.StreamError):
            return ""
        return exc.response.text


def _is_context_overflow(exc: httpx.HTTPStatusError) -> bool:
    return (
        exc.response.status_code == 400
        and "exceeds the available context size" in _http_error_text(exc)
    )


def _context_size_from_error(exc: httpx.HTTPStatusError) -> int | None:
    match = re.search(r"context size \((\d+) tokens\)", _http_error_text(exc))
    if match is None:
        return None
    return int(match.group(1))


def _trim_for_context_collapse(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], int]:
    system_messages = [message for message in messages if message.role == "system"]
    conversation = [message for message in messages if message.role != "system"]

    if not conversation:
        return messages, 0

    latest_user_index = next(
        (
            index
            for index in range(len(conversation) - 1, -1, -1)
            if conversation[index].role == "user"
        ),
        len(conversation) - 1,
    )
    latest_user = conversation[latest_user_index]
    recent_start = max(1, len(conversation) - 3)
    recent = conversation[recent_start:]

    retained: list[ChatMessage] = []
    for message in recent:
        if message is latest_user:
            retained.append(message)
            continue
        if len(message.content) > MAX_COLLAPSE_RETAINED_CHARS:
            retained.append(
                ChatMessage(
                    role=message.role,
                    content=(
                        "[Earlier content collapsed; recent tail retained.]\n"
                        + message.content[-COLLAPSED_MESSAGE_CHARS:]
                    ),
                )
            )
        else:
            retained.append(message)

    if latest_user not in retained:
        retained.append(latest_user)

    retained = list(dict.fromkeys(retained))
    dropped = len(conversation) - len(retained)
    return [*system_messages, *retained], max(0, dropped)


def _parse_messages(raw_messages: Any) -> list[ChatMessage]:
    if not isinstance(raw_messages, list):
        return []

    messages: list[ChatMessage] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            continue
        role = raw_message.get("role")
        content = raw_message.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        messages.append(ChatMessage(role=role, content=content.strip()))

    return messages


def _latest_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _simulated_answer(messages: list[ChatMessage]) -> str:
    latest = _latest_user_text(messages).lower()
    if any(marker in latest for marker in ("story", "故事", "repeat", "loop")):
        phrase = (
            "A tiny local model begins exploring a hidden runtime chamber. "
            "It finds the same sentence again, and the same sentence again. "
        )
        return (
            "I can feel the decode rhythm becoming unstable.\n\n"
            + phrase * 8
            + "\n\nOpenCortex marks this as a thought loop because recent generated "
            "text is repeating instead of moving forward."
        )
    if any(marker in latest for marker in ("context", "window", "memory", "上下文", "记忆")):
        return (
            "OpenCortex treats the context window as the model's active workspace. "
            "As the conversation grows, more tokens must be prefetched before decode "
            "can begin. When the active window fills, older turns are still visible in "
            "the chat log, but they fall outside what the model can reliably use."
        )
    return (
        "OpenCortex is running in Space demo mode. It simulates the same runtime "
        "event stream used by the local llama.cpp integration: prefill, first token, "
        "decode throughput, context pressure, and completion. Run it locally with "
        "llama.cpp to replace this simulated stream with live engine evidence."
    )


def _token_chunks(text: str) -> list[str]:
    chunks = re.findall(r"\S+\s*", text)
    return chunks or [text]


def _stream_simulated_events(messages: list[ChatMessage]) -> Iterator[RuntimeEvent]:
    answer = _simulated_answer(messages)
    chunks = _token_chunks(answer)
    context_size = DEFAULT_CONTEXT_SIZE
    prompt_tokens = max(24, sum(len(message.content) for message in messages) // 3)
    base_context_tokens = min(context_size, prompt_tokens + 96)
    delay_seconds = _simulator_delay_seconds()

    yield RuntimeEvent(kind="request_started", text_delta="", ttft_ms=None, snapshot=None)

    snapshot = RuntimeSnapshot(
        prompt_tps=72.4,
        decode_tps=24.6,
        requests_processing=1,
        requests_deferred=0,
        active_slots=1,
        slot_context_tokens=(base_context_tokens,),
        slot_context_size=context_size,
    )

    generated_text = ""
    started_at = time.perf_counter()
    for index, chunk in enumerate(chunks, start=1):
        if delay_seconds:
            time.sleep(delay_seconds)
        generated_text += chunk
        context_tokens = min(context_size, base_context_tokens + index)
        event_kind = "first_token" if index == 1 else "token"
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000)
        live_tps = None if index == 1 or elapsed_ms == 0 else index / (elapsed_ms / 1000)
        repetition_detected = "same sentence again" in generated_text and index > 18

        yield RuntimeEvent(
            kind=event_kind,
            text_delta=chunk,
            ttft_ms=420.0 if index == 1 else None,
            snapshot=snapshot if index == 1 else None,
            generated_tokens=index,
            elapsed_ms=0.0 if index == 1 else elapsed_ms,
            live_tps=live_tps,
            repetition_detected=repetition_detected,
            context_tokens=context_tokens,
            context_size=context_size,
            working_memory_percent=round(min(100.0, context_tokens / context_size * 100), 1),
        )

    final_context_tokens = min(context_size, base_context_tokens + len(chunks))
    yield RuntimeEvent(
        kind="request_completed",
        text_delta="",
        ttft_ms=None,
        snapshot=None,
        generated_tokens=len(chunks),
        repetition_detected="same sentence again" in generated_text,
        context_tokens=final_context_tokens,
        context_size=context_size,
        working_memory_percent=round(min(100.0, final_context_tokens / context_size * 100), 1),
        prompt_tokens=prompt_tokens,
        completion_tokens=len(chunks),
        prompt_tps=72.4,
        decode_tps=24.6,
    )


def _stream_events(messages: list[ChatMessage]) -> Iterator[str]:
    if _backend_mode() == "simulated":
        for event in _stream_simulated_events(messages):
            yield json.dumps(event_to_payload(event), ensure_ascii=False) + "\n"
        return

    try:
        for event in stream_chat_events(messages):
            yield json.dumps(event_to_payload(event), ensure_ascii=False) + "\n"
    except httpx.HTTPStatusError as exc:
        if _is_context_overflow(exc):
            trimmed_messages, dropped_messages = _trim_for_context_collapse(messages)
            yield json.dumps(
                {
                    "kind": "context_collapse",
                    "message": (
                        "The earliest turns fell outside the active context. "
                        "OpenCortex trimmed old history and retried with recent memory."
                    ),
                    "dropped_messages": dropped_messages,
                    "retained_messages": len(trimmed_messages),
                    "context_size": _context_size_from_error(exc),
                },
                ensure_ascii=False,
            ) + "\n"
            try:
                for event in stream_chat_events(trimmed_messages):
                    yield json.dumps(event_to_payload(event), ensure_ascii=False) + "\n"
                return
            except httpx.HTTPStatusError as retry_exc:
                if _is_context_overflow(retry_exc):
                    payload = {
                        "kind": "error",
                        "code": "context_overflow",
                        "message": (
                            "The conversation still exceeds llama.cpp's context window "
                            "after collapse. Start a new run or restart llama-server with "
                            "a larger -c value."
                        ),
                    }
                else:
                    payload = {
                        "kind": "error",
                        "code": "backend_http_error",
                        "message": f"{type(retry_exc).__name__}: {retry_exc}",
                    }
        else:
            payload = {
                "kind": "error",
                "code": "backend_http_error",
                "message": f"{type(exc).__name__}: {exc}",
            }
        yield json.dumps(payload, ensure_ascii=False) + "\n"
    except Exception as exc:
        yield json.dumps(
            {
                "kind": "error",
                "code": "backend_error",
                "message": f"{type(exc).__name__}: {exc}",
            },
            ensure_ascii=False,
        ) + "\n"


def create_app() -> FastAPI:
    app = FastAPI(title="OpenCortex")
    app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_index()

    @app.post("/api/chat")
    async def chat(request: Request) -> StreamingResponse:
        payload = await request.json()
        messages = _parse_messages(payload.get("messages"))
        return StreamingResponse(
            _stream_events(messages),
            media_type="application/x-ndjson",
        )

    return app


app = create_app()


def main() -> None:
    host = os.getenv("OPEN_CORTEX_HOST", "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1")
    uvicorn.run(
        "open_cortex.ui.app:app",
        host=host,
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
