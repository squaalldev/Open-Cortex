import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

from open_cortex.runtime.events import RuntimeEvent
from open_cortex.runtime.messages import ChatMessage
from open_cortex.runtime.metrics import RuntimeSnapshot
from open_cortex.ui.app import (
    create_app,
    event_to_payload,
    render_index,
    _backend_mode,
    _stream_events,
    _stream_simulated_events,
    _http_error_text,
    _is_context_overflow,
    _trim_for_context_collapse,
)


def test_render_index_uses_original_product_shell():
    html = render_index()

    assert 'id="app"' in html
    assert 'class="topbar"' in html
    assert "OpenCortex" in html
    assert "llama.cpp · local" in html
    assert "EN / 中" not in html
    assert 'id="lang"' not in html
    assert 'id="clear"' not in html
    assert ">Clear<" not in html
    assert ">Send<" in html
    assert "Send message" not in html
    assert 'class="workspace"' in html
    assert 'id="messages"' in html
    assert 'id="prompt"' in html
    assert 'id="send"' in html
    assert 'id="runtime-event"' in html
    assert 'id="experiment"' in html
    assert 'id="run-experiment"' in html
    assert "Live detected" in html
    assert "Sim · Memory pressure" in html
    assert "Sim · Context collapse" in html
    assert 'id="memory-organ"' in html
    assert 'id="context-used"' in html
    assert 'id="core-state"' in html
    assert "/assets/open_cortex.css?v=" in html
    assert "/assets/open_cortex.js?v=" in html


def test_styles_use_dynamic_viewport_height():
    css = (create_app.__globals__["ASSET_DIR"] / "open_cortex.css").read_text()

    assert "height: 100dvh" in css
    assert "min-height: 100dvh" in css


def test_frontend_exposes_experiment_and_engine_state_handlers():
    js = (create_app.__globals__["ASSET_DIR"] / "open_cortex.js").read_text()

    assert "function renderMarkdown" in js
    assert "message.role === \"assistant\" ? renderMarkdown" in js
    assert "isStreaming" in js
    assert 'message.role === "assistant" && message.isStreaming' in js
    assert "activeAssistant.isStreaming = false" in js
    assert "function runExperiment" in js
    assert "function runMemoryPressureExperiment" in js
    assert "function runSlowDecodeExperiment" in js
    assert "function runContextCollapseExperiment" in js
    assert "collapseHoldUntil" in js
    assert "contextWindowFullPending" in js
    assert "function applyContextFullWarning" in js
    assert "function applyDeferredContextEjection" in js
    assert "CONTEXT WINDOW FULL" in js
    assert "hazard-context-full" in js
    assert "hazard-loop" in js
    assert "runtimeEvent.classList.add(\"hazard\"" in js
    assert "Earlier turns fell outside the active context." in js
    assert "engine-strained" in js
    assert "Engine recovering" in js


def test_danger_states_have_decisive_visual_treatment():
    css = (create_app.__globals__["ASSET_DIR"] / "open_cortex.css").read_text()

    assert ".app.hazard-context-full .runtime-event" in css
    assert ".app.hazard-loop .runtime-event" in css
    assert ".runtime-event.hazard" in css
    assert ".runtime-event.critical" in css
    assert "context-full-shock" in css
    assert "loop-warning-scan" in css
    assert "danger-atmosphere" in css


def test_context_collapse_styles_pop_out_oldest_message():
    css = (create_app.__globals__["ASSET_DIR"] / "open_cortex.css").read_text()

    assert "context-pop-out" in css
    assert "@keyframes context-pop-out" in css


def test_create_app_serves_index():
    app = create_app()
    client = TestClient(app)

    response = client.get("/")

    assert isinstance(app, FastAPI)
    assert response.status_code == 200
    assert "OpenCortex" in response.text
    assert 'id="send"' in response.text


def test_space_defaults_to_simulated_backend(monkeypatch):
    monkeypatch.delenv("OPEN_CORTEX_BACKEND", raising=False)
    monkeypatch.setenv("SPACE_ID", "build-small-hackathon/open-cortex")

    assert _backend_mode() == "simulated"


def test_explicit_backend_overrides_space_default(monkeypatch):
    monkeypatch.setenv("OPEN_CORTEX_BACKEND", "llama_cpp")
    monkeypatch.setenv("SPACE_ID", "build-small-hackathon/open-cortex")

    assert _backend_mode() == "llama_cpp"


def test_stream_events_can_run_without_llama_cpp_in_simulated_mode(monkeypatch):
    monkeypatch.setenv("OPEN_CORTEX_BACKEND", "simulated")
    monkeypatch.setenv("OPEN_CORTEX_SIMULATOR_DELAY_SECONDS", "0")

    messages = [ChatMessage(role="user", content="What is OpenCortex?")]

    payloads = [json.loads(line) for line in _stream_events(messages)]

    assert payloads[0]["kind"] == "request_started"
    assert payloads[1]["kind"] == "first_token"
    assert payloads[-1]["kind"] == "request_completed"
    assert "OpenCortex" in "".join(payload["text_delta"] for payload in payloads)
    assert payloads[-1]["context_size"] == 2048
    assert payloads[-1]["decode_tps"] > 0


def test_simulated_story_can_surface_thought_loop(monkeypatch):
    monkeypatch.setenv("OPEN_CORTEX_SIMULATOR_DELAY_SECONDS", "0")

    messages = [ChatMessage(role="user", content="Write a long repeating story.")]

    payloads = [
        event_to_payload(event)
        for event in _stream_simulated_events(messages)
    ]

    assert any(payload["repetition_detected"] for payload in payloads)


def test_event_to_payload_includes_live_loop_state():
    event = RuntimeEvent(
        kind="token",
        text_delta="银河系",
        ttft_ms=None,
        snapshot=RuntimeSnapshot(
            prompt_tps=88.8,
            decode_tps=21.5,
            requests_processing=1,
            requests_deferred=0,
            active_slots=1,
            slot_context_tokens=(512,),
            slot_context_size=2048,
        ),
        generated_tokens=66,
        elapsed_ms=2100.0,
        live_tps=19.7,
        repetition_detected=True,
        context_tokens=578,
        context_size=2048,
        working_memory_percent=28.2,
    )

    payload = event_to_payload(event)

    assert payload["kind"] == "token"
    assert payload["text_delta"] == "银河系"
    assert payload["generated_tokens"] == 66
    assert payload["live_tps"] == 19.7
    assert payload["repetition_detected"] is True
    assert payload["context_tokens"] == 578
    assert payload["context_size"] == 2048
    assert payload["working_memory_percent"] == 28.2
    assert payload["snapshot"]["slot_context_tokens"] == [512]
    assert payload["snapshot"]["slot_context_size"] == 2048


def test_stream_events_emits_context_collapse_and_retries_trimmed_history(monkeypatch):
    calls: list[list[ChatMessage]] = []
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    overflow = httpx.HTTPStatusError(
        "400 Bad Request",
        request=request,
        response=httpx.Response(
            400,
            request=request,
            text="request (2059 tokens) exceeds the available context size (2048 tokens)",
        ),
    )

    def fake_stream_chat_events(messages: list[ChatMessage]):
        calls.append(messages)
        if len(calls) == 1:
            yield RuntimeEvent(
                kind="request_started",
                text_delta="",
                ttft_ms=None,
                snapshot=None,
            )
            raise overflow
        yield RuntimeEvent(
            kind="first_token",
            text_delta="继续",
            ttft_ms=12.3,
            snapshot=None,
            context_tokens=128,
            context_size=2048,
            working_memory_percent=6.2,
        )

    monkeypatch.setattr("open_cortex.ui.app.stream_chat_events", fake_stream_chat_events)
    messages = [
        ChatMessage(role="system", content="You are Cortex."),
        ChatMessage(role="user", content="old question"),
        ChatMessage(role="assistant", content="old long answer"),
        ChatMessage(role="user", content="recent question"),
        ChatMessage(role="assistant", content="recent answer"),
        ChatMessage(role="user", content="继续"),
    ]

    payloads = [json.loads(line) for line in _stream_events(messages)]

    assert [payload["kind"] for payload in payloads] == [
        "request_started",
        "context_collapse",
        "first_token",
    ]
    assert payloads[1]["dropped_messages"] == 2
    assert payloads[1]["retained_messages"] == 4
    assert payloads[1]["context_size"] == 2048
    assert [message.content for message in calls[1]] == [
        "You are Cortex.",
        "recent question",
        "recent answer",
        "继续",
    ]


def test_context_overflow_detection_reads_streaming_error_body():
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        stream=httpx.ByteStream(
            b"request (2059 tokens) exceeds the available context size (2048 tokens)"
        ),
    )
    error = httpx.HTTPStatusError("400 Bad Request", request=request, response=response)

    assert _is_context_overflow(error) is True


def test_http_error_text_handles_closed_stream_without_crashing():
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        stream=httpx.ByteStream(b"closed"),
    )
    response.close()
    error = httpx.HTTPStatusError("400 Bad Request", request=request, response=response)

    assert _http_error_text(error) == ""


def test_trim_for_context_collapse_drops_oversized_assistant_history():
    messages = [
        ChatMessage(role="user", content="write a long story"),
        ChatMessage(role="assistant", content="故事" * 4000),
        ChatMessage(role="user", content="what happened next?"),
        ChatMessage(role="assistant", content="recent answer"),
        ChatMessage(role="user", content="继续"),
    ]

    trimmed, dropped = _trim_for_context_collapse(messages)

    assert dropped == 2
    assert trimmed == [
        ChatMessage(role="user", content="what happened next?"),
        ChatMessage(role="assistant", content="recent answer"),
        ChatMessage(role="user", content="继续"),
    ]


def test_trim_for_context_collapse_keeps_recent_tail_of_oversized_answer():
    messages = [
        ChatMessage(role="user", content="write a long story"),
        ChatMessage(role="assistant", content="A" * 5000 + "recent ending"),
        ChatMessage(role="user", content="1"),
    ]

    trimmed, dropped = _trim_for_context_collapse(messages)

    assert dropped == 1
    assert len(trimmed) == 2
    assert trimmed[0].role == "assistant"
    assert "Earlier content collapsed" in trimmed[0].content
    assert "recent ending" in trimmed[0].content
    assert trimmed[1] == ChatMessage(role="user", content="1")
