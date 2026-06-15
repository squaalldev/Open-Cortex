import json
import time
from collections.abc import Iterator
import httpx

from open_cortex.runtime.events import RuntimeEvent
from open_cortex.runtime.metrics import fetch_runtime_snapshot
from open_cortex.runtime.messages import ChatMessage, to_llama_messages


CHAT_URL = "http://127.0.0.1:8080/v1/chat/completions"
METRICS_URL = "http://127.0.0.1:8080/metrics"
SLOTS_URL = "http://127.0.0.1:8080/slots"


def _raise_for_status_with_body(response: httpx.Response) -> None:
    if response.is_error:
        try:
            response.read()
        except httpx.HTTPError:
            pass
    response.raise_for_status()


def _working_memory_percent(
    context_tokens: int | None,
    context_size: int | None,
) -> float | None:
    if context_tokens is None or not context_size:
        return None
    return round(min(100.0, context_tokens / context_size * 100), 1)


def _detect_repetition(text: str) -> bool:
    normalized = " ".join(text.split())
    if len(normalized) < 120:
        return False

    for window in (96, 72, 48, 32):
        if len(normalized) <= window * 2:
            continue
        tail = normalized[-window:]
        if normalized[:-window].count(tail) >= 1:
            return True

    return False


def stream_chat_events(message: list[ChatMessage]) -> Iterator[RuntimeEvent]:
    request_body = {
        "messages": to_llama_messages(message),
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,
    }

    request_started = time.perf_counter()
    first_token_seen = False
    first_token_at = None
    generated_tokens = 0
    generated_text = ""
    final_stats = None
    base_context_tokens = None
    context_size = None

    yield RuntimeEvent(
        kind="request_started",
        text_delta="",
        ttft_ms=None,
        snapshot=None,
    )

    with httpx.Client(timeout=120.0, trust_env=False) as client:
        with client.stream("POST", CHAT_URL, json=request_body) as response:
            _raise_for_status_with_body(response)

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue

                data = line.removeprefix("data: ")

                if data == "[DONE]":
                    break

                event = json.loads(data)
                choices = event.get("choices", [])

                if choices:
                    content = choices[0].get("delta", {}).get("content")

                    if content:
                        now = time.perf_counter()
                        generated_tokens += 1
                        generated_text += content
                        elapsed_ms = (
                            (now - first_token_at) * 1000
                            if first_token_at is not None
                            else 0.0
                        )
                        live_tps = (
                            generated_tokens / (elapsed_ms / 1000)
                            if elapsed_ms > 0
                            else None
                        )
                        repetition_detected = _detect_repetition(generated_text)

                        if not first_token_seen:
                            first_token_seen = True
                            first_token_at = now
                            ttft_ms = (first_token_at - request_started) * 1000
                            snapshot = fetch_runtime_snapshot(
                                client,
                                METRICS_URL,
                                SLOTS_URL,
                            )
                            base_context_tokens = (
                                snapshot.slot_context_tokens[0]
                                if snapshot.slot_context_tokens
                                else None
                            )
                            context_size = snapshot.slot_context_size
                            context_tokens = (
                                base_context_tokens + generated_tokens
                                if base_context_tokens is not None
                                else None
                            )
                            yield RuntimeEvent(
                                kind="first_token",
                                text_delta=content,
                                ttft_ms=ttft_ms,
                                snapshot=snapshot,
                                generated_tokens=generated_tokens,
                                elapsed_ms=0.0,
                                live_tps=None,
                                repetition_detected=repetition_detected,
                                context_tokens=context_tokens,
                                context_size=context_size,
                                working_memory_percent=_working_memory_percent(
                                    context_tokens,
                                    context_size,
                                ),
                            )
                        else:
                            context_tokens = (
                                base_context_tokens + generated_tokens
                                if base_context_tokens is not None
                                else None
                            )
                            yield RuntimeEvent(
                                kind="token",
                                text_delta=content,
                                ttft_ms=None,
                                snapshot=None,
                                generated_tokens=generated_tokens,
                                elapsed_ms=elapsed_ms,
                                live_tps=live_tps,
                                repetition_detected=repetition_detected,
                                context_tokens=context_tokens,
                                context_size=context_size,
                                working_memory_percent=_working_memory_percent(
                                    context_tokens,
                                    context_size,
                                ),
                            )

                if event.get("usage"):
                    final_stats = event

    if final_stats is not None:
        usage = final_stats["usage"]
        timings = final_stats["timings"]
        context_tokens = usage["prompt_tokens"] + usage["completion_tokens"]

        yield RuntimeEvent(
            kind="request_completed",
            text_delta="",
            ttft_ms=None,
            snapshot=None,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            prompt_tps=timings["prompt_per_second"],
            decode_tps=timings["predicted_per_second"],
            generated_tokens=usage["completion_tokens"],
            repetition_detected=_detect_repetition(generated_text),
            context_tokens=context_tokens,
            context_size=context_size,
            working_memory_percent=_working_memory_percent(
                context_tokens,
                context_size,
            ),
        )
