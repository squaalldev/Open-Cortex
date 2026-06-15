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
    final_stats = None

    yield RuntimeEvent(
        kind="request_started",
        text_delta="",
        ttft_ms=None,
        snapshot=None,
    )

    with httpx.Client(timeout=120.0, trust_env=False) as client:
        with client.stream("POST", CHAT_URL, json=request_body) as response:
            response.raise_for_status()

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
                        if not first_token_seen:
                            first_token_seen = True
                            first_token_at = time.perf_counter()
                            ttft_ms = (first_token_at - request_started) * 1000
                            snapshot = fetch_runtime_snapshot(
                                client,
                                METRICS_URL,
                                SLOTS_URL,
                            )
                            yield RuntimeEvent(
                                kind="first_token",
                                text_delta=content,
                                ttft_ms=ttft_ms,
                                snapshot=snapshot,
                            )
                        else:
                            yield RuntimeEvent(
                                kind="token",
                                text_delta=content,
                                ttft_ms=None,
                                snapshot=None,
                            )

                if event.get("usage"):
                    final_stats = event

    if final_stats is not None:
        usage = final_stats["usage"]
        timings = final_stats["timings"]

        yield RuntimeEvent(
            kind="request_completed",
            text_delta="",
            ttft_ms=None,
            snapshot=None,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            prompt_tps=timings["prompt_per_second"],
            decode_tps=timings["predicted_per_second"],
        )