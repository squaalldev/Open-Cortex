import json
import sys
import time

import httpx


URL = "http://127.0.0.1:8080/v1/chat/completions"

request_body = {
    "messages": [
        {
            "role": "user",
            "content": "用三句话解释大语言模型的推理过程。",
        }
    ],
    "temperature": 0.2,
    "max_tokens": 100,
    "stream": True,
    "stream_options": {"include_usage": True},
    "timings_per_token": True,
}

request_started = time.perf_counter()
first_token_at = None
final_stats = None

with httpx.Client(timeout=120.0) as client:
    with client.stream("POST", URL, json=request_body) as response:
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
                    if first_token_at is None:
                        first_token_at = time.perf_counter()

                    print(content, end="", flush=True)

            if event.get("usage"):
                final_stats = event

print()

if first_token_at is not None:
    ttft_ms = (first_token_at - request_started) * 1000
    print(f"TTFT: {ttft_ms:.1f} ms", file=sys.stderr)

if final_stats is not None:
    usage = final_stats["usage"]
    timings = final_stats["timings"]

    print(f"Prompt tokens: {usage['prompt_tokens']}", file=sys.stderr)
    print(f"Output tokens: {usage['completion_tokens']}", file=sys.stderr)
    print(
        f"Prefill: {timings['prompt_per_second']:.1f} tok/s",
        file=sys.stderr,
    )
    print(
        f"Decode: {timings['predicted_per_second']:.1f} tok/s",
        file=sys.stderr,
    )