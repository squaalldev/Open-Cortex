import sys

from open_cortex.runtime.client import stream_chat_events
from open_cortex.runtime.messages import ChatMessage


messages = [
    ChatMessage(
        role="system",
        content="你是一名严谨的机器学习系统教师。",
    ),
    ChatMessage(
        role="user",
        content="用三句话解释大语言模型推理。",
    ),
]

for event in stream_chat_events(messages):
    if event.text_delta:
        print(event.text_delta, end="", flush=True)

    if event.kind in {"first_token", "request_completed"}:
        print(f"\n\nEVENT: {event}\n", file=sys.stderr)