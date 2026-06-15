import gradio as gr
from open_cortex.ui.gradio_history import history_to_chat_messages
from open_cortex.runtime.client import stream_chat_events


def format_runtime(event) -> str:
    if event.kind == "request_started":
        return "Phase: request started\nWaiting for first token..."

    if event.kind == "first_token" and event.snapshot is not None:
        snapshot = event.snapshot
        context_tokens = (
            snapshot.slot_context_tokens[0]
            if snapshot.slot_context_tokens
            else None
        )

        return "\n".join(
            [
                "Phase: first token",
                f"TTFT: {event.ttft_ms:.1f} ms",
                f"Context: {context_tokens} / {snapshot.slot_context_size}",
                f"Token Stream: {snapshot.decode_tps} tok/s",
                (
                    "Engine: "
                    f"processing={snapshot.requests_processing} "
                    f"deferred={snapshot.requests_deferred}"
                ),
            ]
        )

    if event.kind == "request_completed":
        return "\n".join(
            [
                "Phase: completed",
                f"Prompt tokens: {event.prompt_tokens}",
                f"Output tokens: {event.completion_tokens}",
                f"Prefill: {event.prompt_tps:.1f} tok/s",
                f"Decode: {event.decode_tps:.1f} tok/s",
            ]
        )

    return "Phase: decoding"

def user(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    if not user_message.strip():
        return "", history

    return "", history + [
        {
            "role": "user",
            "content": user_message,
        }
    ]

def bot(history: list):
    messages = history_to_chat_messages(history)

    history.append(
        {
            "role": "assistant",
            "content": "",
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": "",
        }
    )

    runtime_text = "Phase: idle"

    for event in stream_chat_events(messages):
        runtime_text = format_runtime(event)

        if event.text_delta:
            history[-1]["content"] += event.text_delta

        yield history, runtime_text


with gr.Blocks() as demo:
    gr.Markdown("# OpenCortex Minimal Chat")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=480,
            )
            msg = gr.Textbox(
                placeholder="Ask the local model...",
                show_label=False,
            )
            clear = gr.Button("Clear")

        with gr.Column(scale=1):
            runtime = gr.Textbox(
                label="Runtime",
                value="Phase: idle",
                lines=10,
                interactive=False,
            )


    msg.submit(
        user,
        [msg, chatbot],
        [msg, chatbot],
        queue=False,
    ).then(
        bot,
        chatbot,
        [chatbot,runtime],
    )

    clear.click(
        lambda: ([],"Phase: idle"),
        None,
        [chatbot, runtime],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()