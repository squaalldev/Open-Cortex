from open_cortex.runtime.messages import ChatMessage


def normalize_gradio_content(content) -> str | None:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)

        return "\n".join(parts)

    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]

    return None


def history_to_chat_messages(history: list) -> list[ChatMessage]:
    messages = []

    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = normalize_gradio_content(item.get("content"))

            if role in {"system", "user", "assistant"} and content and content.strip():
                messages.append(ChatMessage(role=role, content=content.strip()))

        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_content = normalize_gradio_content(item[0])
            assistant_content = normalize_gradio_content(item[1])

            if user_content and user_content.strip():
                messages.append(ChatMessage(role="user", content=user_content.strip()))

            if assistant_content and assistant_content.strip():
                messages.append(ChatMessage(role="assistant", content=assistant_content.strip()))

    return messages