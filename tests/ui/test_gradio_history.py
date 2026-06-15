from open_cortex.ui.gradio_history import history_to_chat_messages


def test_history_to_chat_messages_extracts_text_from_gradio_list_content():
    history = [
        {
            "role": "user",
            "content": [
                {
                    "text": "请解释 KV Cache",
                    "type": "text",
                }
            ],
        }
    ]

    messages = history_to_chat_messages(history)

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "请解释 KV Cache"


def test_history_to_chat_messages_ignores_non_text_content():
    history = [
        {
            "role": "user",
            "content": [
                {
                    "path": "/tmp/image.png",
                    "type": "image",
                }
            ],
        }
    ]

    messages = history_to_chat_messages(history)

    assert messages == []


def test_history_to_chat_messages_supports_legacy_pair_format():
    history = [
        [
            "hello",
            "hi",
        ]
    ]

    messages = history_to_chat_messages(history)

    assert [(m.role, m.content) for m in messages] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]