import pytest
import httpx

from open_cortex.runtime.client import (
    _detect_repetition,
    _raise_for_status_with_body,
    _working_memory_percent,
)


def test_detect_repetition_marks_repeated_generation_loop():
    repeated = (
        "艾丽和卡斯一起探索了整个宇宙，他们发现了一个惊人的事实："
        "整个宇宙的中心并不是银河系，而是整个宇宙的中心。"
    )

    assert _detect_repetition(repeated * 3) is True


def test_detect_repetition_allows_short_normal_text():
    text = "KV Cache stores attention keys and values so decode can reuse prior context."

    assert _detect_repetition(text) is False


def test_working_memory_percent_is_context_derived_and_capped():
    assert _working_memory_percent(512, 2048) == 25.0
    assert _working_memory_percent(4096, 2048) == 100.0
    assert _working_memory_percent(None, 2048) is None


def test_raise_for_status_with_body_preserves_streaming_error_body():
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        stream=httpx.ByteStream(
            b"request (2059 tokens) exceeds the available context size (2048 tokens)"
        ),
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        _raise_for_status_with_body(response)

    assert "exceeds the available context size" in exc_info.value.response.text
