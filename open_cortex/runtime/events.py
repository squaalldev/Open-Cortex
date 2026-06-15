from dataclasses import dataclass
from typing import Literal

from open_cortex.runtime.metrics import RuntimeSnapshot


RuntimeEventKind = Literal[
    "request_started",
    "first_token",
    "token",
    "request_completed",
]


@dataclass(frozen=True)
class RuntimeEvent:
    kind: RuntimeEventKind
    text_delta: str
    ttft_ms: float | None
    snapshot: RuntimeSnapshot | None
    generated_tokens: int = 0
    elapsed_ms: float | None = None
    live_tps: float | None = None
    repetition_detected: bool = False
    context_tokens: int | None = None
    context_size: int | None = None
    working_memory_percent: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_tps: float | None = None
    decode_tps: float | None = None
