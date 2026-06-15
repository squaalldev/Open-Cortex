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
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_tps: float | None = None
    decode_tps: float | None = None