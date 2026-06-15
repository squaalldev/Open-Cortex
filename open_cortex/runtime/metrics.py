from dataclasses import dataclass
import httpx


@dataclass(frozen=True)
class RuntimeSnapshot:
    prompt_tps: float | None
    decode_tps: float | None
    requests_processing: int | None
    requests_deferred: int | None
    active_slots: int
    slot_context_tokens: tuple[int, ...]
    slot_context_size: int | None


def parse_prometheus_value(text: str, name: str) -> float | None:
    prefix = f"{name} "
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix))
    return None

def fetch_runtime_snapshot(
    client: httpx.Client,
    metrics_url: str,
    slots_url: str,
) -> RuntimeSnapshot:
    metrics_response = client.get(metrics_url)
    metrics_response.raise_for_status()

    slots_response = client.get(slots_url)
    slots_response.raise_for_status()

    metrics_text = metrics_response.text
    slots = slots_response.json()

    active_slots = [
        slot
        for slot in slots
        if slot.get("is_processing") is True
    ]

    requests_processing = parse_prometheus_value(
        metrics_text,
        "llamacpp:requests_processing",
    )
    requests_deferred = parse_prometheus_value(
        metrics_text,
        "llamacpp:requests_deferred",
    )

    return RuntimeSnapshot(
        prompt_tps=parse_prometheus_value(
            metrics_text,
            "llamacpp:prompt_tokens_seconds",
        ),
        decode_tps=parse_prometheus_value(
            metrics_text,
            "llamacpp:predicted_tokens_seconds",
        ),
        requests_processing=(
            int(requests_processing)
            if requests_processing is not None
            else None
        ),
        requests_deferred=(
            int(requests_deferred)
            if requests_deferred is not None
            else None
        ),
        active_slots=len(active_slots),
        slot_context_tokens=tuple(
            int(slot["n_prompt_tokens"])
            for slot in active_slots
            if slot.get("n_prompt_tokens") is not None
        ),
        slot_context_size=(
            int(slots[0]["n_ctx"])
            if slots and slots[0].get("n_ctx") is not None
            else None
        ),
    )