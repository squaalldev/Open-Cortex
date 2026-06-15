# OpenCortex MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished bilingual Gradio observatory that streams chat and truthful llama.cpp runtime evidence, distinguishes five experiments, and deploys locally or through Hugging Face Space with an optional Modal backend.

**Architecture:** Gradio owns queuing and per-session state, while custom HTML/CSS and a framework-free browser controller render the approved observatory interface. A backend-neutral event protocol keeps simulated and live inference interchangeable; the live adapter uses llama.cpp `/apply-template`, `/tokenize`, `/completion`, and `/metrics`, with all semantic states derived from measured values and a benchmark profile.

**Tech Stack:** Python 3.12, uv 0.11, Gradio Blocks, httpx, prometheus-client, pytest, pytest-httpx, Node.js 24 built-in test runner, llama.cpp `llama-server`, Hugging Face Spaces, optional Modal.

---

## File Map

Create or modify the following focused units:

```text
app.py                                      # Hugging Face and local entry point
pyproject.toml                              # Python package, runtime, test, lint config
.gitignore                                  # Track deployment files; ignore generated artifacts
README.md                                   # Setup, architecture, truthfulness, deployment
src/open_cortex/config.py                   # Environment-backed immutable settings
src/open_cortex/controller.py               # One-turn orchestration and session state
src/open_cortex/backends/base.py            # Inference backend protocol
src/open_cortex/backends/simulated.py       # Deterministic development backend
src/open_cortex/backends/llama_cpp.py       # llama.cpp HTTP and SSE adapter
src/open_cortex/backends/metrics.py         # Prometheus snapshot parsing
src/open_cortex/backends/load.py            # Dedicated-demo controlled load
src/open_cortex/runtime/models.py           # State, event, message, experiment types
src/open_cortex/runtime/state_engine.py     # Metrics-to-semantic-state mapping
src/open_cortex/runtime/context.py          # Chat template, token budget, forgetting
src/open_cortex/runtime/benchmark.py        # Benchmark profile and CLI
src/open_cortex/ui/app.py                   # Gradio Blocks construction and callbacks
src/open_cortex/ui/render.py                # Escaped chat and state bridge HTML
src/open_cortex/ui/localization.py          # English and Chinese product copy
src/open_cortex/ui/assets/open_cortex.css   # Product surface and organ visuals
src/open_cortex/ui/assets/open_cortex.js    # State application, splitter, drawer, locale
tests/                                     # Unit, contract, UI, and integration tests
deploy/huggingface/README.md                # Space configuration
deploy/modal/modal_app.py                   # Optional remote llama-server deployment
```

The implementation does not add React, a frontend build step, a database, a
metrics database, or a Python llama.cpp binding.

### Task 1: Replace the Scaffold With an Installable Application

**Files:**
- Delete: `main.py`
- Create: `app.py`
- Create: `src/open_cortex/__init__.py`
- Create: `src/open_cortex/config.py`
- Create: `tests/test_config.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Add the failing settings test**

```python
# tests/test_config.py
from open_cortex.config import Settings


def test_settings_default_to_simulator(monkeypatch):
    monkeypatch.delenv("OPEN_CORTEX_BACKEND", raising=False)
    settings = Settings.from_env()

    assert settings.backend == "simulated"
    assert settings.llama_base_url == "http://127.0.0.1:8080"
    assert settings.context_size == 4096
    assert settings.enable_load_experiments is False
```

- [ ] **Step 2: Install and lock the project dependencies**

Run:

```bash
uv add gradio httpx prometheus-client
uv add --dev pytest pytest-httpx pytest-cov ruff
```

Expected: `pyproject.toml` and `uv.lock` contain the resolved packages.

- [ ] **Step 3: Configure packaging and quality commands**

Use this project configuration in `pyproject.toml`, retaining the versions
resolved by `uv add`:

```toml
[project]
name = "open-cortex"
version = "0.1.0"
description = "A real-time observatory for local LLM inference"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "gradio",
  "httpx",
  "prometheus-client",
]

[project.scripts]
open-cortex = "open_cortex.ui.app:main"
open-cortex-benchmark = "open_cortex.runtime.benchmark:main"

[dependency-groups]
dev = [
  "pytest",
  "pytest-cov",
  "pytest-httpx",
  "ruff",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 4: Implement immutable environment settings**

```python
# src/open_cortex/config.py
from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    backend: str
    llama_base_url: str
    model_name: str
    context_size: int
    request_timeout_seconds: float
    enable_load_experiments: bool
    remote_backend: bool

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = os.getenv("OPEN_CORTEX_LLAMA_BASE_URL", "http://127.0.0.1:8080")
        return cls(
            backend=os.getenv("OPEN_CORTEX_BACKEND", "simulated"),
            llama_base_url=base_url.rstrip("/"),
            model_name=os.getenv("OPEN_CORTEX_MODEL_NAME", "Llama 3.2 3B Instruct"),
            context_size=int(os.getenv("OPEN_CORTEX_CONTEXT_SIZE", "4096")),
            request_timeout_seconds=float(
                os.getenv("OPEN_CORTEX_REQUEST_TIMEOUT_SECONDS", "120")
            ),
            enable_load_experiments=_as_bool(
                os.getenv("OPEN_CORTEX_ENABLE_LOAD_EXPERIMENTS")
            ),
            remote_backend=_as_bool(os.getenv("OPEN_CORTEX_REMOTE_BACKEND")),
        )
```

- [ ] **Step 5: Add the Space entry point and clean ignore rules**

```python
# app.py
from open_cortex.ui.app import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
```

Delete `main.py`. Remove the `deploy` line from `.gitignore`; add
`.pytest_cache/`, `.ruff_cache/`, `htmlcov/`, `*.gguf`, and `models/`.

- [ ] **Step 6: Verify and commit**

Run:

```bash
uv run pytest tests/test_config.py -v
uv run ruff check src tests app.py
```

Expected: all tests pass and Ruff reports no errors.

```bash
git add .gitignore app.py pyproject.toml uv.lock src/open_cortex tests/test_config.py
git commit -m "build: scaffold OpenCortex application"
```

### Task 2: Define the Backend-Neutral Runtime Protocol

**Files:**
- Create: `src/open_cortex/runtime/__init__.py`
- Create: `src/open_cortex/runtime/models.py`
- Create: `src/open_cortex/backends/__init__.py`
- Create: `src/open_cortex/backends/base.py`
- Create: `tests/runtime/test_models.py`

- [ ] **Step 1: Write serialization and ordering tests**

```python
# tests/runtime/test_models.py
from open_cortex.runtime.models import (
    ChatMessage,
    ExperimentScenario,
    RuntimeEvent,
    RuntimePhase,
    RuntimeState,
)


def test_runtime_event_serializes_enum_values(healthy_runtime_state):
    event = RuntimeEvent(
        request_id="req-1",
        sequence=2,
        kind="token",
        state=healthy_runtime_state,
        text_delta="hello",
    )

    payload = event.to_dict()

    assert payload["sequence"] == 2
    assert payload["state"]["phase"] == "decode"
    assert payload["text_delta"] == "hello"


def test_scenarios_have_stable_wire_values():
    assert [scenario.value for scenario in ExperimentScenario] == [
        "normal",
        "long_context",
        "memory_pressure",
        "slow_decode",
        "context_collapse",
    ]
```

Add `tests/conftest.py` with a complete `healthy_runtime_state` fixture using
the dataclasses below.

- [ ] **Step 2: Run the test and confirm the missing module failure**

Run: `uv run pytest tests/runtime/test_models.py -v`

Expected: FAIL with `ModuleNotFoundError: open_cortex.runtime.models`.

- [ ] **Step 3: Implement the exact public types**

`runtime/models.py` must define:

```python
class RuntimePhase(StrEnum):
    IDLE = "idle"
    PREFILL = "prefill"
    DECODE = "decode"
    RECOVERY = "recovery"
    ERROR = "error"


class Severity(StrEnum):
    QUIET = "quiet"
    HEALTHY = "healthy"
    BUSY = "busy"
    STRAINED = "strained"
    DEGRADED = "degraded"


class ExperimentScenario(StrEnum):
    NORMAL = "normal"
    LONG_CONTEXT = "long_context"
    MEMORY_PRESSURE = "memory_pressure"
    SLOW_DECODE = "slow_decode"
    CONTEXT_COLLAPSE = "context_collapse"


@dataclass(frozen=True)
class ChatMessage:
    id: str
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: int = 384
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass(frozen=True)
class RawMetrics:
    prompt_tokens: int | None = None
    generated_tokens: int = 0
    kv_usage_ratio: float | None = None
    kv_cache_tokens: int | None = None
    prompt_tokens_per_second: float | None = None
    tokens_per_second: float | None = None
    ttft_ms: float | None = None
    last_token_interval_ms: float | None = None
    cache_evictions: int | None = None
    requests_processing: int | None = None
    requests_deferred: int | None = None
    network_ms: float | None = None
```

Also implement the four organ dataclasses, `RuntimeState`, and `RuntimeEvent`
exactly as specified in
`docs/superpowers/specs/2026-06-09-open-cortex-mvp-design.md`.
Add `network_ms: float | None = None` to `RuntimeEvent` so remote transport
latency remains event evidence rather than being misrepresented as an engine
organ metric.
Implement `to_dict()` with `dataclasses.asdict()` followed by recursive enum
conversion. `RuntimeEvent.kind` is a `Literal` of the seven event names in the
specification.

- [ ] **Step 4: Define the backend protocol**

```python
# src/open_cortex/backends/base.py
from collections.abc import Iterator, Sequence
from typing import Protocol

from open_cortex.runtime.models import (
    ChatMessage,
    ExperimentScenario,
    GenerationConfig,
    RuntimeEvent,
)


class InferenceBackend(Protocol):
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig,
        experiment: ExperimentScenario,
    ) -> Iterator[RuntimeEvent]:
        raise NotImplementedError

    def supported_experiments(self) -> dict[ExperimentScenario, str | None]:
        raise NotImplementedError
```

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/runtime/test_models.py -v`

Expected: PASS.

```bash
git add src/open_cortex/runtime src/open_cortex/backends tests
git commit -m "feat: define runtime event protocol"
```

### Task 3: Derive Semantic State From Metrics and Benchmarks

**Files:**
- Create: `src/open_cortex/runtime/thresholds.py`
- Create: `src/open_cortex/runtime/state_engine.py`
- Create: `tests/runtime/test_state_engine.py`

- [ ] **Step 1: Write table-driven state tests**

Cover these exact cases:

```python
@pytest.mark.parametrize(
    ("ratio", "severity"),
    [(0.59, Severity.HEALTHY), (0.60, Severity.BUSY),
     (0.80, Severity.STRAINED), (0.95, Severity.DEGRADED)],
)
def test_kv_thresholds(ratio, severity, state_engine):
    state = state_engine.derive(
        sequence=1,
        phase=RuntimePhase.DECODE,
        metrics=RawMetrics(kv_usage_ratio=ratio),
        forgotten_message_ids=(),
    )
    assert state.working_memory.severity is severity


def test_context_loss_is_degraded_even_after_usage_recovers(state_engine):
    state = state_engine.derive(
        sequence=3,
        phase=RuntimePhase.RECOVERY,
        metrics=RawMetrics(prompt_tokens=1800),
        forgotten_message_ids=("message-1",),
    )
    assert state.context_window.severity is Severity.DEGRADED
    assert state.context_window.used_tokens == 1800
```

Add tests proving decode severity is relative to a benchmark baseline and that
missing values stay `None`.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/runtime/test_state_engine.py -v`

Expected: FAIL because `RuntimeStateEngine` is undefined.

- [ ] **Step 3: Implement thresholds and profile types**

```python
# src/open_cortex/runtime/thresholds.py
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkProfile:
    model_name: str
    context_size: int
    median_ttft_ms: float
    median_tokens_per_second: float
    median_prompt_tokens_per_second: float


@dataclass(frozen=True)
class Thresholds:
    kv_busy: float = 0.60
    kv_strained: float = 0.80
    kv_degraded: float = 0.95
    context_busy: float = 0.60
    context_strained: float = 0.80
    context_near_limit: float = 0.95
    slow_decode_ratio: float = 0.50
    strained_ttft_ratio: float = 2.0
```

- [ ] **Step 4: Implement `RuntimeStateEngine.derive`**

The constructor accepts `model_name`, `context_size`, `Thresholds`, and an
optional `BenchmarkProfile`. The mapping must obey:

```python
kv severity = ratio bands from Thresholds
context degraded = bool(forgotten_message_ids)
context severity otherwise = prompt_tokens / context_size bands
token stream strained = measured tps < baseline * slow_decode_ratio
engine strained = measured ttft > baseline * strained_ttft_ratio
```

Labels are stable semantic keys, not translated display strings:
`memory_holding`, `memory_busy`, `memory_fragmenting`, `memory_overloaded`,
`context_open`, `context_filling`, `context_compressed`, `context_forgotten`,
`flow_smooth`, `flow_slowing`, `flow_stalled`, `engine_idle`,
`engine_prefill`, `engine_decoding`, `engine_strained`, and `engine_error`.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run pytest tests/runtime/test_state_engine.py -v
uv run pytest tests/runtime -v
```

Expected: PASS.

```bash
git add src/open_cortex/runtime tests/runtime
git commit -m "feat: derive cognitive runtime states"
```

### Task 4: Build the Deterministic Simulator Contract

**Files:**
- Create: `src/open_cortex/backends/simulated.py`
- Create: `tests/backends/test_simulated.py`

- [ ] **Step 1: Write one contract test per experiment**

Assert the event sequence and the primary affected signal:

```python
EXPECTED_NON_TOKEN_KINDS = [
    "request_started",
    "prefill_progress",
    "first_token",
    "request_completed",
]


def test_slow_decode_only_degrades_token_cadence(simulated_backend):
    events = list(
        simulated_backend.stream_chat(
            [ChatMessage("m1", "user", "Explain KV cache.")],
            GenerationConfig(max_tokens=24),
            ExperimentScenario.SLOW_DECODE,
        )
    )

    assert [
        event.kind for event in events if event.kind != "token"
    ] == EXPECTED_NON_TOKEN_KINDS
    first = next(event for event in events if event.kind == "first_token")
    assert first.state.token_stream.severity is Severity.STRAINED
    assert first.state.working_memory.severity is Severity.HEALTHY
    assert first.state.context_window.severity is Severity.HEALTHY
```

For context collapse, require a `context_forgotten` event, non-empty forgotten
IDs, and recovered context usage. For memory pressure, require memory strain but
healthy token/context organs. For long context, require high context usage and
long TTFT but normal decode speed.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/backends/test_simulated.py -v`

Expected: FAIL because `SimulatedBackend` is undefined.

- [ ] **Step 3: Implement deterministic fixture timelines**

Use fixed metrics and response fragments, with no random values or sleeps.
Every emitted event passes through `RuntimeStateEngine.derive`. Use these
scenario signatures:

| Scenario | Prompt tokens | KV | TTFT | TPS | Forgotten |
| --- | ---: | ---: | ---: | ---: | --- |
| Normal | 820 | 0.42 | 380 ms | 31 | none |
| Long context | 3500 | 0.68 | 1900 ms | 29 | none |
| Memory pressure | 1500 | 0.87 | 620 ms | 28 | none |
| Slow decode | 1100 | 0.48 | 480 ms | 6 | none |
| Context collapse | 1900 after release | 0.55 | 650 ms | 25 | oldest ID |

The assistant text demonstrates the effect instead of narrating UI controls.
For example, context collapse says that the recent thread is available but
cannot correctly recall the first user request.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/backends/test_simulated.py -v`

Expected: PASS.

```bash
git add src/open_cortex/backends/simulated.py tests/backends
git commit -m "feat: add deterministic runtime simulator"
```

### Task 5: Implement Safe Rendering and Localization

**Files:**
- Create: `src/open_cortex/ui/__init__.py`
- Create: `src/open_cortex/ui/localization.py`
- Create: `src/open_cortex/ui/render.py`
- Create: `tests/ui/test_render.py`
- Create: `tests/ui/test_localization.py`

- [ ] **Step 1: Write escaping, boundary, and translation tests**

```python
def test_chat_renderer_escapes_user_html():
    html = render_chat(
        [ChatMessage("m1", "user", "<script>alert(1)</script>")],
        forgotten_message_ids=(),
        locale="en",
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_boundary_precedes_first_active_message():
    html = render_chat(
        [
            ChatMessage("old", "user", "old"),
            ChatMessage("active", "assistant", "active"),
        ],
        forgotten_message_ids=("old",),
        locale="en",
    )
    assert html.index("outside-active-context") < html.index("active-context-boundary")
    assert html.index("active-context-boundary") < html.index('data-message-id="active"')


def test_chinese_copy_contains_semantic_state():
    assert translate("memory_fragmenting", "zh") == "工作记忆正在碎片化"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/ui/test_render.py tests/ui/test_localization.py -v`

Expected: FAIL because the UI modules do not exist.

- [ ] **Step 3: Implement a closed translation catalog**

`localization.py` must contain English and Chinese strings for all state labels,
five experiments, header controls, `Active Context Boundary`,
`outside active context`, metric units, disconnected state, retry, and
simulator disclosure. `translate(key, locale)` falls back to English and raises
`KeyError` for an unknown key during tests.

- [ ] **Step 4: Implement render functions**

```python
def render_chat(
    messages: Sequence[ChatMessage],
    forgotten_message_ids: Sequence[str],
    locale: str,
    incomplete_message_id: str | None = None,
) -> str:
    forgotten = set(forgotten_message_ids)
    chunks = ['<div class="chat-thread">']
    boundary_written = False
    for message in messages:
        if message.id not in forgotten and forgotten and not boundary_written:
            chunks.append(
                '<div class="active-context-boundary">'
                f'<span>{html.escape(translate("active_context_boundary", locale))}</span>'
                "</div>"
            )
            boundary_written = True
        classes = ["chat-message", f"role-{message.role}"]
        if message.id in forgotten:
            classes.append("outside-active-context")
        if message.id == incomplete_message_id:
            classes.append("is-incomplete")
        chunks.append(
            f'<article class="{" ".join(classes)}" '
            f'data-message-id="{html.escape(message.id, quote=True)}">'
            f"<p>{html.escape(message.content)}</p></article>"
        )
    chunks.append("</div>")
    return "".join(chunks)


def render_state_payload(
    event: RuntimeEvent,
    locale: str,
    backend_mode: str,
    network_ms: float | None,
) -> str:
    payload = {
        "requestId": event.request_id,
        "sequence": event.sequence,
        "kind": event.kind,
        "locale": locale,
        "backendMode": backend_mode,
        "networkMs": network_ms,
        "state": event.to_dict()["state"],
    }
    encoded = html.escape(json.dumps(payload, ensure_ascii=False), quote=True)
    return f'<div class="runtime-state-payload" data-state="{encoded}"></div>'
```

`render_chat` places forgotten messages inside
`.outside-active-context`, inserts one `.active-context-boundary` before the
first active message, and never emits unescaped message content.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/ui -v`

Expected: PASS.

```bash
git add src/open_cortex/ui tests/ui
git commit -m "feat: render bilingual runtime state safely"
```

### Task 6: Rebuild the Approved Product Surface as Formal Assets

**Files:**
- Create: `src/open_cortex/ui/assets/open_cortex.css`
- Create: `src/open_cortex/ui/assets/open_cortex.js`
- Create: `tests/ui/test_assets.py`
- Create: `tests/js/open_cortex.test.js`

- [ ] **Step 1: Write structural CSS and JavaScript tests**

The Python test reads both asset files and requires selectors/IDs for:
`#conversation-panel`, `#runtime-panel`, `#workspace-divider`,
`#conversation-restore`, `#cortex-core`, `#working-memory`,
`#context-window`, `#token-stream`, `#engine-state`, and
`#runtime-state-bridge`.

The Node test requires the browser script and checks pure functions:

```javascript
const test = require("node:test");
const assert = require("node:assert/strict");
require("../../src/open_cortex/ui/assets/open_cortex.js");

const controller = globalThis.OpenCortexController;

test("splitter collapses below threshold", () => {
  assert.deepEqual(
    controller.resolveSplit(148, { min: 280, max: 680, collapse: 220 }),
    { collapsed: true, width: 0 }
  );
});

test("stale state is rejected", () => {
  assert.equal(controller.isNewer({ requestId: "r", sequence: 4 }, {
    requestId: "r", sequence: 3
  }), false);
});
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/ui/test_assets.py -v
node --test tests/js/open_cortex.test.js
```

Expected: FAIL because the assets do not exist.

- [ ] **Step 3: Implement the formal visual system**

The CSS must use these restrained tokens:

```css
:root {
  --oc-bg: #070b12;
  --oc-panel: rgba(13, 19, 29, 0.82);
  --oc-panel-raised: rgba(17, 25, 38, 0.94);
  --oc-border: rgba(148, 163, 184, 0.16);
  --oc-text: #e8eef7;
  --oc-muted: #8290a3;
  --oc-cyan: #5cd6e8;
  --oc-teal: #38bfa7;
  --oc-orange: #f2a65a;
  --oc-red: #ff5f73;
  --oc-magenta: #df6cff;
  --conversation-width: 42%;
}
```

Implement a near-black radial background, thin borders, no global glow, and
unique organ structures:

- Working Memory: a grid of KV cells; `.is-fragmenting` offsets, dims, and
  reorders cells with an irregular animation.
- Context Window: a horizontal memory tape with a fill mask and released oldest
  segments; `.is-forgotten` uses one red boundary, not a red panel.
- Token Stream: conduit particles whose CSS duration and opacity are set by JS;
  `.is-slowing` uses stop-burst-stop keyframes.
- Engine State: compact rhythm trace and TTFT evidence.
- Cortex Core: layered rings and four named connections. Prefill charges the
  outer ring, decode pulses regularly, memory pressure disturbs only the memory
  connection, slow decode breaks only the token connection, and context loss
  dims only the context connection.

The conversation panel uses modern product spacing and sentence case. Reserve
uppercase for short machine labels. Hide duplicate descriptions and secondary
metrics at widths below 1180 px.

- [ ] **Step 4: Implement the browser controller**

Expose these pure functions on `globalThis.OpenCortexController`:

```javascript
resolveSplit(candidate, limits)
isNewer(next, current)
stateClasses(payload)
formatMetric(value, unit, digits)
```

On `DOMContentLoaded`, the controller:

1. Observes `#runtime-state-bridge` with `MutationObserver`.
2. Parses `.runtime-state-payload.dataset.state`.
3. Rejects lower sequence values for the same request.
4. Updates text, CSS custom properties, severity classes, phase classes, and
   only the affected core connection.
5. Handles pointer drag on `#workspace-divider`.
6. Collapses at 220 px, restores to the last width, and toggles on double click.
7. Stores width and locale in `localStorage`.
8. Honors `prefers-reduced-motion`.

Guard browser initialization with `if (typeof document !== "undefined")` so
Node can test the pure functions without a DOM.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run pytest tests/ui/test_assets.py -v
node --test tests/js/open_cortex.test.js
```

Expected: PASS.

```bash
git add src/open_cortex/ui/assets tests/ui/test_assets.py tests/js
git commit -m "feat: add living observatory visual system"
```

### Task 7: Wire a Complete Simulator-Driven Gradio Product

**Files:**
- Create: `src/open_cortex/controller.py`
- Create: `src/open_cortex/ui/app.py`
- Create: `tests/test_controller.py`
- Create: `tests/ui/test_app.py`

- [ ] **Step 1: Write orchestration tests**

Test that one submitted turn:

1. Adds the user message immediately.
2. Appends assistant deltas to one assistant message.
3. Yields chat HTML and runtime payload from the same event.
4. Keeps monotonically increasing sequence values.
5. Preserves partial output on `request_failed`.

Also test `build_app(Settings.from_env())` returns a `gr.Blocks`.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_controller.py tests/ui/test_app.py -v`

Expected: FAIL because the controller and app are undefined.

- [ ] **Step 3: Implement session orchestration**

```python
@dataclass
class SessionState:
    messages: list[ChatMessage] = field(default_factory=list)
    forgotten_message_ids: tuple[str, ...] = ()
    locale: str = "en"
    experiment: ExperimentScenario = ExperimentScenario.NORMAL
    incomplete_message_id: str | None = None


class ChatController:
    def __init__(self, backend: InferenceBackend, backend_mode: str):
        self.backend = backend
        self.backend_mode = backend_mode

    def stream_turn(
        self,
        prompt: str,
        session: SessionState,
    ) -> Iterator[tuple[str, str, SessionState, str]]:
        prompt = prompt.strip()
        if not prompt:
            return
        session.messages.append(ChatMessage(uuid.uuid4().hex, "user", prompt))
        assistant = ChatMessage(uuid.uuid4().hex, "assistant", "")
        session.messages.append(assistant)
        session.incomplete_message_id = assistant.id
        for event in self.backend.stream_chat(
            session.messages[:-1],
            GenerationConfig(),
            session.experiment,
        ):
            if event.text_delta:
                assistant = replace(
                    assistant,
                    content=assistant.content + event.text_delta,
                )
                session.messages[-1] = assistant
            if event.forgotten_message_ids:
                session.forgotten_message_ids = event.forgotten_message_ids
            if event.kind == "request_completed":
                session.incomplete_message_id = None
            yield (
                render_chat(
                    session.messages,
                    session.forgotten_message_ids,
                    session.locale,
                    session.incomplete_message_id,
                ),
                render_state_payload(
                    event,
                    session.locale,
                    self.backend_mode,
                    event.network_ms,
                ),
                session,
                "",
            )
```

The tuple is `chat_html`, `state_payload_html`, updated session, and cleared
composer value. Generate message/request IDs with `uuid.uuid4().hex`. Do not
sleep in the controller; simulator timing belongs in the simulator and live
timing comes from SSE arrival.

- [ ] **Step 4: Build the Gradio layout**

Use `gr.Blocks(fill_height=True, css_paths=[asset_dir / "open_cortex.css"],
js=asset_text)` with:

- Compact header: identity, backend status, language button.
- A `gr.Row` workspace.
- Left `gr.Column(elem_id="conversation-panel")` containing experiment label,
  `gr.HTML(elem_id="chat-transcript")`, and composer controls.
- `gr.HTML` divider and restore control.
- Right `gr.Column(elem_id="runtime-panel")` containing model/phase controls,
  static observatory HTML, and compact evidence strip.
- A visually hidden, but DOM-mounted,
  `gr.HTML(elem_id="runtime-state-bridge")`.
- `gr.State(SessionState())`.

The static observatory HTML contains all IDs required in Task 6; only the bridge
is replaced while streaming so core animations are not restarted per token.

- [ ] **Step 5: Wire events**

`Textbox.submit` and Send button call the same generator. Scenario buttons set
`session.experiment`. The language button toggles `en`/`zh`, rerenders the chat,
and sends a bridge payload that updates static labels. Call `.queue()` so
streaming generators work on Spaces.

- [ ] **Step 6: Run the simulator and inspect all scenarios**

Run: `uv run python app.py`

Expected: the page opens in simulator mode; all five scenarios are visibly
different, context collapse adds the boundary, and splitter collapse/restore
works.

- [ ] **Step 7: Verify and commit**

Run:

```bash
uv run pytest -v
node --test tests/js/open_cortex.test.js
uv run ruff check .
```

Expected: PASS.

```bash
git add app.py src/open_cortex/controller.py src/open_cortex/ui tests
git commit -m "feat: deliver simulator-driven OpenCortex UI"
```

### Task 8: Parse llama.cpp Metrics Without Inventing Values

**Files:**
- Create: `src/open_cortex/backends/metrics.py`
- Create: `tests/backends/test_metrics.py`
- Create: `tests/fixtures/llama_metrics.txt`

- [ ] **Step 1: Add a representative Prometheus fixture and failing tests**

Fixture metrics:

```text
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 4120
llamacpp:tokens_predicted_total 830
llamacpp:prompt_tokens_seconds 117.5
llamacpp:predicted_tokens_seconds 29.7
llamacpp:kv_cache_usage_ratio 0.63
llamacpp:kv_cache_tokens 2580
llamacpp:requests_processing 1
llamacpp:requests_deferred 0
```

Assert exact mapping into `RawMetrics`, plus a second test where absent metrics
remain `None`.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/backends/test_metrics.py -v`

Expected: FAIL because `parse_llama_metrics` is undefined.

- [ ] **Step 3: Implement the parser**

Use `prometheus_client.parser.text_string_to_metric_families`. Match exact
metric names, read the first unlabeled aggregate sample, convert the KV ratio
to a ratio rather than a percentage, and never default an absent metric to
zero.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/backends/test_metrics.py -v`

Expected: PASS.

```bash
git add src/open_cortex/backends/metrics.py tests/backends tests/fixtures
git commit -m "feat: parse llama.cpp runtime telemetry"
```

### Task 9: Implement Exact Prompt Formatting, Tokenization, and Forgetting

**Files:**
- Create: `src/open_cortex/runtime/context.py`
- Create: `tests/runtime/test_context.py`

- [ ] **Step 1: Write HTTP-backed context policy tests**

Mock `/apply-template` to return the model-formatted prompt and `/tokenize` to
return token IDs. Assert:

- Messages stay intact when formatted prompt tokens fit the budget.
- Oldest non-system turns are removed as user/assistant pairs.
- The latest user message is never removed.
- Returned forgotten IDs exactly match removed visible messages.
- The final formatted prompt token count is at or below
  `context_size - max_generation_tokens`.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/runtime/test_context.py -v`

Expected: FAIL because `LlamaContextPolicy` is undefined.

- [ ] **Step 3: Implement the server-backed policy**

```python
@dataclass(frozen=True)
class PreparedPrompt:
    prompt: str
    prompt_tokens: int
    active_messages: tuple[ChatMessage, ...]
    forgotten_message_ids: tuple[str, ...]


class ContextBudgetError(ValueError):
    pass


class LlamaContextPolicy:
    def __init__(self, client: httpx.Client, context_size: int):
        self.client = client
        self.context_size = context_size

    def prepare(
        self,
        messages: Sequence[ChatMessage],
        max_generation_tokens: int,
    ) -> PreparedPrompt:
        active = list(messages)
        forgotten: list[str] = []
        budget = self.context_size - max_generation_tokens
        while True:
            response = self.client.post(
                "/apply-template",
                json={
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in active
                    ]
                },
            )
            response.raise_for_status()
            prompt = response.json()["prompt"]
            token_response = self.client.post(
                "/tokenize",
                json={"content": prompt, "add_special": False},
            )
            token_response.raise_for_status()
            prompt_tokens = len(token_response.json()["tokens"])
            if prompt_tokens <= budget:
                return PreparedPrompt(
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    active_messages=tuple(active),
                    forgotten_message_ids=tuple(forgotten),
                )
            removable = next(
                (
                    index
                    for index, message in enumerate(active[:-1])
                    if message.role != "system"
                ),
                None,
            )
            if removable is None:
                raise ContextBudgetError(
                    "latest message cannot fit within the configured context"
                )
            removed = active.pop(removable)
            forgotten.append(removed.id)
            if (
                removed.role == "user"
                and removable < len(active) - 1
                and active[removable].role == "assistant"
            ):
                forgotten.append(active.pop(removable).id)
```

For each candidate history, POST message dictionaries to `/apply-template`,
then POST the returned string to `/tokenize` with `add_special=False`. Remove
the oldest complete conversational turn until it fits. This uses the active
model's real chat template and tokenizer; no character-count estimate is
allowed.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/runtime/test_context.py -v`

Expected: PASS.

```bash
git add src/open_cortex/runtime/context.py tests/runtime/test_context.py
git commit -m "feat: enforce model-accurate context boundaries"
```

### Task 10: Stream Live llama.cpp Events

**Files:**
- Create: `src/open_cortex/backends/llama_cpp.py`
- Create: `tests/backends/test_llama_cpp.py`
- Create: `tests/fixtures/completion_stream.sse`

- [ ] **Step 1: Add an SSE fixture and event contract tests**

The fixture must contain prompt progress events, content events, and a final
timings object. Test:

- `request_started` precedes `prefill_progress`.
- `first_token` is emitted exactly once.
- token text is preserved exactly.
- TTFT uses `time.monotonic()` at the request boundary.
- rolling TPS and last-token interval use actual event arrival times.
- `/metrics` failure leaves KV evidence `None` while chat continues.
- interrupted SSE emits `request_failed` with partial text retained.
- context-policy forgotten IDs produce `context_forgotten`.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/backends/test_llama_cpp.py -v`

Expected: FAIL because `LlamaCppBackend` is undefined.

- [ ] **Step 3: Implement the client around llama.cpp native endpoints**

`LlamaCppBackend` owns one `httpx.Client`, `LlamaContextPolicy`,
`RuntimeStateEngine`, and a monotonic clock dependency. For generation:

1. Prepare the exact prompt with `/apply-template` and `/tokenize`.
2. Snapshot `/metrics`.
3. POST `/completion` with:

```json
{
  "prompt": "<formatted prompt>",
  "stream": true,
  "return_progress": true,
  "cache_prompt": true,
  "n_predict": 384,
  "temperature": 0.7,
  "top_p": 0.9
}
```

4. Parse `data:` SSE lines and ignore `[DONE]`.
5. Emit prefill progress from `prompt_progress.total`,
   `prompt_progress.processed`, and `prompt_progress.time_ms`.
6. Throttle `/metrics` sampling to at most once every 250 ms.
7. Measure TTFT and token intervals at the OpenCortex boundary.
8. Derive each state through `RuntimeStateEngine`.
9. Emit completion or failure without fabricating unsupported evidence.

Scenario preparation must be explicit:

- Long context adds a deterministic hidden reference document and uses
  `/apply-template` plus `/tokenize` to size the actual foreground prompt to
  80-90% of the configured context. The document remains in the generated
  request, so high prefill and TTFT are real.
- Context collapse first applies a deterministic overflow document to identify
  which oldest visible turns no longer fit. It then removes the overflow
  document and those forgotten turns before generation. The generation request
  therefore has a recovered context level while the UI accurately marks the
  visible turns excluded from the active prompt.
- Memory pressure and slow decode do not pad the foreground context.

The base URL is used identically for local and remote modes. If remote mode is
enabled, separately capture HTTP connect/round-trip time in `network_ms`; never
label it inference latency.

- [ ] **Step 4: Expose support status**

Normal, long-context, and context-collapse are always supported. Memory
pressure and slow decode return a concise disabled reason unless
`enable_load_experiments` is true.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run pytest tests/backends/test_llama_cpp.py -v
uv run pytest tests/backends -v
```

Expected: PASS.

```bash
git add src/open_cortex/backends/llama_cpp.py tests/backends tests/fixtures
git commit -m "feat: stream truthful llama.cpp runtime events"
```

### Task 11: Add Controlled, Isolated Experiment Load

**Files:**
- Create: `src/open_cortex/backends/load.py`
- Create: `tests/backends/test_load.py`
- Modify: `src/open_cortex/backends/llama_cpp.py`

- [ ] **Step 1: Write lifecycle and scenario-isolation tests**

Using `pytest-httpx`, assert:

- Normal and long-context scenarios create no background requests.
- Memory pressure starts long-prompt, cache-retaining requests.
- Slow decode starts several short-prompt generation requests.
- Every background request is cancelled/joined after foreground completion.
- One session cannot stop another session's load handle.
- Disabled settings prevent all load.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/backends/test_load.py -v`

Expected: FAIL because `ControlledLoadGenerator` is undefined.

- [ ] **Step 3: Implement bounded load handles**

```python
@dataclass
class LoadHandle:
    request_id: str
    stop_event: threading.Event
    futures: list[Future[None]]

    def close(self) -> None:
        self.stop_event.set()
        for future in self.futures:
            future.cancel()
        wait(self.futures, timeout=5)


class ControlledLoadGenerator:
    def __init__(self, base_url: str, timeout: float, max_workers: int = 4):
        self.base_url = base_url
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def start(
        self,
        request_id: str,
        scenario: ExperimentScenario,
    ) -> LoadHandle | None:
        if scenario is ExperimentScenario.MEMORY_PRESSURE:
            jobs = [(MEMORY_PRESSURE_PROMPT, 96)] * 2
        elif scenario is ExperimentScenario.SLOW_DECODE:
            jobs = [(SLOW_DECODE_PROMPT, 512)] * 4
        else:
            return None
        stop_event = threading.Event()
        futures = [
            self.executor.submit(
                run_load_request,
                self.base_url,
                self.timeout,
                prompt,
                n_predict,
                stop_event,
            )
            for prompt, n_predict in jobs
        ]
        return LoadHandle(request_id, stop_event, futures)
```

Memory pressure uses two long prompts with `cache_prompt=true` and bounded
`n_predict`. Slow decode uses four short-context requests with larger
`n_predict`. Use a dedicated `ThreadPoolExecutor(max_workers=4)`, explicit
timeouts, and per-request stop events. No load runs for unsupported scenarios.
Define the worker and prompts in the same module:

```python
MEMORY_PRESSURE_PROMPT = (
    "Retain the following numbered facts and wait for a recall question. "
    + " ".join(f"fact-{index}={index * 17}" for index in range(1200))
)
SLOW_DECODE_PROMPT = "Write a detailed but concise explanation of KV cache reuse."


def run_load_request(
    base_url: str,
    timeout: float,
    prompt: str,
    n_predict: int,
    stop_event: threading.Event,
) -> None:
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        with client.stream(
            "POST",
            "/completion",
            json={
                "prompt": prompt,
                "stream": True,
                "cache_prompt": True,
                "n_predict": n_predict,
            },
        ) as response:
            response.raise_for_status()
            for _line in response.iter_lines():
                if stop_event.is_set():
                    return
```

- [ ] **Step 4: Integrate cleanup with live streaming**

Start load immediately before the foreground `/completion`. Close it in a
`finally` block for success, cancellation, timeout, and parse errors. Display
only resulting llama.cpp metrics and foreground cadence; do not display the
requested worker count as runtime evidence.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/backends/test_load.py tests/backends/test_llama_cpp.py -v`

Expected: PASS and no worker threads remain alive.

```bash
git add src/open_cortex/backends/load.py src/open_cortex/backends/llama_cpp.py tests
git commit -m "feat: add isolated runtime stress experiments"
```

### Task 12: Select Backends and Complete Live UI Behavior

**Files:**
- Modify: `src/open_cortex/ui/app.py`
- Modify: `src/open_cortex/controller.py`
- Modify: `src/open_cortex/config.py`
- Create: `tests/ui/test_backend_selection.py`
- Create: `tests/ui/test_experiment_behavior.py`

- [ ] **Step 1: Write backend selection and experiment behavior tests**

Assert `simulated` selects `SimulatedBackend`, `llama_cpp` selects
`LlamaCppBackend`, and unknown values raise a startup error. For each scenario,
feed contract events through `ChatController` and assert:

| Scenario | Required visible effect |
| --- | --- |
| Normal | stable core and smooth token flow |
| Long context | filled tape and longer prefill; decode can recover |
| Memory pressure | fragmenting KV cells only |
| Slow decode | broken token conduit and naturally delayed text only |
| Context collapse | old messages dimmed, boundary visible, context tape released |

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/ui/test_backend_selection.py \
  tests/ui/test_experiment_behavior.py -v
```

Expected: FAIL until backend selection and scenario flags are wired.

- [ ] **Step 3: Implement backend factory and disabled controls**

Add `create_backend(settings)` to `ui/app.py`. Render disabled experiment
buttons with the exact reason returned by `supported_experiments()`. The header
must show `SIMULATED` in simulator mode and `LIVE` in llama.cpp mode.

- [ ] **Step 4: Make model behavior expressive**

Inject one short scenario system instruction only in live mode:

- Memory pressure: preserve recent details but admit uncertainty when asked
  about early details.
- Slow decode: no explanatory instruction; actual streaming timing is enough.
- Context collapse: answer only from active messages and do not claim access to
  forgotten turns.
- Long context: answer normally; do not narrate context pressure.

Do not tell the model to describe UI organs, colors, metrics, or experiments.

- [ ] **Step 5: Verify live behavior manually**

Start llama.cpp:

```bash
llama-server -m "$MODEL_GGUF" -c 4096 --metrics --host 127.0.0.1 --port 8080
```

Start OpenCortex:

```bash
OPEN_CORTEX_BACKEND=llama_cpp uv run python app.py
```

Expected: all supported experiments affect the intended organ and assistant
experience; metrics disappear rather than becoming synthetic when `/metrics`
is disabled.

- [ ] **Step 6: Verify and commit**

Run:

```bash
uv run pytest -v
node --test tests/js/open_cortex.test.js
uv run ruff check .
```

Expected: PASS.

```bash
git add src/open_cortex tests
git commit -m "feat: connect live runtime to observatory"
```

### Task 13: Benchmark the Model and Choose Context From Evidence

**Files:**
- Create: `src/open_cortex/runtime/benchmark.py`
- Create: `tests/runtime/test_benchmark.py`
- Create: `benchmarks/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Write aggregation and selection tests**

Test median aggregation for TTFT, prompt TPS, decode TPS, and peak KV usage.
Test selection across 2048, 4096, and 8192 contexts using these rules:

1. Bilingual prompts complete without error.
2. Peak KV usage remains below 80% under normal chat.
3. Median TTFT remains below the configured demo ceiling.
4. Highest context satisfying the first three wins.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/runtime/test_benchmark.py -v`

Expected: FAIL because benchmark functions are undefined.

- [ ] **Step 3: Implement the CLI**

`open-cortex-benchmark` accepts:

```text
--base-url
--model-name
--contexts 2048 4096 8192
--runs 3
--ttft-ceiling-ms 2500
--output benchmarks/<name>.json
```

For each context configuration, run one English prompt, one Chinese prompt,
one long-context prompt, and one 128-token decode. Store raw runs plus a
`BenchmarkProfile`. The CLI does not restart llama-server; print the exact
`llama-server -c <size>` command required for each pass and accept one context
per invocation when automated restart is unavailable.

- [ ] **Step 4: Document and perform the benchmark**

Run once for 2K, 4K, and 8K against the target hardware. Commit the resulting
small JSON profiles, not model files or raw prompts containing user data.
Update `OPEN_CORTEX_CONTEXT_SIZE` only after comparing the profiles.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/runtime/test_benchmark.py -v`

Expected: PASS.

```bash
git add src/open_cortex/runtime/benchmark.py tests/runtime/test_benchmark.py \
  benchmarks .gitignore
git commit -m "feat: benchmark runtime and select context size"
```

### Task 14: Package Hugging Face and Optional Modal Deployment

**Files:**
- Create: `deploy/huggingface/README.md`
- Create: `deploy/modal/modal_app.py`
- Create: `tests/deploy/test_modal_config.py`
- Modify: `README.md`

- [ ] **Step 1: Write a deployment configuration test**

Import `deploy/modal/modal_app.py` without starting Modal and assert its
llama-server command includes `--metrics`, `--host 0.0.0.0`, the configured
context, and no public `/slots` requirement. Keep command construction in a
pure `build_llama_command()` function so the test does not need Modal
credentials.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/deploy/test_modal_config.py -v`

Expected: FAIL because the deployment module is absent.

- [ ] **Step 3: Add the self-contained Space instructions**

Document:

- Space SDK `gradio`.
- Required environment variables.
- CPU command that launches llama-server before `python app.py`.
- Model license and download procedure.
- Why ZeroGPU is not assumed for llama.cpp.
- Health checks for `/health` and `/metrics`.
- Simulator mode for screenshots only, clearly labeled.

- [ ] **Step 4: Add the optional Modal backend**

`modal_app.py` must:

- Read model repo/file and context from secrets/environment.
- Cache the GGUF in a Modal Volume.
- Start `llama-server` with `--metrics`, bounded parallel slots, and the
  benchmark-selected context.
- Expose one authenticated or unguessable HTTPS endpoint to the Space.
- Keep `/slots` disabled publicly.
- Build the server command in a pure function tested in Step 1.

The Gradio application remains unchanged; only
`OPEN_CORTEX_LLAMA_BASE_URL` and `OPEN_CORTEX_REMOTE_BACKEND=true` differ.

- [ ] **Step 5: Complete the repository README**

Include:

- Product statement and screenshot location.
- `mise install`, `uv sync`, and simulator quick start.
- Local llama.cpp command.
- Live-mode environment table.
- Benchmark command.
- Five experiment semantics.
- Data truthfulness statement.
- Local, Space, and Space-plus-Modal deployment choices.
- Test commands.

- [ ] **Step 6: Verify and commit**

Run:

```bash
uv run pytest -v
node --test tests/js/open_cortex.test.js
uv run ruff check .
```

Expected: PASS.

```bash
git add README.md deploy tests/deploy
git commit -m "docs: package OpenCortex demo deployment"
```

### Task 15: Perform Acceptance Verification and Demo Rehearsal

**Files:**
- Create: `docs/demo-script.md`
- Create: `docs/verification.md`

- [ ] **Step 1: Run the complete automated suite**

Run:

```bash
uv run pytest --cov=open_cortex --cov-report=term-missing
node --test tests/js/open_cortex.test.js
uv run ruff check .
```

Expected: all tests pass, no uncovered critical branch in state derivation,
context forgetting, SSE failure, or load cleanup.

- [ ] **Step 2: Verify simulator acceptance**

Run: `OPEN_CORTEX_BACKEND=simulated uv run python app.py`

Check and record:

- First screen reads as an inference observatory.
- All five experiments differ without reading metric labels.
- Context collapse retains old messages and shows the active boundary.
- Splitter collapses below threshold, restores, and supports double click.
- English/Chinese switch updates static and semantic copy.
- Reduced-motion mode remains understandable.

- [ ] **Step 3: Verify live local acceptance**

Run the selected GGUF and benchmark context with `--metrics`. Check:

- Prefill and decode are distinct.
- Output timing follows actual SSE arrivals.
- KV, context, TPS, and TTFT values match source evidence.
- `/metrics` outage hides evidence.
- Stream interruption preserves partial text.
- Load experiment workers stop after each request.

- [ ] **Step 4: Verify remote acceptance**

Point the same UI at the Modal URL. Confirm backend network time is displayed
separately from TTFT and no code or visual changes are needed.

- [ ] **Step 5: Write the 3-5 minute demo script**

Use this order:

1. Normal chat establishes the core pulse and real evidence.
2. Long context visibly fills the tape and lengthens prefill.
3. Memory pressure fragments KV cells without slowing the token conduit.
4. Slow decode produces stop-burst-stop output with healthy memory.
5. Context collapse marks forgotten history and shows failed early recall.
6. Collapse the chat drawer to finish on the full observatory view.

- [ ] **Step 6: Record final evidence and commit**

`docs/verification.md` records date, model, quantization, hardware, selected
context, benchmark profile, test commands, Space URL, backend mode, and known
limitations.

```bash
git add docs/demo-script.md docs/verification.md
git commit -m "test: verify OpenCortex MVP acceptance"
```

## Completion Gate

Do not call the MVP complete until all conditions hold:

- Simulator and live backends emit the same event contract.
- No live metric is estimated or copied from simulator values.
- The five experiments have distinct causes, visual effects, and assistant
  behavior.
- Context forgetting uses the model's own chat template and tokenizer.
- Memory pressure and slow decode load are bounded, isolated, and disabled
  unless explicitly enabled.
- The UI works in English and Chinese, with a draggable/collapsible conversation
  panel.
- Local and remote llama.cpp use the same backend adapter.
- Automated tests, manual acceptance, and deployment documentation are complete.
