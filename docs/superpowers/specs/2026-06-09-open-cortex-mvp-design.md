# OpenCortex MVP Design

## 1. Product Definition

OpenCortex is a real-time observatory for local LLM inference. It combines a
chat experience with a visual representation of the runtime that produces each
response.

The product must communicate within ten seconds that:

1. The model is actively processing and generating.
2. Memory, context, speed, and engine health are distinct runtime concerns.
3. These runtime states affect the user's conversation experience.
4. The evidence comes from the inference engine, not from decorative UI.

The product is positioned as:

- Chrome DevTools for LLM inference.
- A restrained sci-fi cockpit for a local AI engine.
- A transparent artificial cognitive machine.

It must not resemble a generic metrics dashboard, Grafana, or a default Gradio
application.

## 2. MVP Scope

### Included

- Streaming bilingual chat input and output.
- English and Chinese UI switching.
- A resizable conversation panel.
- Automatic conversation drawer collapse below a width threshold.
- A runtime observatory with:
  - Cortex Core.
  - Working Memory.
  - Context Window.
  - Token Stream.
  - Engine State.
- Runtime experiments:
  - Normal chat.
  - Long context stress.
  - Memory pressure.
  - Slow decode.
  - Context collapse.
- A development-only runtime simulator.
- A real llama.cpp backend with streamed inference and telemetry.
- Local and remote llama.cpp deployment modes.
- Hugging Face Gradio Space packaging.
- Modal-compatible remote deployment without coupling the UI to Modal.

### Excluded

- Multi-user production scheduling.
- Authentication and persistent user accounts.
- Historical dashboards and trace search.
- Runtime replay.
- A Prometheus or Grafana deployment.
- vLLM integration in v0.1.
- WebSocket infrastructure beyond what Gradio and streaming HTTP require.
- Claims about neural or biological equivalence.

## 3. Design Principles

### 3.1 Cognitive Naming, Engineering Structure

The interface uses human-readable cognitive concepts while rendering them as
engineering components:

| Product concept | Engineering form | Runtime evidence |
| --- | --- | --- |
| Cortex Core | Transparent processor/reactor core | Current phase and decode step |
| Working Memory | KV cell bank | KV/cache usage and cache events |
| Context Window | Sliding memory tape | Used and maximum context tokens |
| Token Stream | Decode pulse conduit | Tokens per second and token intervals |
| Engine State | Rhythm monitor | TTFT and engine health |

The interface must not depict literal human brain regions. That would imply
unsupported neuroscience mappings and weaken technical credibility.

### 3.2 Semantic State With Metric Evidence

Every organ shows:

1. A semantic state, such as `Memory holding`.
2. One primary metric, such as `63%`.
3. A unique visual structure that changes with the metric.

Raw evidence remains available in a compact telemetry strip. It does not compete
with the main state visualization.

### 3.3 Color Semantics

- Cyan: active computation, data flow, and healthy runtime activity.
- Orange: recoverable pressure or degraded performance.
- Red/magenta: irreversible user-visible loss, such as forgotten context.
- Slate: inactive structure, labels, boundaries, and secondary information.

Only the affected organ, connection, and corresponding part of the Cortex Core
receive warning color. The whole screen must not change color for a local fault.

### 3.4 Information Hierarchy

The interface uses three levels:

1. Semantic state.
2. Primary metric.
3. Compact raw engine evidence.

The MVP removes repeated subtitles, engineering aliases, metric explanations,
turn counters, decorative status labels, and duplicated telemetry.

## 4. Layout

The application has a compact product header and a two-panel workspace.

### Header

- OpenCortex identity.
- Backend connection status.
- English/Chinese switch.

### Conversation Panel

- Current experiment label.
- Conversation history.
- Active context boundary when older messages leave model-visible context.
- Streaming assistant message.
- Message composer.

### Divider

- A vertical divider supports horizontal panel resizing.
- The conversation width is constrained to a practical maximum.
- Below a collapse threshold, the conversation becomes a hidden drawer.
- A visible control restores the drawer.
- Double-clicking the divider toggles the drawer.

### Runtime Observatory

- Compact model identity, runtime phase, and experiment controls.
- Cortex Core in the center.
- Four organs connected to the core.
- A compact engine evidence strip.

The V6 visual prototype is the design reference. Prototype artifacts remain
outside version control under `.superpowers/`.

## 5. Runtime State Model

The frontend receives one normalized state independent of the inference
backend.

```python
from dataclasses import dataclass
from enum import StrEnum


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


@dataclass(frozen=True)
class WorkingMemoryState:
    severity: Severity
    label: str
    usage_percent: float
    cache_evictions: int | None


@dataclass(frozen=True)
class ContextWindowState:
    severity: Severity
    label: str
    used_tokens: int
    max_tokens: int
    forgotten_message_ids: tuple[str, ...]


@dataclass(frozen=True)
class TokenStreamState:
    severity: Severity
    label: str
    tokens_per_second: float
    last_token_interval_ms: float | None


@dataclass(frozen=True)
class EngineState:
    severity: Severity
    label: str
    ttft_ms: float | None
    prompt_tokens_per_second: float | None


@dataclass(frozen=True)
class RuntimeState:
    sequence: int
    phase: RuntimePhase
    model_name: str
    working_memory: WorkingMemoryState
    context_window: ContextWindowState
    token_stream: TokenStreamState
    engine: EngineState
```

`sequence` is monotonically increasing per request so the browser can ignore
late events.

## 6. Runtime Event Protocol

State snapshots are accompanied by streamed events:

```python
@dataclass(frozen=True)
class RuntimeEvent:
    request_id: str
    sequence: int
    kind: str
    state: RuntimeState
    text_delta: str = ""
    forgotten_message_ids: tuple[str, ...] = ()
```

Supported `kind` values:

- `request_started`
- `prefill_progress`
- `first_token`
- `token`
- `context_forgotten`
- `request_completed`
- `request_failed`

The Python layer serializes these objects to JSON-compatible dictionaries.
Gradio yields chat and runtime updates from the same generator so UI state and
text stay causally aligned.

## 7. State Derivation

Semantic states are deterministic mappings from measured values. Thresholds are
configuration, not hard-coded UI behavior.

Example defaults:

| Signal | Healthy | Busy | Strained | Degraded |
| --- | --- | --- | --- | --- |
| KV/cache usage | `< 60%` | `60-79%` | `80-94%` | `>= 95%` |
| Context usage | `< 60%` | `60-79%` | `80-94%` | forgotten context event |
| Decode speed | benchmark-dependent | below baseline | `< 50%` baseline | stalled |
| TTFT | benchmark-dependent | above baseline | `> 2x` baseline | timeout/error |

Speed and latency thresholds are relative to a benchmark profile recorded for
the active model and deployment. The profile is generated by the project
benchmark command and loaded at application startup. Fixed universal thresholds
would be misleading across CPU, GPU, local, and remote deployments.

## 8. Experiment Semantics

Experiments identify causes. Severity identifies impact. They are separate.

### Normal Chat

- Standard prompt and generation.
- Stable core pulse.
- All organs remain healthy unless real metrics indicate otherwise.

### Long Context Stress

- Submit a deliberately long prompt/history.
- Context tape fills toward the limit.
- Prefill duration and TTFT increase.
- Decode can remain healthy after prefill.
- Early messages visually fade but remain active until the engine reports they
  are no longer in context.

### Memory Pressure

- Create controlled slot/cache pressure with background requests that retain
  long contexts on the dedicated demo backend.
- Read the resulting cache and slot behavior from llama.cpp telemetry.
- KV cells visibly relocate, darken, and refill.
- The memory-to-core connection becomes irregular.
- Context and token flow remain visually stable unless their measured values
  also degrade.
- The assistant expresses uncertainty about early details instead of narrating
  the interface.

### Slow Decode

- Create controlled decode contention with parallel short-context generation
  requests on the dedicated demo backend.
- Use the measured foreground token arrival cadence as the displayed speed.
- Token pulses follow a stop-burst-stop rhythm.
- Text streaming uses the actual token arrival timing.
- Working memory and context remain visually healthy.

### Context Collapse

- Fill the active context until the oldest conversation content is no longer
  included in the model prompt.
- Show an `Active Context Boundary` in the chat history.
- Messages outside the active context remain in the UI but become visually
  marked as forgotten.
- The memory tape releases its oldest segments and returns to a stable usage
  level.
- The assistant demonstrates lost long-range recall.

The experiment controller may prepare prompts and backend configuration, but it
must not invent final runtime metrics.

The controlled load generator is part of the demo backend, not a production
scheduler. Experiments that cannot be isolated safely on the active deployment
are disabled.

## 9. Data Truthfulness

There are two explicit modes.

### Simulator Mode

- Used for frontend development, tests, screenshots, and offline demos.
- Clearly labeled as simulated.
- Produces deterministic fixture events.

### Live Mode

- Used for the submitted interactive inference demo.
- Displays only values measured by llama.cpp or directly measured at the
  OpenCortex request boundary.
- Unsupported metrics are omitted, not estimated.
- Network latency and backend inference latency remain distinguishable for a
  remote backend.

## 10. Architecture

```text
Gradio Blocks
  |
  +-- Custom HTML/CSS shell based on V6
  +-- Small browser controller
  |     +-- runtime animations
  |     +-- resizable split view
  |     +-- drawer behavior
  |     +-- localization
  |
  +-- ChatController
        |
        +-- RuntimeStateEngine
        |
        +-- InferenceBackend
              +-- SimulatedBackend
              +-- LlamaCppBackend
                    +-- local llama-server
                    +-- remote llama-server
              +-- future VllmBackend
```

### Frontend Strategy

Use Gradio Blocks for event wiring, queuing, session state, and Hugging Face
Space compatibility. Render the product surface with custom HTML and CSS. Use a
small JavaScript controller for interactions Gradio does not model well.

The browser controller must be framework-free for the MVP. Adding React would
create a second application lifecycle and unnecessary build complexity.

### Backend Interface

```python
class InferenceBackend(Protocol):
    def stream_chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig,
        experiment: ExperimentScenario,
    ) -> Iterator[RuntimeEvent]:
        ...
```

The UI and state engine depend only on this protocol.

### llama.cpp Integration

Use `llama-server` rather than embedding a Python binding. This provides:

- OpenAI-compatible streaming chat.
- Prompt progress and timing fields.
- `/metrics` when enabled.
- `/slots` for slot and cache inspection.
- Local and remote operation through the same HTTP client.

The integration layer correlates a chat request with snapshots taken before,
during, and after generation. Request-boundary timings are captured with a
monotonic clock.

## 11. Model and Context Selection

The initial candidate is a GGUF instruction model with at most 4B parameters,
with Llama 3.2 3B Instruct as the first benchmark target.

The final model and default context are chosen by benchmark, not assumption.
Benchmark configurations include 2K, 4K, and 8K context where supported.

Selection criteria:

- Time to first token.
- Decode tokens per second.
- Memory/cache headroom.
- Bilingual response quality.
- Ability to demonstrate context behavior within a short session.
- Hugging Face licensing and redistribution requirements.

## 12. Deployment

### Mode A: Self-Contained Hugging Face Space

- Gradio and llama.cpp run in the Space.
- Preferred when CPU or available Space hardware gives an acceptable
  experience.
- Strongest fit for a self-contained demo.

### Mode B: Hugging Face Space With Remote llama.cpp

- The Space hosts Gradio.
- Modal or another host runs llama.cpp.
- Selected by environment configuration.
- The UI separately reports frontend-to-backend network latency when relevant.

The repository supports both modes. The final submission mode is selected after
benchmarking rather than embedded into the product architecture.

ZeroGPU is not the primary llama.cpp deployment assumption because its
request-scoped GPU lifecycle is designed around supported PyTorch workloads.

## 13. Error Handling

- Backend unavailable: keep chat history, mark runtime disconnected, and show a
  retry action.
- Stream interrupted: retain partial output and mark the request incomplete.
- Metrics endpoint unavailable: continue chat, hide unavailable evidence, and
  never substitute estimates.
- Context exceeds configured input: apply the same deterministic prompt
  truncation policy used by the backend and emit `context_forgotten`.
- Stale browser event: discard by `request_id` and `sequence`.
- Experiment unsupported by active backend: disable it with a concise reason.

## 14. Testing

### Unit Tests

- Semantic threshold mapping.
- Context boundary and forgotten message selection.
- Event ordering and stale event rejection.
- Local/remote backend URL configuration.
- llama.cpp response and metrics parsing.

### Contract Tests

- Fixture streams for all runtime event types.
- SimulatedBackend and LlamaCppBackend conform to the same event protocol.
- Missing metrics remain `None` and do not become estimated values.

### UI Tests

- Each experiment activates only the intended organs and connection.
- Slow decode changes text timing.
- Context collapse displays the active context boundary.
- Dragging the divider resizes the panels.
- Dragging below the threshold collapses the conversation drawer.
- English/Chinese switching updates semantic labels.

### Integration Tests

- A small llama.cpp test model produces a complete prefill/decode event stream.
- Chat text and runtime state remain ordered under streaming.
- Remote mode reports network and backend timing separately.

## 15. Repository Shape

```text
open-cortex/
  app.py
  pyproject.toml
  README.md
  src/open_cortex/
    backends/
      base.py
      llama_cpp.py
      simulated.py
    runtime/
      models.py
      state_engine.py
      thresholds.py
    ui/
      app.py
      localization.py
      assets/
        open_cortex.css
        open_cortex.js
  tests/
    backends/
    runtime/
    ui/
  deploy/
    modal/
    huggingface/
```

Implementation replaces the current scaffold `main.py` with `app.py` as the
Space and local development entry point.

## 16. Acceptance Criteria

The MVP is accepted when:

1. A user can chat with a model and see streamed output.
2. Runtime state changes during prefill and decode.
3. Working Memory, Context Window, Token Stream, and Engine State are visually
   distinct without relying on labels alone.
4. The five experiments are distinguishable by affected organ, timing, and
   assistant behavior.
5. Context collapse leaves history visible while marking forgotten messages
   outside the active context boundary.
6. The conversation panel can be resized, collapsed by threshold, and restored.
7. Live mode displays only real measured evidence.
8. The same UI runs against local and remote llama.cpp through configuration.
9. The application deploys as a Gradio Hugging Face Space.
10. The first screen reads as an inference observatory rather than a chatbot or
    generic monitoring dashboard.
