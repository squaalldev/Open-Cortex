---
title: OpenCortex
emoji: 🧠
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
license: mit
tags:
  - build-small-hackathon
  - backyard-ai
  - off-brand
  - tiny-titan
  - best-demo
  - codex
  - llama-cpp
  - local-ai
  - ai-infra
  - observability
short_description: Feel a local LLM think,  work, and forget.
---


# OpenCortex

**Making LLM inference visible.**

OpenCortex is a real-time observatory for local LLM inference. It pairs a chat
interface with a living view of the runtime behind the answer: working memory,
context pressure, token flow, and engine health.

Most chat interfaces show only two things:

```text
user input -> assistant output
```

OpenCortex shows the machine in the middle.

```text
prompt prefill -> KV/cache pressure -> decode rhythm -> context boundary -> answer behavior
```

It is built for learners, local AI users, and AI infrastructure engineers who
want to understand why an LLM slows down, loops, forgets, or runs out of
context.

![OpenCortex runtime observatory](docs/assets/opencortex-hero.png)

## Why this exists

Small local models make AI personal again. But local inference is still a black
box for most users. When a response becomes slow or repetitive, the user rarely
knows whether the model is thinking, overloaded, stuck in a loop, or simply
running out of context.

OpenCortex turns runtime signals into a product experience:

| Runtime signal | Human-readable concept | Visual feedback |
| --- | --- | --- |
| KV/cache usage proxy | Working Memory | block pressure and fragmentation |
| Context tokens | Context Window | filling chamber and active boundary |
| Decode throughput | Token Stream | flowing, broken, or stalled token river |
| TTFT and queue evidence | Engine State | pulse, charging, recovery, hazard state |
| Repeated generated text | Thought Loop | red loop warning and irregular core pulse |

The goal is not to build another metrics dashboard. The goal is to let people
feel the hidden mechanics of inference while they chat.

## Demo links

- **Hugging Face Space:** https://huggingface.co/spaces/build-small-hackathon/open-cortex
- **Demo video:** https://youtu.be/edxZdttCf-s
- **Social post:** https://x.com/ZhaoJ90682/status/2066576258042155518

The Build Small Hackathon requires a deployed Gradio Space, a demo video, a
social post, and README tags for tracks and badges. This README is structured
for that submission flow.

The hosted Space runs in **simulated runtime mode** so judges can open the demo
without a private llama.cpp server. The local path below connects to live
llama.cpp metrics.

## What you can try

OpenCortex has one live mode and four built-in runtime experiments.

### 1. Live local chat

Ask the local model a question and watch the observatory react while the answer
streams. The UI listens to llama.cpp OpenAI-compatible streaming events,
timings, `/metrics`, and `/slots`.

### 2. Long context stress

Shows how prompt growth increases prefill work before generation begins.

### 3. Memory pressure

Shows working memory blocks fragmenting and reallocating as the runtime becomes
strained.

### 4. Slow decode

Shows token flow breaking into a stop-burst-stop rhythm when generation slows.

### 5. Context collapse

Shows the moment earlier turns leave the active context window. The chat history
stays visible, but OpenCortex marks the active context boundary so the user can
see what the model can no longer reliably use.

![Context boundary event](docs/assets/opencortex-context-collapse.png)

### 6. Thought loop detection

If generation begins repeating the same pattern, OpenCortex marks it as a
thought loop. The Token Stream and Cortex Core switch into a red hazard state.

![Thought loop detected](docs/assets/opencortex-thought-loop.png)

## How it works

OpenCortex is intentionally lightweight:

```text
Browser UI
  |
  | POST /api/chat
  v
FastAPI / Space app
  |
  | OpenAI-compatible streaming request
  v
llama.cpp llama-server
  |
  | /v1/chat/completions
  | /metrics
  | /slots
  v
Runtime events -> semantic state -> visual organs
```

The backend converts low-level runtime evidence into normalized events:

- `request_started`
- `first_token`
- `token`
- `context_collapse`
- `request_completed`
- `error`

The frontend consumes those events and updates the chat and observatory in the
same stream, so visual state and language output stay aligned.

## Runtime evidence

The MVP uses real llama.cpp evidence where available:

| Evidence | Source | Used for |
| --- | --- | --- |
| Time to first token | measured in Python client | Engine State |
| Prompt tokens | llama.cpp usage/timings | Context Window |
| Completion tokens | llama.cpp usage/timings | Token Stream |
| Prompt throughput | llama.cpp timings | Prefill evidence |
| Decode throughput | llama.cpp timings and metrics | Token Stream |
| Active slot context | llama.cpp `/slots` | Context and memory proxy |
| Processing/deferred requests | llama.cpp `/metrics` | Engine State |
| Repetition pattern | generated text heuristic | Thought loop hazard |

Working Memory is labeled as a **context-derived proxy** in this MVP. llama.cpp
does not expose a simple per-request KV cache percentage through the HTTP API we
use, so OpenCortex derives a truthful pressure estimate from active slot context
tokens and context size.

## Model

The current local demo uses:

```text
Qwen/Qwen2.5-1.5B-Instruct-GGUF
quantization: Q4_K_M
context: 2048 tokens in the local demo
runtime: llama.cpp llama-server
```

This keeps the submission inside the Build Small model limit and qualifies the
project for tiny-model storytelling. The interface is backend-neutral enough to
evolve toward Ollama, vLLM, or Modal-hosted llama.cpp later.

On Hugging Face Spaces, OpenCortex defaults to `OPEN_CORTEX_BACKEND=simulated`
when `SPACE_ID` is present. Set `OPEN_CORTEX_BACKEND=llama_cpp` only when the
Space can reach a running llama.cpp backend.

## Run locally

### 1. Install project dependencies

```bash
uv sync
```

### 2. Start llama.cpp

From your llama.cpp checkout:

```bash
llama-server \
  -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M \
  -c 2048 -t 8 -tb 8 \
  --metrics \
  --host 127.0.0.1 --port 8080
```

Check that the server is alive:

```bash
curl --noproxy '*' http://127.0.0.1:8080/health
```

Expected:

```json
{"status":"ok"}
```

### 3. Start OpenCortex

```bash
uv run open-cortex
```

Then open:

```text
http://127.0.0.1:7860
```

## Suggested demo script

Use this for the video:

1. Open the app and send a normal question.
2. Point out prefill, first-token latency, context usage, and token flow.
3. Run **Slow decode** to show token flow breaking.
4. Ask for a long story and show **Thought Loop Detected** when repetition
   appears.
5. Fill the context window and show **Context Window Full**.
6. Send one more message and show the earliest message leaving the active
   context boundary.
7. Close with the core claim: OpenCortex turns runtime internals into something
   users can see and reason about.

## Hackathon fit

OpenCortex targets the **Backyard AI** track: it is a practical tool for
learning and debugging local AI behavior on hardware you own.

It also targets these badges:

- **Off Brand**: custom runtime cockpit UI, not default Gradio components.
- **Tiny Titan**: built around a 1.5B local model.
- **Best Demo**: the experience is designed around a clear visual story.
- **Best Use of Codex**: the project was developed with Codex assistance and
  should be submitted with Codex-attributed commits in the connected repository.

Modal credits were used/planned for development exploration, but the current MVP
does not require Modal at runtime.

## Truthfulness and limitations

OpenCortex is a visualization layer over real runtime evidence, but it does not
claim to expose every internal tensor or true biological cognition.

Current limitations:

- KV cache usage is approximated from active context pressure.
- Thought loop detection is heuristic pattern detection over generated text.
- Context collapse is handled at the application layer by trimming oldest turns.
- The current demo is optimized for a single local runtime, not multi-user
  production serving.
- vLLM integration is a planned evolution, not part of v0.1.

These constraints are visible by design. The product shows both semantic states
and metric evidence so users can tell which parts are measured and which parts
are interpreted.

## Project structure

```text
app.py                         Space/local entry point
open_cortex/ui/app.py          HTTP app, event streaming, HTML shell
open_cortex/ui/assets/         Product UI CSS and browser controller
open_cortex/runtime/client.py  llama.cpp streaming client
open_cortex/runtime/metrics.py llama.cpp metrics and slots parser
open_cortex/runtime/events.py  Runtime event model
tests/                         Runtime and UI regression tests
```

## Development checks

```bash
uv run pytest -q
mise exec -- node --check open_cortex/ui/assets/open_cortex.js
```

## Submission checklist

- [ ] Deploy Space inside `build-small-hackathon/`.
- [ ] Confirm the Space launches as a Gradio-compatible submission.
- [ ] Add final Space URL to this README.
- [ ] Add demo video URL to this README.
- [ ] Add social post URL to this README.
- [ ] Run the Build Small README validator.
- [ ] Confirm the model is under 32B parameters.

## License

MIT
