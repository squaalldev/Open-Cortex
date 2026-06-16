# Repository Instructions

## Project Direction

This repository currently contains OpenCortex, a FastAPI-based app with custom HTML/CSS/JS assets. Future work should transform the existing application into a Hugging Face Space demo called **Copy Review App**.

Do **not** create a brand-new project from scratch. Reuse the existing codebase, FastAPI app structure, static asset pipeline, runtime streaming patterns, and deployment setup where practical.

## Target Application

The final application must:

- Run on Hugging Face Spaces.
- Present itself as **Copy Review App**.
- Use the existing FastAPI/custom HTML/CSS/JS architecture where it remains practical.
- Call a Llama model through Hugging Face Inference Providers.
- Read the Hugging Face token from `HF_TOKEN` on the server side.
- Keep a demo or simulated mode as a fallback when `HF_TOKEN` is missing, invalid, or unavailable.

## Security Requirements

`HF_TOKEN` is a secret and must never be exposed.

Do not include `HF_TOKEN` or derived secret values in:

- Frontend JavaScript.
- Rendered HTML.
- CSS or static assets.
- Logs.
- Exceptions returned to users.
- API responses.
- Test snapshots or fixtures committed to the repo.

All calls that require `HF_TOKEN` must happen from server-side Python code only.

## Implementation Guidelines

- Prefer simple, readable Python over clever abstractions.
- Keep the app easy to run locally and easy to deploy to Hugging Face Spaces.
- Make incremental changes that reuse existing modules and routes where possible.
- Keep frontend code understandable and avoid unnecessary build tooling.
- Preserve a working simulated/demo path for development and unauthenticated Space previews.
- Update documentation when commands, environment variables, or app behavior change.
- Explain changed files clearly in final responses and pull request descriptions.

## Validation Expectations

After every implementation task, run syntax checks or tests if available. At minimum, run a Python syntax check for changed Python files when tests are not applicable.

Recommended validation commands:

```bash
uv run pytest
```

```bash
python -m compileall app.py open_cortex tests scripts
```

If `uv` dependencies are not installed yet, either install them first or use an available Python environment for syntax checks.

## Recommended Commands

### Install dependencies

```bash
uv sync
```

Fallback with pip:

```bash
python -m pip install -r requirements.txt
```

### Run the app locally

```bash
uv run open-cortex
```

Fallback:

```bash
python app.py
```

The app normally serves on:

```text
http://127.0.0.1:7860
```

### Run tests or syntax checks

```bash
uv run pytest
```

```bash
python -m compileall app.py open_cortex tests scripts
```
