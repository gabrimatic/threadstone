# Threadstone

[![CI](https://github.com/gabrimatic/threadstone/actions/workflows/ci.yml/badge.svg)](https://github.com/gabrimatic/threadstone/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/threadstone.svg)](https://pypi.org/project/threadstone/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-required-blue.svg)]()
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)]()
[![Local-first](https://img.shields.io/badge/local--first-MLX-green.svg)]()

Threadstone is offline terminal chat for local MLX language models on Apple Silicon. It starts a local `mlx-vlm` server for the model you choose, streams the answer into your terminal, keeps the conversation usable across crashes, and shuts the server down when you leave.

The useful part is ownership. The network is only for setup and model downloads. After that, chat runs against local snapshots with Hugging Face offline flags, no telemetry, no hosted API, and no account.

## Quick Start

Requirements: **macOS**, **Apple Silicon**, **Python 3.13+**, Homebrew, and about **12 GB** for the default model set.

```bash
git clone https://github.com/gabrimatic/threadstone.git
cd threadstone
./setup.sh
```

`setup.sh` creates `~/mlx-env`, installs the MLX runtime, downloads the configured Qwen model snapshots, installs the `threadstone` CLI, and writes shell helpers for `oracle`, `forge`, and `quench`.

Start a chat:

```bash
oracle
oracle 4B
oracle "be terse" 2B
```

Use the installed CLI directly:

```bash
threadstone --list-models
threadstone --doctor
threadstone "answer like a systems engineer" 9B
```

## Local Runtime

Threadstone uses the network during setup. Runtime chat stays on localhost and local model files.

| Path | Runtime scope |
|------|---------------|
| Model snapshots | Hugging Face cache on disk |
| Inference server | `mlx_vlm.server` on `127.0.0.1` |
| Chat client | Python stdlib HTTP and SSE client |
| Session history | In memory for the current terminal session |
| Session restore | `~/.cache/threadstone/` per model and terminal tab |
| Telemetry | Disabled through environment flags |

Offline guard:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
HF_HUB_DISABLE_IMPLICIT_TOKEN=1
HF_TOKEN=
DO_NOT_TRACK=1
DISABLE_TELEMETRY=1
ANONYMIZED_TELEMETRY=0
```

## Commands

Inside chat:

| Command | Effect |
|---------|--------|
| `/read <path>` | Attach a file or directory listing to the next message |
| `/drop` | Cancel the pending attachment |
| `/history` | Show recent visible conversation turns |
| `/restore` | Restore the saved session for this terminal tab |
| `/clear` | Reset conversation state and pending attachment |
| `/help` | Show in-session commands |
| `exit` / `quit` | Stop the chat and terminate the owned server |

Shell helpers from `setup.sh`:

| Command | Effect |
|---------|--------|
| `oracle [prompt] [size]` | Start chat with an owned server |
| `forge [size]` | Start a manual background server |
| `quench [size|all]` | Stop manual servers on default ports |
| `threadstone --doctor` | Check platform, venv, model snapshots, offline flags, and port state |
| `threadstone --list-models` | Print configured models, ports, context limits, and paths |

## Models

Default models are 4-bit MLX snapshots from `mlx-community`.

| Key | Model | Thinking | Max tokens | Context trim |
|-----|-------|----------|------------|--------------|
| `9B` | Qwen3.5-9B-MLX-4bit | Yes | 4096 | 8000 |
| `4B` | Qwen3.5-4B-MLX-4bit | Yes | 4096 | 8000 |
| `2B` | Qwen3.5-2B-MLX-4bit | No | 2048 | 4000 |
| `0.8B` | Qwen3.5-0.8B-MLX-4bit | No | 1024 | 2000 |

Each model has its own default port, memory estimate, context threshold, and response budget in `config.py`.

## Behavior

Threadstone owns the server lifecycle for normal chat.

- **Port recovery**: if the default port is busy, Threadstone scans forward and starts the model on the next available localhost port.
- **RAM guard**: startup checks free and reclaimable memory before launching a model, including other reachable model servers.
- **Crash recovery**: if the server disappears during a turn, Threadstone restarts it and resends the pending message.
- **Interrupt safety**: Ctrl-C mid-answer keeps the partial output; Ctrl-C while a request is in flight cancels the turn. The chat survives both.
- **Thinking models**: reasoning streams dimmed until `</think>`, then the final answer prints normally and only the final answer is sent back in later history.
- **Attachments**: `/read` accepts text files and directory listings, rejects non-regular files, rejects binary-looking content, and caps file payloads at 50 KB.
- **Context trimming**: old turns are trimmed when the approximate context crosses the configured threshold while preserving valid role alternation.

## Architecture

```text
oracle / threadstone
    |
    v
threadstone.py
    |-- argument parsing, doctor, REPL, streaming, history
    |-- ServerManager starts and monitors mlx_vlm.server
    |-- /read attaches bounded local text context
    |
    v
config.py
    |-- model registry, ports, limits, memory estimates
    |-- snapshot resolution from the Hugging Face cache
    |-- offline environment guard
```

Files:

| File | Role |
|------|------|
| `threadstone.py` | CLI, server lifecycle, chat loop, streaming parser, attachment handling |
| `config.py` | Model registry, offline env, runtime limits, validation |
| `setup.sh` | One-time macOS installer and shell helper setup |
| `tests/` | Unit coverage for parsing, streaming, history, config, attachments, and snapshots |

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python3 -m unittest discover -s tests -t . -v
python3 -m build --sdist --wheel
python3 -m twine check dist/*
```

Run the CLI without starting a model:

```bash
threadstone --version
threadstone --list-models
```

Run the local setup check after `./setup.sh`:

```bash
threadstone --doctor --all-models
```

## Package Release

Threadstone is a Python package, so the package registry is **PyPI**, not pub.dev. pub.dev is for Dart and Flutter packages.

Release path:

1. Update `CHANGELOG.md` and `pyproject.toml`.
2. Run tests, build, and `twine check`.
3. Create a GitHub release tag such as `v1.1.0`.
4. The release workflow builds the sdist and wheel, then publishes to PyPI through Trusted Publishing.

PyPI Trusted Publishing must be configured for:

| Field | Value |
|-------|-------|
| Owner | `gabrimatic` |
| Repository | `threadstone` |
| Workflow | `release.yml` |
| Environment | `pypi` |

## Security

Runtime chat is local by design, but model files and dependencies still come from external package and model hosts during setup.

Report vulnerabilities through [GitHub private vulnerability reporting](https://github.com/gabrimatic/threadstone/security/advisories/new). Do not open a public issue for security reports.
