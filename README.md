# Threadstone

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-required-blue.svg)]()
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)]()

**Offline terminal chat for local LLMs on Apple Silicon. No cloud, no telemetry, no network after setup.**

Runs [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) inference servers locally, streams responses with ANSI-styled output, manages its own server lifecycle. Stdlib only, zero third-party dependencies in the client.

---

## Quick Start

**Apple Silicon required.** Models are downloaded once during setup (~12 GB total).

```bash
git clone https://github.com/gabrimatic/threadstone.git
cd threadstone
./setup.sh
```

One command. Creates a Python venv, installs MLX packages, downloads four Qwen3.5 models, writes shell aliases.

```sh
oracle              # 9B model, no system prompt
oracle 4B           # choose model size
oracle "be terse"   # system prompt, default 9B
oracle "be terse" 2B
```

---

## Commands

| Command | Effect |
|---------|--------|
| `/read <path>` | Attach a file or directory to the next message |
| `/drop` | Cancel pending attachment |
| `/history` | Show conversation turns |
| `/restore` | Reload session from the current tab |
| `/clear` | Reset conversation and attachment |
| `/help` | List available commands |
| `exit` / `quit` | End session |

---

## Models

All 4-bit quantised, MLX format, from [mlx-community](https://huggingface.co/mlx-community) on HuggingFace. Downloaded once, local forever.

| Key | Model | Thinking | Defaults |
|-----|-------|----------|----------|
| `9B` | Qwen3.5-9B-MLX-4bit | yes | 4096 tokens, 8K context |
| `4B` | Qwen3.5-4B-MLX-4bit | yes | 4096 tokens, 8K context |
| `2B` | Qwen3.5-2B-MLX-4bit | no | 2048 tokens, 4K context |
| `0.8B` | Qwen3.5-0.8B-MLX-4bit | no | 1024 tokens, 2K context |

Each model has its own port, max tokens, and context thresholds configured in `config.py`.

### Adding a Model

1. Add an entry to the `MODELS` dict in `config.py`
2. Add the matching port to `_forge_port` in your shell rc file
3. Run `setup.sh` to download and configure

### Thinking Mode

Larger models (9B, 4B) emit a reasoning block before answering. The reasoning streams in dim italic, a separator line marks the transition, and the final answer prints at full brightness. Smaller models (2B, 0.8B) answer directly.

If a thinking model exhausts its token budget during reasoning without producing an answer, a `(no answer produced)` notice appears and ANSI codes reset cleanly.

---

## Multi-Instance

Each `oracle` session manages its own server. Open five terminals, run `oracle` in each. Port conflicts are detected automatically and resolved by assigning the next available port. No shared state, no interleaved output.

When a session exits, its server is stopped. No orphan processes.

---

## Offline Enforcement

Every process (server and client) runs with these environment variables:

```
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

`HF_HUB_OFFLINE=1` causes `huggingface_hub` to raise on any HTTP attempt. Verified with a socket-level intercept test during setup: zero external connections at import or runtime.

---

## Session Persistence

Conversation history is saved per model size and terminal tab. If the process exits unexpectedly, `/restore` reloads the session (up to 24 hours). History files live in `~/.cache/threadstone/`.

---

## Self-Healing

If the server crashes mid-session, the client detects the broken connection, restarts `mlx_vlm.server`, waits for health, and automatically resends the last message. Server logs are written to a temp file and printed on failure for diagnostics.

---

## Shell Aliases

Installed by `setup.sh` into your shell rc file (zsh or bash):

```sh
oracle [prompt] [size]   # start server + chat
forge  [size]            # start server only (background)
quench [size|all]        # stop server(s)
```

---

## Architecture

Two files, clear separation:

| File | Role |
|------|------|
| `config.py` | All tunables: model paths, ports, inference params, context limits, offline env vars |
| `threadstone.py` | The engine: server management, streaming, REPL, history. Stdlib only |
| `setup.sh` | One-time installer: venv, models, shell aliases |
| `test_threadstone.py` | Unit tests |

---

## Requirements

- macOS on Apple Silicon (MLX is Metal-only)
- Python 3.13+
- ~12 GB disk for all four models
