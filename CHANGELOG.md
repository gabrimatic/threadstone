# Changelog

This changelog tracks notable Threadstone changes.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - 2026-07-06

### Fixed

- A fresh `pip install threadstone` crashed every command — including `--version` and `--doctor` — with a config traceback when `~/mlx-env` did not exist, because the venv check ran at import time. The CLI now runs without the venv; chat and doctor report the missing venv with a `./setup.sh` hint instead. CI no longer creates a dummy venv, so this path stays tested.
- The inference server is now spawned with `--host 127.0.0.1`. `mlx_vlm.server` defaults to `0.0.0.0`, which exposed the local model server to the network; the shell `forge` helper pins loopback too.
- Ctrl-C while a request was in flight (model loading, server restarting, or resend) crashed the session with a raw traceback. It now cancels the turn and keeps the chat alive.
- A server crash with a clean TCP teardown mid-answer was silently recorded as a complete reply. A stream that ends without `[DONE]` now triggers the same restart-and-resend recovery as a dropped connection.
- A server that failed its startup health check was left running in the background, holding the model in memory. Failed startups now stop the spawned process, and cleanup is registered before startup so interrupts cannot leak it either.
- Thinking-model dim+italic styling leaked onto subsequent terminal output when a stream died mid-reasoning; styling is now reset on every error path.
- A failed session save (disk full, permissions) crashed the REPL after the answer had already streamed; it now warns and continues.
- Multi-line pastes were split into one model turn per line; pasted lines are now joined into a single message.
- The RAM guard ignored another instance of the same model running on its default port; it now excludes only the port it is about to bind.
- Ignore null stream content deltas (previously merged as #4, listed here for the release record).
- `setup.sh` wrote shell helpers to `~/.bashrc` for bash users, which macOS login shells never read; it now targets `~/.bash_profile`.

### Changed

- Startup and restart health windows widened from 20s/25s to 45s. Checks poll once per second and return as soon as the server is healthy, so only genuinely broken startups wait longer; cold starts on a loaded machine no longer fail spuriously.
- Session files older than the 24-hour restore window are pruned from `~/.cache/threadstone/` at chat startup.
- `--doctor` prints a repair hint when any check fails.

## [1.1.0] - 2026-05-13

### Added

- Added a real installable CLI surface through `pyproject.toml`: `threadstone` and `oracle` console scripts now point to the runtime.
- Added `threadstone --list-models` for a quick view of configured models, ports, context limits, and snapshot paths.
- Added `threadstone --doctor` for local setup checks covering macOS, Apple Silicon, Python version, `~/mlx-env`, `mlx_vlm.server`, offline flags, model snapshots, and port state.
- Added persisted `/restore` support for recent terminal sessions. Threadstone now saves conversation history under `~/.cache/threadstone/` and restores sessions up to 24 hours old.
- Added PyPI-ready build metadata, optional runtime/dev extras, project URLs, and package scripts.
- Added GitHub release automation for PyPI Trusted Publishing.
- Added Dependabot coverage for GitHub Actions and Python package metadata.
- Added security and contribution docs.

### Changed

- Rewrote the README in the same local-first style as the current GitHub-facing projects: concrete runtime boundaries, direct setup, command tables, architecture, development, and release notes.
- Updated CI to check Python 3.13 and 3.14, CLI smoke commands, build artifacts, and package metadata.
- Updated `setup.sh` so a full setup also installs Threadstone as an editable package inside `~/mlx-env`.

### Fixed

- Closed the old docs/runtime gap where `/restore` was documented but not implemented.
- Replaced stale GitHub issue/discussion references with the currently enabled project surfaces.

## [1.0.0] - 2026-03-04

### Added

- Offline terminal chat for local MLX language models on Apple Silicon.
- Per-model local server lifecycle management for Qwen3.5 MLX snapshots.
- Streaming terminal output with special handling for thinking models.
- `/read`, `/drop`, `/history`, `/clear`, and `/help` commands.
- Attachment safety checks for large files, directories, non-regular files, and binary-looking content.
- RAM guard, port conflict handling, server restart, and retry after connection loss.
- Unit coverage for argument parsing, config validation, snapshot resolution, streaming, attachments, and history trimming.
