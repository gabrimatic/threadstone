# Changelog

This changelog tracks notable Threadstone changes.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
