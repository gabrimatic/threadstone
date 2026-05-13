# Contributing

Contribute focused fixes: runtime reliability, model registry changes, setup hardening, CLI behavior, docs, tests, and package release work.

## Development Setup

```bash
git clone https://github.com/gabrimatic/threadstone.git
cd threadstone
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python3 -m unittest discover -s tests -t . -v
```

For a full local runtime install:

```bash
./setup.sh
threadstone --doctor --all-models
```

`setup.sh` installs the MLX runtime and downloads model snapshots. Use the lightweight development setup for normal code and docs changes.

## Architecture

Threadstone is intentionally small.

```text
threadstone.py
├── CLI parsing and diagnostics
├── ServerManager for mlx_vlm.server lifecycle
├── streaming SSE parser
├── REPL commands
├── attachment reader
├── history trimming
└── restore-file persistence

config.py
├── offline environment flags
├── model registry
├── Hugging Face snapshot lookup
└── runtime limit validation
```

The client uses the Python standard library. Runtime ML dependencies live in `~/mlx-env` after setup.

## Adding a Model

1. Add the model entry to `MODELS` in `config.py`.
2. Add its default port to `_forge_port()` in `setup.sh`.
3. Add the Hugging Face repo to `MODELS_LIST` in `setup.sh`.
4. Update the model table in `README.md`.
5. Add or update tests for config validation.

Keep model defaults explicit: port, thinking mode, `max_tokens`, context warn/trim thresholds, kept turns, and approximate RAM.

## Testing

```bash
python3 -m unittest discover -s tests -t . -v
python3 -m compileall config.py threadstone.py tests
bash -n setup.sh
python3 -m build --sdist --wheel
python3 -m twine check dist/*
```

Manual runtime checks after setup:

```bash
threadstone --version
threadstone --list-models
threadstone --doctor
oracle 0.8B
```

If your change touches server lifecycle, verify startup, a streamed response, `exit`, and no orphan server on the selected port.

If your change touches attachments, test `/read` with a text file, directory, oversized file, and binary file.

If your change touches restore behavior, test `/history`, process exit, new `oracle` session, `/restore`, and `/clear`.

## Pull Requests

- Keep one coherent change per PR.
- Preserve the local-first runtime boundary.
- Update `README.md` and `CHANGELOG.md` when behavior changes.
- Add tests for parsing, state, or failure paths when the runtime contract changes.
- Do not add hosted API fallbacks.
- Do not introduce telemetry.
- Do not include local paths, logs, model cache paths, or private machine details in public docs or comments.

## Vulnerability Reporting

See [SECURITY.md](SECURITY.md). Do not open public issues for security vulnerabilities.
