# oracle config — all settings live here
# Edit this file to change anything. threadstone.py is just the engine.
import os
from pathlib import Path

# ── offline guard — applied immediately so the main process is also covered ───
# Subprocesses inherit os.environ, so setting here covers everything.
# Do not remove these. They are what makes the app 100% offline.
OFFLINE_ENV = {
    "HF_HUB_OFFLINE":                "1",
    "TRANSFORMERS_OFFLINE":          "1",
    "HF_DATASETS_OFFLINE":           "1",
    "HF_HUB_DISABLE_TELEMETRY":      "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_TOKEN":                      "",
    "DO_NOT_TRACK":                  "1",
    "DISABLE_TELEMETRY":             "1",
    "ANONYMIZED_TELEMETRY":          "0",
}
os.environ.update(OFFLINE_ENV)

# ── venv — override with THREADSTONE_VENV env var ─────────────────────────────
VENV = Path(os.environ.get("THREADSTONE_VENV", Path.home() / "mlx-env"))

# ── HF cache — respects HF_HOME and XDG_CACHE_HOME before falling back ────────
def _hf_cache() -> Path:
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home) / "hub"
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg) / "huggingface/hub"
    return Path.home() / ".cache/huggingface/hub"

_HF = _hf_cache()

# ── snapshot resolution — finds the latest snapshot without hardcoding hashes ──
# Strategy:
#   1. Check refs/main for the pinned hash (most reliable).
#   2. Fall back to the single entry in snapshots/ if there's exactly one.
#   3. Fall back to the hardcoded path so existing installs never break.
def _snapshot(repo: str, fallback_hash: str) -> Path:
    repo_dir = _HF / repo
    refs_main = repo_dir / "refs" / "main"
    snapshots_dir = repo_dir / "snapshots"
    try:
        if refs_main.exists():
            h = refs_main.read_text().strip()
            p = snapshots_dir / h
            if p.is_dir():
                return p
        if snapshots_dir.is_dir():
            entries = [e for e in snapshots_dir.iterdir() if e.is_dir()]
            if len(entries) == 1:
                return entries[0]
    except OSError:
        pass
    return snapshots_dir / fallback_hash  # original hardcoded path as last resort

# ── models: per-size runtime config ───────────────────────────────────────────
# Add a new model here and one entry in _forge_port() in setup.sh.
# ram_gb: approximate peak unified memory usage (model weights + MLX overhead).
# Used for the pre-launch memory guard.
MODELS: dict[str, dict[str, object]] = {
    "9B": {
        "path": _snapshot("models--mlx-community--Qwen3.5-9B-MLX-4bit", "d0b3cb793b1b12acf826571ae1bb2bc819a7a37f"),
        "port": 8080,
        "thinking": True,
        "max_tokens": 4096,
        "ctx_warn": 6000,
        "ctx_trim": 8000,
        "ctx_keep": 12,
        "ram_gb": 7.5,
    },
    "4B": {
        "path": _snapshot("models--mlx-community--Qwen3.5-4B-MLX-4bit", "32f3e8ecf65426fc3306969496342d504bfa13f3"),
        "port": 8081,
        "thinking": True,
        "max_tokens": 4096,
        "ctx_warn": 6000,
        "ctx_trim": 8000,
        "ctx_keep": 12,
        "ram_gb": 3.5,
    },
    "2B": {
        "path": _snapshot("models--mlx-community--Qwen3.5-2B-MLX-4bit", "93760be4f1f69842a46bc13dbdc0f19e291392a3"),
        "port": 8082,
        "thinking": False,
        "max_tokens": 2048,
        "ctx_warn": 3000,
        "ctx_trim": 4000,
        "ctx_keep": 8,
        "ram_gb": 2.0,
    },
    "0.8B": {
        "path": _snapshot("models--mlx-community--Qwen3.5-0.8B-MLX-4bit", "5d894f8cc4ef3e6c88537bf3746ed262f549da6a"),
        "port": 8083,
        "thinking": False,
        "max_tokens": 1024,
        "ctx_warn": 1500,
        "ctx_trim": 2000,
        "ctx_keep": 4,
        "ram_gb": 1.5,
    },
}

TEMPERATURE = 0.7   # 0.0 = deterministic, 1.0 = creative
REQ_TIMEOUT = 300   # seconds — max wait for a response

# ── server ────────────────────────────────────────────────────────────────────
SERVER_WAIT  = 20   # seconds — wait for server to become healthy on startup
RESTART_WAIT = 25   # seconds — wait after auto-restart on crash

# ── /read attachment ──────────────────────────────────────────────────────────
MAX_FILE_BYTES = 51200  # 50 KB cap — larger files are truncated

# ── validation ────────────────────────────────────────────────────────────────
def validate():
    errors = []

    if not VENV.exists():
        errors.append(f"VENV does not exist: {VENV}")

    if REQ_TIMEOUT <= 0:
        errors.append(f"REQ_TIMEOUT must be positive, got {REQ_TIMEOUT}")
    if SERVER_WAIT <= 0:
        errors.append(f"SERVER_WAIT must be positive, got {SERVER_WAIT}")
    if RESTART_WAIT <= 0:
        errors.append(f"RESTART_WAIT must be positive, got {RESTART_WAIT}")
    if MAX_FILE_BYTES <= 0:
        errors.append(f"MAX_FILE_BYTES must be positive, got {MAX_FILE_BYTES}")

    required: dict[str, type | tuple[type, ...]] = {
        "path": Path,
        "port": int,
        "thinking": bool,
        "max_tokens": int,
        "ctx_warn": int,
        "ctx_trim": int,
        "ctx_keep": int,
        "ram_gb": (int, float),
    }

    for key, model in MODELS.items():
        for field, field_type in required.items():
            if field not in model:
                errors.append(f"MODELS[{key!r}] is missing {field!r}")
                continue
            if not isinstance(model[field], field_type):
                errors.append(f"MODELS[{key!r}][{field!r}] must be {field_type.__name__ if isinstance(field_type, type) else ' or '.join(t.__name__ for t in field_type)}")
        if isinstance(model.get("max_tokens"), int) and model["max_tokens"] <= 0:
            errors.append(f"MODELS[{key!r}]['max_tokens'] must be positive")
        if isinstance(model.get("ctx_keep"), int) and model["ctx_keep"] <= 0:
            errors.append(f"MODELS[{key!r}]['ctx_keep'] must be positive")
        if isinstance(model.get("ctx_warn"), int) and model["ctx_warn"] < 0:
            errors.append(f"MODELS[{key!r}]['ctx_warn'] must be non-negative")
        if isinstance(model.get("ctx_trim"), int) and isinstance(model.get("ctx_warn"), int):
            if model["ctx_trim"] <= model["ctx_warn"]:
                errors.append(f"MODELS[{key!r}]['ctx_trim'] must be greater than 'ctx_warn'")
        if isinstance(model.get("ram_gb"), (int, float)) and model["ram_gb"] <= 0:
            errors.append(f"MODELS[{key!r}]['ram_gb'] must be positive")

    if errors:
        raise ValueError("config errors:\n" + "\n".join(f"  · {e}" for e in errors))

validate()
