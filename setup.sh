#!/usr/bin/env bash
# setup.sh — full install on a fresh macOS (Apple Silicon)
# Run once. After this: oracle [size] from any terminal.
set -euo pipefail

ORACLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HOME}/mlx-env"

RED='\033[31m'; GRN='\033[32m'; YEL='\033[33m'; DIM='\033[2m'; BOLD='\033[1m'; RST='\033[0m'
info()  { echo -e "${DIM}▸ $*${RST}"; }
ok()    { echo -e "${GRN}✔ $*${RST}"; }
warn()  { echo -e "${YEL}⚠ $*${RST}"; }
die()   { echo -e "${RED}✗ $*${RST}"; exit 1; }

# ── cleanup trap — warn about partial state on unexpected exit ─────────────────
_cleanup() {
  local code=$?
  [[ $code -eq 0 ]] && return
  echo ""
  warn "setup exited with error (code $code)"
  warn "the venv or model downloads may be in a partial state"
  warn "re-run setup.sh to resume, or remove ${VENV} and start fresh"
}
trap _cleanup EXIT

# ── 0. idempotency notice ──────────────────────────────────────────────────────
_already_installed=false
if [[ -d "${VENV}" && -x "${VENV}/bin/python3" ]]; then
  _already_installed=true
fi
# Check for oracle block in the detected rc file (determined below)
# (checked again after rc file detection)

# ── 1. macOS + Apple Silicon ──────────────────────────────────────────────────
[[ "$(uname)" == "Darwin" ]] || die "macOS only"
[[ "$(uname -m)" == "arm64" ]] || die "Apple Silicon (arm64) required for MLX"
ok "platform: macOS arm64"

# ── 1b. detect shell rc file ──────────────────────────────────────────────────
# Prefer the user's actual login shell; fall back to zsh.
_shell_rc() {
  local sh
  sh="$(dscl . -read "/Users/$USER" UserShell 2>/dev/null | awk '{print $2}')"
  case "${sh##*/}" in
    zsh)  echo "${HOME}/.zshrc" ;;
    bash) echo "${HOME}/.bashrc" ;;
    *)
      warn "shell '${sh##*/}' is not zsh or bash; writing aliases to ~/.zshrc anyway"
      echo "${HOME}/.zshrc"
      ;;
  esac
}
ZSHRC="$(_shell_rc)"

# Now check idempotency for the rc file too
if $_already_installed && grep -q "# ── oracle" "${ZSHRC}" 2>/dev/null; then
  warn "setup has already been run (venv exists, oracle block found in ${ZSHRC})"
  warn "continuing anyway; existing config will be updated"
fi

# ── 2. Homebrew ───────────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  info "installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
ok "homebrew: $(brew --version | head -1)"

# ── 3. Python 3.13 ───────────────────────────────────────────────────────────
if ! brew list python@3.13 &>/dev/null; then
  info "installing python@3.13..."
  brew install python@3.13
fi
PY=$(brew --prefix python@3.13)/bin/python3.13
ok "python: $("${PY}" --version)"

# ── 4. venv ───────────────────────────────────────────────────────────────────
if [[ -d "$VENV" ]]; then
  if [[ ! -x "$VENV/bin/python3" ]]; then
    warn "venv directory exists at ${VENV} but python3 binary is missing or broken"
    warn "recreating venv..."
    rm -rf "${VENV}"
    "${PY}" -m venv "${VENV}"
  else
    info "venv already exists at ${VENV}..."
  fi
else
  info "creating venv at ${VENV}..."
  "${PY}" -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
ok "venv: ${VENV}"

# ── 5. pip config: no update checks ──────────────────────────────────────────
# Only write the setting if it's not already present; don't clobber existing config.
_pip_conf_set() {
  local conf="$1"
  local dir; dir="$(dirname "$conf")"
  mkdir -p "$dir"
  if [[ ! -f "$conf" ]]; then
    printf '[global]\ndisable-pip-version-check = true\n' > "$conf"
  elif ! grep -q "disable-pip-version-check" "$conf"; then
    # File exists but setting is missing — append under [global] or add section.
    if grep -q '^\[global\]' "$conf"; then
      # Insert after [global] line
      python3 - "$conf" <<'PYEOF'
import sys, re
path = sys.argv[1]
text = open(path).read()
text = re.sub(r'(\[global\]\n)', r'\1disable-pip-version-check = true\n', text, count=1)
open(path, 'w').write(text)
PYEOF
    else
      printf '\n[global]\ndisable-pip-version-check = true\n' >> "$conf"
    fi
  fi
}
_pip_conf_set "${VENV}/pip.conf"
_pip_conf_set "${HOME}/.config/pip/pip.conf"

# ── 6. pip packages ───────────────────────────────────────────────────────────
info "installing mlx, mlx-lm, mlx-vlm..."
pip install --quiet --upgrade pip
pip install --quiet mlx mlx-lm "mlx-vlm[torch]"
ok "packages installed: $(pip show mlx-vlm | grep Version)"

# ── 7. download models (requires network — one time only) ─────────────────────
info "downloading models (this will take a while)..."

# Repo passed via env var to avoid shell injection. Stderr flows through for progress.
download_model() {
  local repo="$1"
  ORACLE_DL_REPO="$repo" python3 - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
repo = os.environ["ORACLE_DL_REPO"]
try:
    path = snapshot_download(repo, local_files_only=False)
    print(path)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

MODELS_LIST=(
  "mlx-community/Qwen3.5-9B-MLX-4bit"
  "mlx-community/Qwen3.5-4B-MLX-4bit"
  "mlx-community/Qwen3.5-2B-MLX-4bit"
  "mlx-community/Qwen3.5-0.8B-MLX-4bit"
)

for MODEL in "${MODELS_LIST[@]}"; do
  info "  $MODEL"
  if download_model "$MODEL" >/dev/null; then
    ok "  downloaded: $MODEL"
  else
    die "model download failed: ${MODEL}"
  fi
done

# ── 8. zshrc aliases ──────────────────────────────────────────────────────────
info "writing shell aliases..."

# Remove old oracle block. Line-anchored pattern avoids over-eating.
# Falls back to truncation if the closing marker is malformed.
python3 - "${ZSHRC}" <<'PYEOF'
import sys, re
from pathlib import Path

rc = Path(sys.argv[1])
text = rc.read_text(encoding="utf-8") if rc.exists() else ""

# Match from the oracle start marker to the closing dashes line, line by line.
# The pattern anchors on the exact marker strings so it cannot over-eat.
pattern = re.compile(
    r'\n# ── oracle [─]+\n'   # opening marker line
    r'(?:.*\n)*?'             # any lines in between (lazy, line by line)
    r'# [─]{10,}\n',          # closing dashes line
    re.MULTILINE
)
cleaned = pattern.sub('\n', text)

# If the marker still exists after cleaning, the regex failed to match.
if '# ── oracle' in cleaned:
    # Fallback: remove everything from the marker to end of file
    # (handles malformed blocks that are missing the closing line).
    idx = cleaned.find('\n# ── oracle')
    if idx != -1:
        cleaned = cleaned[:idx] + '\n'

rc.write_text(cleaned, encoding="utf-8")
PYEOF

# Only append if the oracle block is truly gone (avoids duplication).
if grep -q "# ── oracle" "${ZSHRC}" 2>/dev/null; then
  warn "oracle block already exists in ${ZSHRC} after cleanup attempt; skipping append to avoid duplication"
  warn "manually remove the oracle block from ${ZSHRC} and re-run if you want to update it"
else
  cat >> "${ZSHRC}" <<ZSHBLOCK

# ── oracle ───────────────────────────────────────────────────────────────────
_forge_port() {
  case "\${1:-9B}" in
    9B) echo 8080 ;;
    4B) echo 8081 ;;
    2B) echo 8082 ;;
    0.8B) echo 8083 ;;
    *) return 1 ;;
  esac
}

_offline_env() {
  echo \\
    HF_HUB_OFFLINE=1 \\
    TRANSFORMERS_OFFLINE=1 \\
    HF_DATASETS_OFFLINE=1 \\
    HF_HUB_DISABLE_TELEMETRY=1 \\
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \\
    HF_TOKEN= \\
    DO_NOT_TRACK=1 \\
    DISABLE_TELEMETRY=1 \\
    ANONYMIZED_TELEMETRY=0
}

# Only kill python/mlx_vlm processes; poll instead of blind sleep.
# Handles multiple PIDs (lsof can return several).
_stop_port() {
  local port="\$1" pids pname p killed=0
  pids=\$(lsof -ti tcp:"\$port" 2>/dev/null) || return 1
  [[ -z "\$pids" ]] && return 1
  for p in \$pids; do
    pname=\$(ps -o comm= -p "\$p" 2>/dev/null || true)
    case "\$pname" in
      *python*|*mlx*)
        kill "\$p" 2>/dev/null || true
        killed=1
        local i=0
        while kill -0 "\$p" 2>/dev/null && [[ \$i -lt 6 ]]; do
          sleep 0.5; i=\$((i + 1))
        done
        ;;
      *)
        echo "port \$port pid \$p is '\$pname' (not python/mlx); skipping" >&2
        ;;
    esac
  done
  [[ \$killed -eq 1 ]]
}

# forge [size] - start an inference server on the default port for that size
forge() {
  local size="\${1:-9B}" port
  port=\$(_forge_port "\$size") || { echo "unknown size: \$size" >&2; return 1; }
  source "${VENV}/bin/activate"
  _stop_port "\$port" && echo "stopped previous \$size server"
  env \$(_offline_env) \\
    python3 -m mlx_vlm.server --port \$port </dev/null >/dev/null 2>&1 &
  disown
  echo "\$size on :\$port"
}

# quench [size|all] - stop manual forge server(s) on default ports
quench() {
  local target="\${1:-9B}" ports
  if [[ "\${target}" == "all" ]]; then
    ports=(8080 8081 8082 8083)
  else
    ports=("\$(_forge_port "\${target}")") || { echo "unknown size: \${target}" >&2; return 1; }
  fi
  for p in "\${ports[@]}"; do _stop_port "\$p"; done
  echo "stopped"
}

# oracle [system_prompt] [size]
# threadstone.py owns argument parsing and per-instance server lifecycle.
oracle() {
  source "${VENV}/bin/activate"
  env \$(_offline_env) python3 "${ORACLE_DIR}/threadstone.py" "\$@"
}
# ─────────────────────────────────────────────────────────────────────────────
ZSHBLOCK

  ok "shell rc updated: ${ZSHRC}"
fi

# ── 9. verify offline ─────────────────────────────────────────────────────────
info "verifying offline mode (socket intercept test)..."
"${VENV}/bin/python3" - <<'PYEOF'
import socket
_orig = socket.getaddrinfo
def _guard(host, *a, **kw):
    if host not in ('localhost', '127.0.0.1', '::1', None, ''):
        raise RuntimeError(f"NETWORK CALL BLOCKED: {host}")
    return _orig(host, *a, **kw)
socket.getaddrinfo = _guard
import mlx_vlm, mlx_lm, huggingface_hub, transformers
print("✔ zero network calls at import")
PYEOF

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GRN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo -e "${GRN}  setup complete. open a new terminal and run:${RST}"
echo -e ""
echo -e "  ${BOLD}oracle${RST}           # 9B, no prompt"
echo -e "  ${BOLD}oracle 4B${RST}        # 4B model"
echo -e "  ${BOLD}oracle \"...\" 2B${RST}  # system prompt + 2B"
echo -e "${GRN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
