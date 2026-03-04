#!/usr/bin/env python3
"""threadstone chat runtime."""

from __future__ import annotations

import atexit
import errno
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MAX_FILE_BYTES, MODELS, OFFLINE_ENV, REQ_TIMEOUT, RESTART_WAIT, SERVER_WAIT, TEMPERATURE, VENV

DIM = "\033[2m"
CYAN = "\033[36m"
RED = "\033[31m"
YEL = "\033[33m"
BOLD = "\033[1m"
RST = "\033[0m"
THINK = "\033[2;3m"

THINK_CLOSE = "</think>"
THINK_OPEN = "<think>"
DIR_LIMIT = 200
_MIN_PRINTABLE_RATIO = 0.70
NETWORK_ERRORS = (
    urllib.error.URLError,
    ConnectionError,
    BrokenPipeError,
    ConnectionResetError,
    TimeoutError,
    socket.timeout,
)
_CJK_RANGES = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x3000, 0x303F),
]


@dataclass(frozen=True)
class ModelConfig:
    size: str
    path: Path
    port: int
    thinking: bool
    max_tokens: int
    ctx_warn: int
    ctx_trim: int
    ctx_keep: int
    ram_gb: float


@dataclass(frozen=True)
class CliConfig:
    sys_prompt: str
    model: ModelConfig
    port_override: int | None
    extra_args: list[str]


@dataclass
class StreamResult:
    raw_text: str
    history_text: str
    api_text: str
    had_think_close: bool
    interrupted: bool = False


class StreamInterrupted(KeyboardInterrupt):
    def __init__(self, result: StreamResult) -> None:
        super().__init__("stream interrupted")
        self.result = result


def _vm_stat_available_gb() -> float | None:
    """Return approximate free + reclaimable RAM in GB via vm_stat."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True, timeout=3)
    except Exception:
        return None
    m = re.search(r"page size of (\d+) bytes", out)
    if not m:
        return None
    page_size = int(m.group(1))
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{label}:\s+(\d+)\.", out)
        if m:
            pages += int(m.group(1))
    if pages == 0:
        return None
    return pages * page_size / (1024 ** 3)


def _running_model_ram_gb(exclude_port: int) -> float:
    """Sum the approximate RAM of model servers that are currently reachable."""
    total = 0.0
    for model_data in MODELS.values():
        port = int(model_data["port"])
        if port == exclude_port:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    total += float(model_data["ram_gb"])
            except OSError:
                pass
    return total


class ServerManager:
    def __init__(self, cli: CliConfig) -> None:
        self.cli = cli
        self.port: int | None = None
        self.server = ""
        self.proc: subprocess.Popen | None = None
        self.log_path = Path(tempfile.gettempdir()) / f"threadstone-server-{os.getpid()}.log"
        self.log_fh = None
        self.printed_waiting = False

    def start(self) -> bool:
        available_gb = _vm_stat_available_gb()
        if available_gb is not None:
            model_ram_gb = self.cli.model.ram_gb
            running_gb = _running_model_ram_gb(self.cli.model.port)
            if running_gb > 0:
                print(f"{DIM}other model servers: ~{running_gb:.1f} GB in use{RST}")
            # Threshold is 1.5× model size: ram_gb already includes MLX overhead,
            # so 1.5× leaves headroom for the OS and any other active processes.
            if available_gb < model_ram_gb * 1.5:
                print(
                    f"{RED}insufficient RAM: ~{available_gb:.1f} GB available, "
                    f"~{model_ram_gb:.1f} GB needed for {self.cli.model.size}{RST}"
                )
                return False

        preferred = self.cli.port_override or self.cli.model.port
        try:
            candidate = self._find_available_port(preferred)
        except RuntimeError as exc:
            print(f"{RED}{exc}{RST}")
            return False

        for _ in range(10):
            self.port = candidate
            self.server = f"http://127.0.0.1:{candidate}"
            if self.port != preferred:
                print(f"{DIM}port {preferred} busy, using {candidate}{RST}")
            self._spawn()
            if self.wait_for_health(SERVER_WAIT):
                return True
            if self._log_contains("Address already in use"):
                self.stop()
                preferred = candidate + 1
                try:
                    candidate = self._find_available_port(preferred)
                except RuntimeError as exc:
                    print(f"{RED}{exc}{RST}")
                    return False
                continue
            return False
        return False

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self.proc = None
        if self.log_fh is not None:
            try:
                self.log_fh.close()
            except OSError:
                pass
            self.log_fh = None

    def restart(self) -> bool:
        if self.port is None:
            return False
        print(f"{YEL}server unavailable, restarting...{RST}")
        self.stop()
        self._spawn()
        ok = self.wait_for_health(RESTART_WAIT)
        if not ok:
            self._print_log_tail()
        return ok

    def wait_for_health(self, secs: int) -> bool:
        self.printed_waiting = False
        for attempt in range(secs):
            if self.healthy():
                return True
            sys.stdout.write("waiting..." if attempt == 0 else ".")
            sys.stdout.flush()
            self.printed_waiting = True
            time.sleep(1)
        print(f"\n{RED}server failed to start on port {self.port}{RST}")
        self._print_log_tail()
        return False

    def healthy(self) -> bool:
        if not self.server:
            return False
        try:
            urllib.request.urlopen(f"{self.server}/health", timeout=2)
            return True
        except Exception:
            return False

    def chat(self, messages: list[dict[str, str]]) -> urllib.response.addinfourl:
        body = json.dumps(
            {
                "model": str(self.cli.model.path),
                "messages": messages,
                "max_tokens": self.cli.model.max_tokens,
                "temperature": TEMPERATURE,
                "stream": True,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.server}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=REQ_TIMEOUT)

    def _spawn(self) -> None:
        self.stop()
        self.log_fh = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [str(VENV / "bin/python3"), "-m", "mlx_vlm.server", "--port", str(self.port)],
            env={**os.environ, **OFFLINE_ENV},
            stdin=subprocess.DEVNULL,
            stdout=self.log_fh,
            stderr=subprocess.STDOUT,
        )

    def _print_log_tail(self) -> None:
        try:
            tail_lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        except OSError:
            tail_lines = []
        if tail_lines:
            print(f"{DIM}server log: {self.log_path}{RST}")
            for line in tail_lines:
                print(f"{DIM}{line}{RST}")

    def _log_contains(self, text: str) -> bool:
        try:
            return text in self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

    @staticmethod
    def _find_available_port(preferred: int) -> int:
        candidate = preferred
        while True:
            if candidate > 65535:
                raise RuntimeError("no available port found in valid range")
            if ServerManager._port_available(candidate):
                return candidate
            candidate = (preferred + 10) if candidate == preferred else (candidate + 1)

    @staticmethod
    def _port_available(port: int) -> bool:
        if not (1 <= port <= 65535):
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    return False
                raise
        return True


def model_config(size: str) -> ModelConfig:
    data = MODELS[size]
    return ModelConfig(
        size=size,
        path=data["path"],
        port=data["port"],
        thinking=data["thinking"],
        max_tokens=data["max_tokens"],
        ctx_warn=data["ctx_warn"],
        ctx_trim=data["ctx_trim"],
        ctx_keep=data["ctx_keep"],
        ram_gb=float(data["ram_gb"]),
    )


def parse_args(argv: list[str], env: dict[str, str] | None = None) -> CliConfig:
    env = env or os.environ
    first = argv[1].upper() if len(argv) > 1 else ""
    if first in MODELS:
        sys_prompt = ""
        size = first
    elif len(argv) > 1:
        sys_prompt = argv[1]
        size = (argv[2] if len(argv) > 2 else "9B").upper()
    else:
        sys_prompt = ""
        size = "9B"

    if size not in MODELS:
        print(f'{YEL}unknown size "{size}", using 9B{RST}')
        size = "9B"

    extra_args = argv[3:] if first not in MODELS else argv[2:]
    if extra_args:
        print(f"{YEL}warning: extra arguments ignored: {extra_args}{RST}")

    port_override = None
    raw_port = env.get("FORGE_PORT")
    if raw_port is not None and raw_port != "":
        try:
            port_override = int(raw_port)
        except ValueError:
            print(f"{RED}FORGE_PORT={raw_port!r} is not a valid integer{RST}")
            raise SystemExit(1)
        if not (1 <= port_override <= 65535):
            print(f"{RED}FORGE_PORT={raw_port!r} must be 1-65535{RST}")
            raise SystemExit(1)

    return CliConfig(
        sys_prompt=sys_prompt,
        model=model_config(size),
        port_override=port_override,
        extra_args=extra_args,
    )


def _is_cjk(ch: str) -> bool:
    codepoint = ord(ch)
    return any(lo <= codepoint <= hi for lo, hi in _CJK_RANGES)


def token_est(history: list[dict[str, str]]) -> int:
    total = 0
    for message in history:
        text = message["content"]
        cjk_chars = sum(1 for ch in text if _is_cjk(ch))
        non_cjk = "".join(ch if not _is_cjk(ch) else " " for ch in text)
        word_tokens = len(non_cjk.split()) * 4 // 3
        total += cjk_chars + word_tokens
    return total


def trim_history(history: list[dict[str, str]], keep: int) -> list[dict[str, str]]:
    system = [m for m in history if m["role"] == "system"]
    turns = [m for m in history if m["role"] != "system"]
    kept = turns[-keep:]
    # Ensure the kept slice starts with a user turn to preserve role alternation.
    while kept and kept[0]["role"] != "user":
        kept = kept[1:]
    trimmed = len(turns) - len(kept)
    if trimmed > 0:
        print(f"{YEL}trimmed {trimmed} old messages{RST}")
    return system + kept


def sanitize_assistant_for_api(message: dict[str, str]) -> dict[str, str]:
    content = message.get("api_content")
    if content is None:
        content = message["content"]
        if THINK_CLOSE in content:
            content = content.split(THINK_CLOSE, 1)[1].lstrip("\n")
        content = content.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    return {"role": message["role"], "content": content}


def build_messages(sys_prompt: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    for message in history:
        # Skip system messages in history; the sys_prompt above is authoritative.
        if message["role"] == "system":
            continue
        if message["role"] == "assistant":
            messages.append(sanitize_assistant_for_api(message))
        else:
            messages.append({"role": message["role"], "content": message["content"]})
    return messages


def _is_printable_text(text: str, sample: int = 4096) -> bool:
    """Return True if at least _MIN_PRINTABLE_RATIO of the sample is printable."""
    chunk = text[:sample]
    if not chunk:
        return False
    printable = sum(1 for ch in chunk if ch.isprintable() or ch in "\n\r\t")
    return printable / len(chunk) >= _MIN_PRINTABLE_RATIO


def decode_attachment_bytes(raw: bytes, path: Path) -> str:
    # Explicit BOM — trust it, but verify the result is actual text.
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        decoded = raw.decode("utf-16", errors="replace")
        if not _is_printable_text(decoded):
            raise ValueError(f"binary file: {path.name}")
        return decoded

    # Null bytes in the header suggest UTF-16 or raw binary.
    if b"\x00" in raw[:8192]:
        candidates = []
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                decoded = raw.decode(encoding)
            except UnicodeDecodeError:
                continue
            score = decoded.count("\x00")
            candidates.append((score, decoded))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            decoded = candidates[0][1].replace("\x00", "")
            if decoded.strip() and _is_printable_text(decoded):
                return decoded
        raise ValueError(f"binary file: {path.name}")

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    return raw.decode("utf-8", errors="replace")


def read_path(arg: str, max_file_bytes: int) -> tuple[str, str, bool]:
    arg = arg.strip()
    if not arg:
        raise ValueError("usage: /read <path>")
    path = Path(arg).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"not found: {path}")
    if path.is_dir():
        entries: list[Path] = []
        total = 0
        try:
            for entry in path.iterdir():
                total += 1
                if len(entries) < DIR_LIMIT:
                    entries.append(entry)
        except PermissionError as exc:
            raise ValueError(f"permission denied: {path}") from exc
        except OSError as exc:
            raise ValueError(f"cannot read directory: {exc}") from exc
        entries.sort()
        listing = "\n".join(f'{"d" if entry.is_dir() else "f"}  {entry.name}' for entry in entries)
        if total > DIR_LIMIT:
            listing += f"\n(showing {DIR_LIMIT} of {total})"
        return str(path), listing, False
    if not path.is_file():
        raise ValueError(f"not a regular file: {path.name}")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(max_file_bytes + 1)
    except PermissionError as exc:
        raise ValueError(f"permission denied: {path.name}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read file: {exc}") from exc
    truncated = len(raw) > max_file_bytes
    raw = raw[:max_file_bytes]
    text = decode_attachment_bytes(raw, path)
    return str(path), text, truncated


def print_help() -> None:
    print(f"{DIM}/read <path>  attach one file or directory for the next prompt")
    print("/drop         clear the pending attachment")
    print("/clear        clear history and pending attachment")
    print("/history      show conversation history")
    print("/help         show this message")
    print(f"exit / quit   end session{RST}")


def display_user(content: str) -> str:
    marker = "\n\n<file "
    idx = content.find(marker)
    return content[:idx] if idx != -1 else content


def display_assistant(content: str) -> str:
    marker = THINK_CLOSE
    idx = content.find(marker)
    if idx != -1:
        return content[idx + len(marker) :].lstrip("\n")
    return content


def _history_content(raw_text: str, thinking: bool, had_think_close: bool) -> tuple[str, str]:
    cleaned = raw_text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    if thinking and had_think_close:
        answer = raw_text.split(THINK_CLOSE, 1)[1].replace(THINK_CLOSE, "").lstrip("\n")
        return answer.strip(), answer.strip()
    if thinking:
        return cleaned.strip(), ""
    return cleaned.strip(), cleaned.strip()


def stream_response(resp, thinking: bool) -> StreamResult:
    raw_text = ""
    had_think_close = False
    printed_count = 0
    if thinking:
        sys.stdout.write(THINK)
    sys.stdout.write("\n")
    sys.stdout.flush()

    try:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                token = json.loads(chunk)["choices"][0]["delta"].get("content", "")
            except (ValueError, KeyError, IndexError):
                continue

            raw_text += token

            if thinking and not had_think_close and THINK_CLOSE in raw_text:
                had_think_close = True
                close_at = raw_text.index(THINK_CLOSE)
                before = raw_text[printed_count:close_at]
                after = raw_text[close_at + len(THINK_CLOSE) :].replace(THINK_CLOSE, "")
                if before:
                    sys.stdout.write(before)
                sys.stdout.write(f"{RST}\n{BOLD}{'-' * 60}{RST}\n")
                if after:
                    sys.stdout.write(after.lstrip("\n"))
                printed_count = len(raw_text)
                sys.stdout.flush()
                continue

            new_text = raw_text[printed_count:]
            if had_think_close:
                new_text = new_text.replace(THINK_CLOSE, "")
            if new_text:
                sys.stdout.write(new_text)
                printed_count = len(raw_text)
            sys.stdout.flush()
    except KeyboardInterrupt as exc:
        sys.stdout.write(RST + "\n")
        sys.stdout.flush()
        history_text, api_text = _history_content(raw_text, thinking, had_think_close)
        raise StreamInterrupted(
            StreamResult(
                raw_text=raw_text,
                history_text=history_text,
                api_text=api_text,
                had_think_close=had_think_close,
                interrupted=True,
            )
        ) from exc

    sys.stdout.write(RST + "\n")
    sys.stdout.flush()

    if thinking and not had_think_close:
        print(f"{YEL}(no answer produced){RST}")
        return StreamResult(
            raw_text=raw_text,
            history_text="(no answer produced)",
            api_text="",
            had_think_close=False,
        )

    history_text, api_text = _history_content(raw_text, thinking, had_think_close)
    return StreamResult(
        raw_text=raw_text,
        history_text=history_text,
        api_text=api_text,
        had_think_close=had_think_close,
    )


def attach_to_user_prompt(user: str, attachment: tuple[str, str] | None) -> str:
    if not attachment:
        return user
    label, body = attachment
    return f'{user}\n\n<file name="{label}">\n{body}\n</file>'


def prompt_display(sys_prompt: str) -> str:
    if not sys_prompt:
        return "no system prompt"
    return f"{sys_prompt[:60]}..." if len(sys_prompt) > 60 else sys_prompt


def append_assistant_message(history: list[dict[str, str]], result: StreamResult) -> None:
    content = result.history_text.strip()
    if not content:
        content = "(no answer produced)"
    history.append({"role": "assistant", "content": content, "api_content": result.api_text.strip()})


def run_chat(cli: CliConfig) -> int:
    if not cli.model.path.exists():
        print(f"{RED}model not on disk: {cli.model.path}{RST}")
        return 1

    server = ServerManager(cli)
    if not server.start():
        return 1
    atexit.register(server.stop)

    history: list[dict[str, str]] = []
    attachment: tuple[str, str] | None = None

    clear_wait = f"\r{' ' * 50}\r" if server.printed_waiting else ""
    print(f"{clear_wait}{BOLD}threadstone {cli.model.size}{RST}  {DIM}{prompt_display(cli.sys_prompt)}{RST}\n")

    while True:
        if token_est(history) > cli.model.ctx_trim:
            history = trim_history(history, cli.model.ctx_keep)

        ctx = token_est(history)
        flag = f" {YEL}(context high){RST}" if ctx > cli.model.ctx_warn else ""
        prompt = f"{CYAN}{BOLD}›{RST}"
        if attachment:
            prompt += " 📎"
        prompt += " "
        print(
            f"{DIM}~{ctx} tokens · {cli.model.size} · /clear /read /drop /history /help · exit/quit{RST}{flag}"
        )

        try:
            user = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        if user.startswith("/"):
            parts = user.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/clear":
                history = []
                attachment = None
                print(f"{DIM}cleared{RST}")

            elif cmd == "/history":
                shown = False
                for message in history:
                    if message["role"] == "system":
                        continue
                    if message["role"] == "user":
                        preview = display_user(message["content"])[:120].replace("\n", " ")
                        print(f"  you  {preview}")
                    else:
                        preview = display_assistant(message["content"])[:120].replace("\n", " ")
                        print(f"   *   {preview}")
                    shown = True
                if not shown:
                    print(f"{DIM}(empty){RST}")

            elif cmd == "/help":
                print_help()

            elif cmd in {"/drop", "/unread"}:
                if attachment:
                    print(f"{DIM}dropped: {attachment[0]}{RST}")
                    attachment = None
                else:
                    print(f"{DIM}no attachment pending{RST}")

            elif cmd == "/read":
                try:
                    label, body, truncated = read_path(arg, MAX_FILE_BYTES)
                    if attachment is not None:
                        print(f"{YEL}replaced previous attachment{RST}")
                    attachment = (label, body)
                    if truncated:
                        print(f"{YEL}truncated attachment at {MAX_FILE_BYTES // 1024} KB{RST}")
                    print(f"{DIM}attached: {label}{RST}")
                except ValueError as exc:
                    print(f"{RED}error: {exc}{RST}")

            else:
                print(f"{RED}unknown command: {cmd}{RST}")

            continue

        content = attach_to_user_prompt(user, attachment)
        attachment = None
        history.append({"role": "user", "content": content})
        messages = build_messages(cli.sys_prompt, history)

        try:
            response = server.chat(messages)
            try:
                result = stream_response(response, cli.model.thinking)
            finally:
                response.close()
            if result.history_text.strip():
                append_assistant_message(history, result)
            else:
                history.append({"role": "assistant", "content": "(no answer produced)", "api_content": ""})
            print()
        except urllib.error.HTTPError as exc:
            history.pop()
            print(f"{RED}server {exc.code}: {exc.reason}{RST}")
        except NETWORK_ERRORS as exc:
            pending_user = history.pop()
            if server.restart():
                print(f"{DIM}reconnected, resending...{RST}")
                history.append(pending_user)
                messages = build_messages(cli.sys_prompt, history)
                try:
                    response = server.chat(messages)
                    try:
                        result = stream_response(response, cli.model.thinking)
                    finally:
                        response.close()
                    append_assistant_message(history, result)
                    print()
                except StreamInterrupted as exc:
                    append_assistant_message(history, exc.result)
                    print(f"{DIM}interrupted{RST}")
                except urllib.error.HTTPError as retry_exc:
                    history.pop()
                    print(f"{RED}server {retry_exc.code}: {retry_exc.reason}{RST}")
                except NETWORK_ERRORS as retry_exc:
                    history.pop()
                    print(f"{RED}retry failed: {retry_exc}{RST}")
            else:
                print(f"{RED}server restart failed: {exc}{RST}")
        except StreamInterrupted as exc:
            append_assistant_message(history, exc.result)
            print(f"{DIM}interrupted{RST}")

    return 0


def main(argv: list[str] | None = None) -> int:
    cli = parse_args(argv or sys.argv)
    return run_chat(cli)


if __name__ == "__main__":
    raise SystemExit(main())
