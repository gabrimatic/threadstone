import io
import json
import os
import stat
import subprocess
import tempfile
import time
import unittest
from copy import deepcopy
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import config as cfg
import threadstone as ts


def _sse(*tokens, done=True):
    lines = []
    for token in tokens:
        payload = json.dumps({"choices": [{"delta": {"content": token}}]})
        lines.append(f"data: {payload}\n".encode())
    if done:
        lines.append(b"data: [DONE]\n")
    return lines


class TestParseArgs(unittest.TestCase):
    def test_size_only(self):
        cli = ts.parse_args(["threadstone.py", "4B"])
        self.assertEqual(cli.sys_prompt, "")
        self.assertEqual(cli.model.size, "4B")

    def test_prompt_then_size(self):
        cli = ts.parse_args(["threadstone.py", "be terse", "2B"])
        self.assertEqual(cli.sys_prompt, "be terse")
        self.assertEqual(cli.model.size, "2B")

    def test_invalid_size_warns_and_falls_back(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cli = ts.parse_args(["threadstone.py", "", "INVALID"])
        self.assertEqual(cli.model.size, "9B")
        self.assertIn('unknown size "INVALID", using 9B', buf.getvalue())

    def test_invalid_forge_port_exits(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with self.assertRaises(SystemExit):
                ts.parse_args(["threadstone.py"], env={"FORGE_PORT": "abc"})
        self.assertIn("is not a valid integer", buf.getvalue())

    def test_forge_port_out_of_range_exits(self):
        for bad in ("-1", "0", "65536", "99999"):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                with self.assertRaises(SystemExit):
                    ts.parse_args(["threadstone.py"], env={"FORGE_PORT": bad})
            self.assertIn("must be 1-65535", buf.getvalue())

    def test_forge_port_boundary_values_accepted(self):
        cli = ts.parse_args(["threadstone.py"], env={"FORGE_PORT": "1"})
        self.assertEqual(cli.port_override, 1)
        cli = ts.parse_args(["threadstone.py"], env={"FORGE_PORT": "65535"})
        self.assertEqual(cli.port_override, 65535)

    def test_cli_port_overrides_environment(self):
        cli = ts.parse_args(["threadstone.py", "--port", "8123"], env={"FORGE_PORT": "8089"})
        self.assertEqual(cli.port_override, 8123)

    def test_list_models_command(self):
        cli = ts.parse_args(["threadstone.py", "--list-models"])
        self.assertEqual(cli.command, "list-models")
        self.assertEqual(cli.model.size, "9B")

    def test_doctor_command_can_check_all_models(self):
        cli = ts.parse_args(["threadstone.py", "--doctor", "--all-models", "2B"])
        self.assertEqual(cli.command, "doctor")
        self.assertTrue(cli.check_all_models)
        self.assertEqual(cli.model.size, "2B")

    def test_invalid_cli_port_exits(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                ts.parse_args(["threadstone.py", "--port", "65536"])


class TestTokenEst(unittest.TestCase):
    def test_english_text(self):
        hist = [{"role": "user", "content": "hello world"}]
        self.assertEqual(ts.token_est(hist), 2)

    def test_cjk_text_counts_per_character(self):
        hist = [{"role": "user", "content": "中文世界"}]
        self.assertEqual(ts.token_est(hist), 4)

    def test_mixed_text(self):
        hist = [{"role": "user", "content": "hello 世界"}]
        self.assertEqual(ts.token_est(hist), 3)


class TestTrimHistory(unittest.TestCase):
    def _turns(self, *roles):
        return [{"role": r, "content": r} for r in roles]

    def test_trim_preserves_system(self):
        history = [{"role": "system", "content": "sys"}] + self._turns("user", "assistant", "user")
        result = ts.trim_history(history, keep=2)
        self.assertEqual(result[0]["role"], "system")

    def test_odd_keep_drops_leading_assistant(self):
        # keep=3 from [u, a, u, a, u, a, u, a, u, a] would give [a, u, a]
        # the leading assistant must be dropped → [u, a]
        turns = self._turns("user", "assistant") * 5
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            result = ts.trim_history(turns, keep=3)
        self.assertEqual(result[0]["role"], "user", "slice must start with a user turn")

    def test_result_always_starts_with_user(self):
        for keep in range(1, 8):
            turns = self._turns("user", "assistant") * 6
            result = ts.trim_history(turns, keep=keep)
            non_system = [m for m in result if m["role"] != "system"]
            if non_system:
                self.assertEqual(non_system[0]["role"], "user", f"keep={keep} starts with assistant")


class TestDisplayHelpers(unittest.TestCase):
    def test_display_user_strips_file_block(self):
        content = 'question\n\n<file name="/tmp/a.txt">\nbody\n</file>'
        self.assertEqual(ts.display_user(content), "question")

    def test_display_assistant_strips_thinking_prefix(self):
        content = "reasoning\n</think>\nanswer"
        self.assertEqual(ts.display_assistant(content), "answer")

    def test_build_messages_strips_thinking_from_api_history(self):
        history = [{"role": "assistant", "content": "hidden\n</think>\nanswer"}]
        self.assertEqual(
            ts.build_messages("", history),
            [{"role": "assistant", "content": "answer"}],
        )

    def test_build_messages_respects_api_content_override(self):
        history = [{"role": "assistant", "content": "partial", "api_content": ""}]
        self.assertEqual(
            ts.build_messages("", history),
            [{"role": "assistant", "content": ""}],
        )

    def test_build_messages_skips_system_in_history(self):
        # Prevents double system prompt when history contains a system entry.
        history = [
            {"role": "system", "content": "old prompt"},
            {"role": "user", "content": "hi"},
        ]
        result = ts.build_messages("new prompt", history)
        system_msgs = [m for m in result if m["role"] == "system"]
        self.assertEqual(len(system_msgs), 1)
        self.assertEqual(system_msgs[0]["content"], "new prompt")

    def test_build_messages_no_system_in_history_no_prompt(self):
        history = [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "hi"},
        ]
        result = ts.build_messages("", history)
        self.assertFalse(any(m["role"] == "system" for m in result))


class TestReadPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_file(self, name, content):
        path = Path(self.tmp) / name
        path.write_bytes(content)
        return path

    def test_regular_utf8_file(self):
        path = self._make_file("hello.txt", b"hello\n")
        label, text, truncated = ts.read_path(str(path), 1024)
        self.assertEqual(label, str(path.resolve()))
        self.assertEqual(text, "hello\n")
        self.assertFalse(truncated)

    def test_directory_listing_is_capped(self):
        for index in range(ts.DIR_LIMIT + 5):
            (Path(self.tmp) / f"file_{index:03d}.txt").write_text("")
        _, listing, truncated = ts.read_path(self.tmp, 1024)
        self.assertIn("(showing 200 of 205)", listing)
        self.assertFalse(truncated)

    def test_non_regular_file_is_rejected(self):
        fifo = Path(self.tmp) / "pipe"
        os.mkfifo(str(fifo))
        with self.assertRaises(ValueError):
            ts.read_path(str(fifo), 1024)

    def test_utf16_without_bom_is_accepted(self):
        raw = "hello".encode("utf-16-le")
        path = self._make_file("utf16.txt", raw)
        _, text, _ = ts.read_path(str(path), 1024)
        self.assertIn("hello", text)

    def test_binary_with_utf16_pattern_is_rejected(self):
        # \x02\x00 pairs have null bytes (enter the UTF-16 path) and decode on
        # little-endian (Apple Silicon) to U+0002 (STX), a non-printable control
        # character.  The printable-ratio check must reject the result.
        raw = b"\x02\x00" * 512
        path = self._make_file("data.bin", raw)
        with self.assertRaises(ValueError):
            ts.read_path(str(path), 4096)

    @unittest.skipIf(os.getuid() == 0, "root bypasses permission checks")
    def test_permission_denied_file(self):
        path = self._make_file("secret.txt", b"secret")
        path.chmod(0o000)
        try:
            with self.assertRaises(ValueError):
                ts.read_path(str(path), 1024)
        finally:
            path.chmod(0o644)


class TestIsPrintableText(unittest.TestCase):
    def test_clean_text_passes(self):
        self.assertTrue(ts._is_printable_text("hello world\n"))

    def test_binary_garbage_fails(self):
        garbage = "".join(chr(i) for i in range(1, 32) if i not in (9, 10, 13)) * 100
        self.assertFalse(ts._is_printable_text(garbage))

    def test_empty_string_fails(self):
        self.assertFalse(ts._is_printable_text(""))


class TestStreamResponse(unittest.TestCase):
    def _capture(self, lines, thinking):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            result = ts.stream_response(lines, thinking)
        return result, buf.getvalue()

    def test_non_thinking_response(self):
        result, out = self._capture(_sse("hello", " world"), thinking=False)
        self.assertEqual(result.history_text, "hello world")
        self.assertEqual(result.api_text, "hello world")
        self.assertIn("hello world", out)

    def test_null_content_delta_is_ignored(self):
        try:
            result, out = self._capture(_sse(None, "OK"), thinking=False)
        except TypeError as exc:
            self.fail(f"null content delta should be ignored: {exc}")
        self.assertEqual(result.history_text, "OK")
        self.assertEqual(result.api_text, "OK")
        self.assertIn("OK", out)

    def test_thinking_response_strips_reasoning_from_history(self):
        result, out = self._capture(_sse("reason", "</thi", "nk>", "answer"), thinking=True)
        self.assertEqual(result.history_text, "answer")
        self.assertEqual(result.api_text, "answer")
        self.assertIn("-" * 60, out)

    def test_duplicate_think_tags_are_not_returned(self):
        result, _ = self._capture(_sse("thought", "</think>", "answer</think>"), thinking=True)
        self.assertEqual(result.history_text, "answer")
        self.assertNotIn("</think>", result.raw_text.replace("</think>", "", 999))

    def test_thinking_exhaustion_inserts_placeholder(self):
        result, out = self._capture(_sse("still thinking"), thinking=True)
        self.assertEqual(result.history_text, "(no answer produced)")
        self.assertEqual(result.api_text, "")
        self.assertIn("(no answer produced)", out)

    def test_interrupt_keeps_partial_output(self):
        class InterruptingResponse:
            def __iter__(self):
                yield _sse("partial", done=False)[0]
                raise KeyboardInterrupt

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with self.assertRaises(ts.StreamInterrupted) as ctx:
                ts.stream_response(InterruptingResponse(), thinking=False)
        self.assertEqual(ctx.exception.result.history_text, "partial")
        self.assertEqual(ctx.exception.result.api_text, "partial")

    def test_stream_ending_without_done_raises(self):
        # A clean TCP teardown after a server crash ends the iterator without
        # [DONE]; treating that as success would record a truncated answer.
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with self.assertRaises(ConnectionError):
                ts.stream_response(_sse("partial", done=False), thinking=False)

    def test_stream_ending_without_done_resets_ansi(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with self.assertRaises(ConnectionError):
                ts.stream_response(_sse("reasoning", done=False), thinking=True)
        self.assertIn(ts.RST, buf.getvalue())

    def test_mid_stream_error_resets_ansi_before_propagating(self):
        class CrashingResponse:
            def __iter__(self):
                yield _sse("reasoning", done=False)[0]
                raise ConnectionResetError("server died")

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with self.assertRaises(ConnectionResetError):
                ts.stream_response(CrashingResponse(), thinking=True)
        out = buf.getvalue()
        self.assertIn(ts.THINK, out)
        self.assertIn(ts.RST, out, "dim+italic must be reset when the stream dies")


class TestRuntimeChecks(unittest.TestCase):
    def test_mlx_server_import_check_allows_slow_cold_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / "bin" / "python3"
            python.parent.mkdir()
            python.write_text("")
            python.chmod(0o755)
            completed = subprocess.CompletedProcess(args=[str(python)], returncode=0)

            with patch.object(ts, "VENV", Path(tmp)):
                with patch("threadstone.subprocess.run", return_value=completed) as run:
                    self.assertTrue(ts._venv_has_mlx_server())

            self.assertGreaterEqual(run.call_args.kwargs["timeout"], 30)


class TestAppendAssistantMessage(unittest.TestCase):
    def test_partial_thinking_interrupt_is_not_sent_back_to_model(self):
        history = []
        ts.append_assistant_message(
            history,
            ts.StreamResult(
                raw_text="reasoning",
                history_text="reasoning",
                api_text="",
                had_think_close=False,
                interrupted=True,
            ),
        )
        self.assertEqual(history[0]["content"], "reasoning")
        self.assertEqual(history[0]["api_content"], "")


class TestServerManager(unittest.TestCase):
    def _cli(self):
        return ts.parse_args(["threadstone.py", "0.8B"])

    def test_spawn_binds_localhost(self):
        # mlx_vlm.server defaults to 0.0.0.0; the spawn must pin loopback.
        manager = ts.ServerManager(self._cli())
        manager.port = 8083
        with patch("threadstone.subprocess.Popen") as popen:
            manager._spawn()
        args = popen.call_args.args[0]
        self.assertIn("--host", args)
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        manager.proc = None
        manager.stop()

    def test_failed_health_wait_stops_spawned_server(self):
        # A server that never turns healthy must not outlive start().
        manager = ts.ServerManager(self._cli())
        with patch("threadstone._vm_stat_available_gb", return_value=None), \
             patch.object(ts.ServerManager, "_find_available_port", return_value=8083), \
             patch.object(manager, "_spawn"), \
             patch.object(manager, "wait_for_health", return_value=False), \
             patch.object(manager, "_log_contains", return_value=False), \
             patch.object(manager, "stop") as stop:
            self.assertFalse(manager.start())
        stop.assert_called()

    def test_insufficient_ram_refuses_without_spawning(self):
        manager = ts.ServerManager(self._cli())
        buf = io.StringIO()
        with patch("threadstone._vm_stat_available_gb", return_value=0.5), \
             patch("threadstone._running_model_ram_gb", return_value=0.0), \
             patch.object(ts.ServerManager, "_find_available_port", return_value=8083), \
             patch.object(manager, "_spawn") as spawn, \
             patch("sys.stdout", buf):
            self.assertFalse(manager.start())
        spawn.assert_not_called()
        self.assertIn("insufficient RAM", buf.getvalue())

    def test_ram_guard_counts_other_running_servers(self):
        # The guard must exclude only the port it is about to bind, so another
        # instance of the same model on the default port is counted.
        manager = ts.ServerManager(self._cli())
        buf = io.StringIO()
        with patch("threadstone._vm_stat_available_gb", return_value=64.0), \
             patch("threadstone._running_model_ram_gb", return_value=1.5) as running, \
             patch.object(ts.ServerManager, "_find_available_port", return_value=8093), \
             patch.object(manager, "_spawn"), \
             patch.object(manager, "wait_for_health", return_value=True), \
             patch("sys.stdout", buf):
            self.assertTrue(manager.start())
        running.assert_called_once_with(8093)
        self.assertIn("other model servers", buf.getvalue())


class _CancellingServer:
    def __init__(self, cli):
        self.cli = cli
        self.printed_waiting = False

    def start(self):
        return True

    def stop(self):
        pass

    def chat(self, messages):
        raise KeyboardInterrupt


class TestRunChatPreflight(unittest.TestCase):
    def test_missing_venv_reports_setup_hint(self):
        cli = ts.parse_args(["threadstone.py", "0.8B"])
        buf = io.StringIO()
        with patch.object(ts, "VENV", Path(tempfile.gettempdir()) / "threadstone-no-such-venv"), \
             patch("sys.stdout", buf):
            self.assertEqual(ts.run_chat(cli), 1)
        out = buf.getvalue()
        self.assertIn("venv not found", out)
        self.assertIn("setup.sh", out)


class TestRunChatCancel(unittest.TestCase):
    def test_ctrl_c_during_request_cancels_turn_and_keeps_repl(self):
        cli = ts.parse_args(["threadstone.py", "0.8B"])
        answers = iter(["hi"])

        def fake_input(prompt=""):
            try:
                return next(answers)
            except StopIteration:
                raise EOFError

        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ts, "ServerManager", _CancellingServer), \
                 patch.object(ts, "SESSION_DIR", Path(tmp)), \
                 patch.object(ts, "session_path", lambda size: Path(tmp) / "s.json"), \
                 patch("builtins.input", fake_input), \
                 patch("pathlib.Path.exists", return_value=True), \
                 patch("sys.stdout", buf):
                exit_code = ts.run_chat(cli)
        self.assertEqual(exit_code, 0)
        self.assertIn("cancelled", buf.getvalue())


class TestReadUserInput(unittest.TestCase):
    def test_multiline_paste_joins_buffered_lines(self):
        fake_stdin = unittest.mock.Mock()
        fake_stdin.isatty.return_value = True
        fake_stdin.readline.side_effect = ["line2\n", "line3\n"]
        selects = iter([([fake_stdin], [], []), ([fake_stdin], [], []), ([], [], [])])
        with patch("builtins.input", return_value="line1"), \
             patch.object(ts.sys, "stdin", fake_stdin), \
             patch.object(ts.select, "select", lambda *a: next(selects)):
            self.assertEqual(ts.read_user_input("> "), "line1\nline2\nline3")

    def test_piped_stdin_is_not_drained(self):
        fake_stdin = unittest.mock.Mock()
        fake_stdin.isatty.return_value = False
        with patch("builtins.input", return_value="only"), \
             patch.object(ts.sys, "stdin", fake_stdin):
            self.assertEqual(ts.read_user_input("> "), "only")
        fake_stdin.readline.assert_not_called()

    def test_single_line_tty_input_passes_through(self):
        fake_stdin = unittest.mock.Mock()
        fake_stdin.isatty.return_value = True
        with patch("builtins.input", return_value="hello"), \
             patch.object(ts.sys, "stdin", fake_stdin), \
             patch.object(ts.select, "select", lambda *a: ([], [], [])):
            self.assertEqual(ts.read_user_input("> "), "hello")


class TestPruneStaleSessions(unittest.TestCase):
    def test_prunes_only_expired_session_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            old_json = directory / "old.json"
            old_tmp = directory / "old.tmp"
            fresh = directory / "fresh.json"
            other = directory / "keep.txt"
            for path in (old_json, old_tmp, fresh, other):
                path.write_text("{}")
            past = time.time() - ts.RESTORE_MAX_AGE_SECONDS - 60
            for path in (old_json, old_tmp, other):
                os.utime(path, (past, past))
            ts.prune_stale_sessions(directory)
            self.assertFalse(old_json.exists())
            self.assertFalse(old_tmp.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(other.exists(), "non-session files must be left alone")

    def test_missing_directory_is_ignored(self):
        ts.prune_stale_sessions(Path(tempfile.gettempdir()) / "threadstone-prune-does-not-exist")


class TestSessionPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "session.json"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_failure_warns_instead_of_crashing(self):
        # Disk-full/permission errors after a successful answer must not kill
        # the REPL.
        buf = io.StringIO()
        with patch("sys.stdout", buf), \
             patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            ts.save_history(self.path, [])
        self.assertIn("session save failed", buf.getvalue())

    def test_save_and_load_history(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "api_content": "hello"},
        ]
        ts.save_history(self.path, history)
        self.assertEqual(ts.load_history(self.path), history)

    def test_old_history_is_rejected(self):
        payload = {"version": 1, "saved_at": time.time() - 10, "history": []}
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            ts.load_history(self.path, max_age_seconds=1)

    def test_invalid_history_is_rejected(self):
        payload = {"version": 1, "saved_at": time.time(), "history": [{"role": "tool", "content": "x"}]}
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            ts.load_history(self.path)


class TestConfigValidate(unittest.TestCase):
    def setUp(self):
        self.original_models = deepcopy(cfg.MODELS)

    def tearDown(self):
        cfg.MODELS.clear()
        cfg.MODELS.update(deepcopy(self.original_models))

    def test_baseline_passes(self):
        cfg.validate()

    def test_validate_does_not_require_venv(self):
        # validate() runs at import time; a missing venv must not break
        # --version/--help/--doctor on a machine that never ran setup.sh.
        with patch.object(cfg, "VENV", Path(tempfile.gettempdir()) / "threadstone-no-such-venv"):
            cfg.validate()

    def test_missing_model_field_raises(self):
        del cfg.MODELS["9B"]["port"]
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_missing_ram_gb_raises(self):
        del cfg.MODELS["9B"]["ram_gb"]
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_zero_ram_gb_raises(self):
        cfg.MODELS["9B"]["ram_gb"] = 0.0
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_trim_must_exceed_warn(self):
        cfg.MODELS["9B"]["ctx_trim"] = cfg.MODELS["9B"]["ctx_warn"]
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_max_tokens_must_be_positive(self):
        cfg.MODELS["9B"]["max_tokens"] = 0
        with self.assertRaises(ValueError):
            cfg.validate()


class TestSnapshot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.original_hf = cfg._HF

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        cfg._HF = self.original_hf

    def test_refs_main_resolution(self):
        repo = "models--test--model"
        repo_dir = Path(self.tmp) / repo
        snap_hash = "abc123"
        snap_dir = repo_dir / "snapshots" / snap_hash
        snap_dir.mkdir(parents=True)
        (repo_dir / "refs").mkdir(parents=True)
        (repo_dir / "refs" / "main").write_text(snap_hash)
        cfg._HF = Path(self.tmp)
        self.assertEqual(cfg._snapshot(repo, "fallback"), snap_dir)

    def test_single_snapshot_fallback(self):
        repo = "models--test--single"
        repo_dir = Path(self.tmp) / repo
        snap_dir = repo_dir / "snapshots" / "hash"
        snap_dir.mkdir(parents=True)
        cfg._HF = Path(self.tmp)
        self.assertEqual(cfg._snapshot(repo, "fallback"), snap_dir)

    def test_missing_repo_returns_fallback_path(self):
        cfg._HF = Path(self.tmp)
        result = cfg._snapshot("models--missing", "fallback")
        self.assertEqual(result, Path(self.tmp) / "models--missing" / "snapshots" / "fallback")


if __name__ == "__main__":
    unittest.main()
