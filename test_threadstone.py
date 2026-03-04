import io
import json
import os
import stat
import tempfile
import unittest
from copy import deepcopy
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

    @unittest.skipIf(os.getuid() == 0, "root bypasses permission checks")
    def test_permission_denied_file(self):
        path = self._make_file("secret.txt", b"secret")
        path.chmod(0o000)
        try:
            with self.assertRaises(ValueError):
                ts.read_path(str(path), 1024)
        finally:
            path.chmod(0o644)


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


class TestConfigValidate(unittest.TestCase):
    def setUp(self):
        self.original_models = deepcopy(cfg.MODELS)

    def tearDown(self):
        cfg.MODELS.clear()
        cfg.MODELS.update(deepcopy(self.original_models))

    def test_baseline_passes(self):
        cfg.validate()

    def test_missing_model_field_raises(self):
        del cfg.MODELS["9B"]["port"]
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
