"""Exercise deployment failure handling without contacting Modal or loading a GPU."""

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


spec = importlib.util.spec_from_file_location(
    "fish_speech_deploy", Path(__file__).resolve().parents[1] / "serve.py"
)
deploy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = deploy
with patch.dict(os.environ, {"FISH_MODEL": "s2-pro", "FISH_COMPILE": "0"}):
    spec.loader.exec_module(deploy)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name)
        for name in ("config.json", "tokenizer.json", "codec.pth", "weights.safetensors"):
            (self.checkpoint / name).write_bytes(b"data")
        (self.checkpoint / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer.weight": "weights.safetensors"}})
        )

    def test_complete_checkpoint(self):
        deploy.validate_checkpoint(self.checkpoint)

    def test_missing_codec(self):
        (self.checkpoint / "codec.pth").unlink()
        with self.assertRaisesRegex(RuntimeError, "codec.pth"):
            deploy.validate_checkpoint(self.checkpoint)

    def test_empty_weight_shard(self):
        (self.checkpoint / "weights.safetensors").write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "weights.safetensors"):
            deploy.validate_checkpoint(self.checkpoint)

    def test_failed_download_keeps_partial_files_without_ready_marker(self):
        marker = self.checkpoint / ".download_complete"
        marker.write_text("old")
        volume = MagicMock()
        with (
            patch.object(deploy, "CHECKPOINT", self.checkpoint),
            patch.object(deploy, "models_volume", volume),
            patch.object(deploy.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "hf")),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                deploy.download_model.local()
        self.assertFalse(marker.exists())
        self.assertTrue((self.checkpoint / "weights.safetensors").exists())
        volume.commit.assert_called_once()

    def test_successful_download_commits_ready_marker(self):
        volume = MagicMock()
        with (
            patch.object(deploy, "CHECKPOINT", self.checkpoint),
            patch.object(deploy, "models_volume", volume),
            patch.object(deploy.subprocess, "run"),
        ):
            deploy.download_model.local()
        self.assertEqual(
            (self.checkpoint / ".download_complete").read_text(), deploy.MODEL_REVISION
        )
        volume.commit.assert_called_once()


class ServerLifecycleTests(unittest.TestCase):
    def test_waits_for_health_after_connection_failure(self):
        process = MagicMock()
        process.poll.return_value = None
        response = io.BytesIO(b'{"status":"ok"}')
        response.status = 200
        with (
            patch.object(deploy.urllib.request, "urlopen", side_effect=[OSError(), response]) as request,
            patch.object(deploy.time, "sleep"),
        ):
            deploy.wait_for_server(process, timeout=5)
        self.assertEqual(request.call_count, 2)

    def test_crashed_process_fails_immediately(self):
        process = MagicMock(returncode=17)
        process.poll.return_value = 17
        with patch.object(deploy.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(RuntimeError, "code 17"):
                deploy.wait_for_server(process, timeout=5)
        request.assert_not_called()

    def test_startup_timeout(self):
        process = MagicMock()
        process.poll.return_value = None
        with (
            patch.object(deploy.time, "monotonic", side_effect=[0, 0, 10]),
            patch.object(deploy.time, "sleep"),
            patch.object(deploy.urllib.request, "urlopen", side_effect=OSError()),
        ):
            with self.assertRaises(TimeoutError):
                deploy.wait_for_server(process, timeout=5)

    def test_unresponsive_process_is_killed(self):
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("server", 15), 0]
        deploy.terminate_server(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
