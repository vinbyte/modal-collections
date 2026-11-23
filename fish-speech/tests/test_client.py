import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "fish_speech_client", Path(__file__).resolve().parents[1] / "client.py"
)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class AudioResponse(io.BytesIO):
    def __init__(self, body=b"generated audio", content_type="audio/mpeg"):
        super().__init__(body)
        self.headers = {"Content-Type": content_type}


class ClientTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.reference = self.folder / "baseline.mp3"
        self.reference.write_bytes(b"baseline audio")
        self.transcript = self.folder / "baseline.txt"
        self.transcript.write_text("\ufeff[calm] Halo bro. [excited] Berhasil!", encoding="utf-8")
        self.output = self.folder / "result.mp3"
        self.args = [
            "--url", "https://example.test/ui",
            "--reference-audio", str(self.reference),
            "--reference-text-file", str(self.transcript),
            "--text", "[whisper] Rahasia.",
            "--output", str(self.output),
        ]

    def run_client(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as stderr:
            status = client.main(args)
        return status, stderr.getvalue()

    def test_reference_files_match_ui_payload_and_target_tags_are_preserved(self):
        with patch.object(client.urllib.request, "urlopen", return_value=AudioResponse()) as request:
            status, _ = self.run_client(self.args + ["--strip-reference-tags", "--seed", "42"])
        self.assertEqual(status, 0)
        sent = request.call_args.args[0]
        self.assertEqual(sent.full_url, "https://example.test/v1/tts")
        payload = json.loads(sent.data)
        self.assertEqual(payload["text"], "[whisper] Rahasia.")
        self.assertEqual(payload["references"][0]["text"], "Halo bro.  Berhasil!")
        self.assertEqual(client.base64.b64decode(payload["references"][0]["audio"]), self.reference.read_bytes())
        self.assertEqual(payload["references"][0]["speaker"], 0)
        self.assertEqual(payload["seed"], 42)
        self.assertNotIn("temperature", payload)
        self.assertEqual(self.output.read_bytes(), b"generated audio")

    def test_inline_transcript_and_text_file(self):
        text_file = self.folder / "script.txt"
        text_file.write_text("[laughing] Lucu banget!", encoding="utf-8")
        args = client.parse_args([
            "--url", "https://example.test/",
            "--reference-audio", str(self.reference),
            "--reference-text", "Halo bro.",
            "--text-file", str(text_file),
            "--output", "output.wav",
        ])
        payload = client.build_payload(args)
        self.assertEqual(payload["text"], "[laughing] Lucu banget!")
        self.assertEqual(payload["references"][0]["text"], "Halo bro.")
        self.assertEqual(payload["format"], "wav")

    def test_http_error_does_not_overwrite_audio(self):
        self.output.write_bytes(b"existing audio")
        error = urllib.error.HTTPError("https://example.test/v1/tts", 503, "Unavailable", {}, io.BytesIO(b"Model unavailable"))
        with patch.object(client.urllib.request, "urlopen", side_effect=error):
            status, stderr = self.run_client(self.args)
        self.assertEqual(status, 1)
        self.assertIn("HTTP 503: Model unavailable", stderr)
        self.assertEqual(self.output.read_bytes(), b"existing audio")

    def test_non_audio_response_is_not_saved_as_mp3(self):
        with patch.object(client.urllib.request, "urlopen", return_value=AudioResponse(b'{"error":"failed"}', "application/json")):
            status, stderr = self.run_client(self.args)
        self.assertEqual(status, 1)
        self.assertIn("instead of audio", stderr)
        self.assertFalse(self.output.exists())

    def test_output_cannot_overwrite_baseline(self):
        with patch.object(client.urllib.request, "urlopen") as request:
            status, _ = self.run_client(self.args + ["--output", str(self.reference)])
        self.assertEqual(status, 1)
        request.assert_not_called()
        self.assertEqual(self.reference.read_bytes(), b"baseline audio")


if __name__ == "__main__":
    unittest.main()
