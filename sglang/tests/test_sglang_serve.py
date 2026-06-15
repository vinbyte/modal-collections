from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SERVE = ROOT / "sglang" / "serve.py"


class SGLangServeTests(unittest.TestCase):
    def test_sglang_entrypoint_configuration(self):
        source = SERVE.read_text()

        expected_snippets = [
            'MODEL_NAME = "microsoft/FastContext-1.0-4B-SFT"',
            '.apt_install("libnuma1")',
            'POST /v1/chat/completions',
            'POST /v1/completions',
            'GET  /v1/models',
            'GPU_TYPE = "L40S"',
            'MIN_CONTAINERS = 1',
            'TIMEOUT = 600',
            'SCALEDOWN_WINDOW = 900',
            'STARTUP_TIMEOUT = 600',
            'MAX_CONCURRENT_INPUTS = 4',
            'ENABLE_MEMORY_SNAPSHOT = False',
            'ENABLE_GPU_SNAPSHOT = True',
            'python3", "-m", "sglang.launch_server"',
            '"--host", "0.0.0.0"',
            '"--port", str(SGLANG_PORT)',
            '"--model-path", MODEL_NAME',
            '"--served-model-name", MODEL_ALIAS',
            'CONTEXT_LENGTH = 262144',
            'MEM_FRACTION_STATIC = 0.8',
            'DTYPE = "bfloat16"',
            'TRUST_REMOTE_CODE = True',
            'TOOL_CALL_PARSER = "qwen"',
            '"--dtype", DTYPE',
            '"--trust-remote-code"',
            '"--tool-call-parser", TOOL_CALL_PARSER',
            'raise RuntimeError("SGLang exited before becoming ready")',
            'raise RuntimeError("SGLang did not become ready before timeout")',
        ]

        for snippet in expected_snippets:
            self.assertIn(snippet, source)


if __name__ == "__main__":
    unittest.main()
