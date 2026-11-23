"""Generate speech through a running Fish Speech endpoint using local reference files.

Uses only the Python standard library. Run `python3 fish-speech/client.py --help`.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("FISH_URL"), help="Modal base URL or /ui URL (default: FISH_URL)")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--text", help="Text to generate, including emotion tags")
    text.add_argument("--text-file", type=Path, help="UTF-8 file containing text to generate")
    parser.add_argument("--reference-audio", type=Path, required=True, help="Local baseline audio file, e.g. MP3 or WAV")
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-text", help="Exact transcript of the baseline audio")
    reference.add_argument("--reference-text-file", type=Path, help="UTF-8 file containing the baseline transcript")
    parser.add_argument("--strip-reference-tags", action="store_true", help="Remove [tags] from the reference transcript; keep tags in generated text")
    parser.add_argument("--output", type=Path, default=Path("speech.mp3"), help="Output path ending in .mp3, .wav, .opus, or .pcm (default: speech.mp3)")
    parser.add_argument("--seed", type=int, help="Optional random seed")
    parser.add_argument("--temperature", type=float, help="Omit to use the loaded model's default")
    parser.add_argument("--top-p", type=float, help="Omit to use the loaded model's default")
    parser.add_argument("--timeout", type=float, default=1800, help="HTTP timeout in seconds, including cold start (default: 1800)")
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("provide --url or set FISH_URL to your running Modal endpoint")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def tts_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("--url must be an http:// or https:// URL")
    path = parts.path.rstrip("/")
    if path.endswith("/ui"):
        path = path[:-3]
    if not path.endswith("/v1/tts"):
        path += "/v1/tts"
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def read_text(inline: str | None, file: Path | None) -> str:
    value = file.expanduser().read_text(encoding="utf-8-sig") if file is not None else inline
    if value is None or not value.strip():
        raise ValueError("Speech text and reference transcript must not be empty")
    return value.strip()


def build_payload(args) -> dict:
    text = read_text(args.text, args.text_file)
    reference_text = read_text(args.reference_text, args.reference_text_file)
    if args.strip_reference_tags:
        reference_text = re.sub(r"\[[^\]\n]*\]", "", reference_text).strip()
        if not reference_text:
            raise ValueError("Reference transcript is empty after removing tags")
    audio = args.reference_audio.expanduser().read_bytes()
    if not audio:
        raise ValueError("Reference audio file is empty")
    audio_format = args.output.suffix.lower().lstrip(".")
    if audio_format not in {"mp3", "wav", "opus", "pcm"}:
        raise ValueError("--output must end in .mp3, .wav, .opus, or .pcm")
    payload = {
        "text": text,
        "references": [{
            "audio": base64.b64encode(audio).decode("ascii"),
            "text": reference_text,
            "speaker": 0,
        }],
        "format": audio_format,
        "streaming": False,
        "normalize": False,
    }
    for name in ("seed", "temperature", "top_p"):
        value = getattr(args, name)
        if value is not None:
            payload[name] = value
    return payload


def generate(url: str, payload: dict, output: Path, timeout: float) -> int:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "audio/*"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if not content_type.startswith("audio/") and content_type != "application/octet-stream":
                detail = response.read(1000).decode("utf-8", errors="replace")
                raise ValueError(f"Server returned {content_type or 'no Content-Type'} instead of audio: {detail}")
            audio = response.read()
    except urllib.error.HTTPError as error:
        with error:
            detail = error.read(2000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    if not audio:
        raise ValueError("Server returned an empty audio response")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(audio)
    return len(audio)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        url = tts_url(args.url)
        payload = build_payload(args)
        output = args.output.expanduser()
        inputs = [args.reference_audio, args.reference_text_file, args.text_file]
        if any(path is not None and output.resolve() == path.expanduser().resolve() for path in inputs):
            raise ValueError("--output must differ from your input files")
        print("Generating audio; waiting for the model response...", flush=True)
        size = generate(url, payload, output, args.timeout)
        print(f"Saved {output.resolve()} ({size:,} bytes)")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
