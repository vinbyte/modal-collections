# Fish Speech on Modal

Run Fish Speech with its **official Awesome WebUI** at `/ui`: text-to-speech, reference voice uploads, multiple speakers, inline emotion instructions, audio playback/download, and sampling controls. The API runs at `/v1/*` on the same URL.

The default is **S2-Pro on one L40S**, with one container maximum and scale-to-zero after five idle minutes. You can also deploy **Drama 3 on one H100** to compare its expressive delivery. Opening the UI starts the GPU; keeping a container warm incurs GPU usage.

![Fish Speech Web UI with reference audio and generation settings](../docs/images/Screenshot%202026-11-23%20at%2013.43.44.png)

## Quick start

Run these commands from the repository root. Only the Modal SDK is needed locally; Modal builds the Python environment and Web UI remotely.

```bash
python -m pip install 'modal>=1.4.2'
modal setup  # once per machine

# CPU-only download to a persistent Volume; no GPU is allocated here.
modal run fish-speech/serve.py::download_model

# Temporary deployment; Ctrl+C stops it.
modal serve fish-speech/serve.py
```

Open the URL printed by Modal **with `/ui` appended**, for example `https://<your-endpoint>.modal.run/ui`. The bare `/` has no page. The first request loads the model and can take several minutes.

For a persistent deployment:

```bash
modal deploy fish-speech/serve.py

# Stop the deployment when finished. Volumes remain available.
modal app stop fish-speech
```

The endpoint is public and has no login by default. Anyone with its URL can use the GPU and access API-saved reference voices; keep the URL private for personal testing.

## Test voice and emotion

1. Open `/ui` and add a speaker with **10–30 seconds of clean reference audio** and its exact transcript. Use a quiet, single-speaker recording. A reference gives comparisons a consistent timbre; without one, the model chooses a voice.
2. Paste a short script into the text editor, including the square-bracket tags.
3. Start with temperature **0.7**, top-p **0.7**, and WAV output. Leave normalization off for the first tag comparison.
4. Generate, listen, and download the result. Compare a neutral sentence and a tagged version with the same reference and settings.

```text
[whisper] Sini, aku mau cerita rahasia. Tapi jangan bilang siapa-siapa, ya.
[laughing] Serius? Kamu datang ke kantor pakai sandal yang beda warna?
[sad] Aku kira kita masih punya banyak waktu.
[excited] Akhirnya berhasil! Kita benar-benar menang!
```

Use one line per generation for the first comparison. More Indonesian and English scripts are in [`emotion-examples.txt`](emotion-examples.txt).

Fish Speech uses **free-form natural-language instructions**, so you can also try `[whisper in small voice]` or `[professional broadcast tone]`. Common tags documented for S2-Pro include:

| Delivery | Tags to try |
| --- | --- |
| Whisper / volume | `[whisper]`, `[low voice]`, `[volume down]`, `[shouting]` |
| Laughter | `[laughing]`, `[chuckle]`, `[chuckling]`, `[laughing tone]` |
| Emotion | `[excited]`, `[angry]`, `[sad]`, `[surprised]`, `[shocked]` |
| Breath / timing | `[sigh]`, `[inhale]`, `[exhale]`, `[pause]`, `[short pause]` |
| Emphasis | `[emphasis]`, `[singing]` |

For laughter, start with the officially documented `[laughing]` or `[chuckle]`. You can experiment with `[laugh]` as a free-form instruction, but there is no guarantee that it behaves like an ElevenLabs tag. Effects vary by sentence, language, reference, and generation; equal quality to ElevenLabs has not been established by this deployment.

For two speakers, add references to speaker 0 and speaker 1, then use:

```text
<|speaker:0|>[whisper] Kamu dengar sesuatu?
<|speaker:1|>[laughing] Tenang, itu cuma kucingku.
```

The reference uploads in the official UI belong to the current browser session. To save a reusable named voice across container restarts, use the reference API below.

## Compare Drama 3

Drama 3 is the newer 35B-A3B model. Upstream describes stronger laughter, crying, and whispers. This deployment uses its FP8 experts on an H100 and keeps `torch.compile` disabled, as required by the model.

```bash
FISH_MODEL=drama-3 modal run fish-speech/serve.py::download_model
FISH_MODEL=drama-3 modal serve fish-speech/serve.py

# Or persist the same configuration:
FISH_MODEL=drama-3 modal deploy fish-speech/serve.py
```

Set `FISH_MODEL` for **both download and serve/deploy**. The two checkpoints occupy separate directories in the same Volume. Redeploying replaces the model in the `fish-speech` app; model selection is not per request. To switch back, run `modal deploy fish-speech/serve.py` with `FISH_MODEL` unset or set to `s2-pro`.

## Generate from the command line

Use [`client.py`](client.py) to send a local baseline audio file, its transcript, and new speech text to your running Modal endpoint. This client needs only Python 3.10+; it does not need the Modal SDK or local model weights.

Keep `modal serve` running in another terminal, or use a persistent deployment URL:

```bash
export FISH_URL='https://<your-endpoint>.modal.run'

python3 fish-speech/client.py \
  --reference-audio /path/to/baseline.mp3 \
  --reference-text-file /path/to/baseline.txt \
  --strip-reference-tags \
  --text '[whisper] Sini, aku mau cerita rahasia. [laughing] Kamu lucu banget!' \
  --output fish-result.mp3
```

`--strip-reference-tags` removes bracketed directions such as `[calm]` and `[excited]` from the baseline transcript. Tags in the new speech text are preserved. Omit this flag if the transcript already contains only spoken words.

Use `--reference-text 'Exact spoken transcript'` instead of `--reference-text-file` for inline text. For a longer script, replace `--text` with `--text-file /path/to/script.txt`. The output extension selects MP3, WAV, Opus, or raw PCM. Optional `--seed`, `--temperature`, and `--top-p` control sampling; unspecified values use the model's defaults. Run `python3 fish-speech/client.py --help` for all options.

You can pass `--url` instead of setting `FISH_URL`; copying the Web UI URL ending in `/ui` also works. The model is whichever one that endpoint loaded, so a Drama 3 endpoint uses Drama 3 without setting `FISH_MODEL` on the client. The baseline is sent with each request, and the client saves the returned audio locally. It does not save a named reference voice on the server.

### Zephlyn baseline example

Run from the repository root with `modal serve` active in another terminal. These paths point to the local ElevenLabs baseline and transcript; the files are outside this repository. Update `--url` if Modal prints a different endpoint.

```bash
REF_DIR="/Users/gavinda/Documents/projects/feedz.work/elevenlabs_baseline_audio"

python3 fish-speech/client.py \
  --url "https://gavindakinandana--fish-speech-fishspeech-serve-dev.us-east.modal.run" \
  --reference-audio "$REF_DIR/elevenlabs_zephlyn_30s_baseline.mp3" \
  --reference-text-file "$REF_DIR/elevenlabs_zephlyn_30s_baseline.txt" \
  --strip-reference-tags \
  --text '[whisper] Sini, aku mau cerita rahasia. [laughing] Kamu lucu banget!' \
  --output fish-zephlyn-test.mp3
```

## API examples

The upstream server accepts JSON as well as MessagePack. Set `BASE_URL` to the endpoint printed by Modal, without `/ui` or a trailing slash.

```bash
export BASE_URL='https://<your-endpoint>.modal.run'
curl --fail-with-body "$BASE_URL/v1/health"

curl --fail-with-body "$BASE_URL/v1/tts" \
  -H 'Content-Type: application/json' \
  -d '{"text":"[whisper] Jangan bilang siapa-siapa, ya.","format":"wav","streaming":false,"temperature":0.7,"top_p":0.7,"seed":42,"normalize":false}' \
  -o whisper.wav
```

Save a reference voice to the persistent `fish-speech-references` Volume:

```bash
curl --fail-with-body "$BASE_URL/v1/references/add" \
  -F 'id=my-voice' \
  -F 'audio=@reference.wav;type=audio/wav' \
  -F 'text=Transkrip persis dari rekaman referensi.'

curl --fail-with-body "$BASE_URL/v1/tts" \
  -H 'Content-Type: application/json' \
  -d '{"text":"[laughing] Kamu lucu banget!","reference_id":"my-voice","format":"wav","normalize":false,"seed":42}' \
  -o laughing.wav
```

Use WAV with `streaming: true` for streaming responses. The official UI handles playback and downloads. Keep scripts under 5,000 characters; this deployment handles one HTTP request at a time with one backend worker to avoid simultaneous GPU inference.

## Configuration

Set environment variables on the local `modal` command. The deployment carries them into the remote containers.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FISH_MODEL` | `s2-pro` | `s2-pro` or `drama-3` |
| `FISH_GPU` | `L40S` / `H100` | One GPU; H100 is the default for Drama 3 |
| `FISH_COMPILE` | `0` | Set `1` for S2-Pro compilation; longer first startup |
| `FISH_MIN_CONTAINERS` | `0` | Set `1` to keep the GPU warm |
| `FISH_SCALEDOWN_WINDOW` | `300` | Idle seconds before scaling to zero |
| `FISH_HF_SECRET` | empty | Optional Modal Secret name containing `HF_TOKEN`, used only by the downloader |

Both pinned checkpoint repositories were public when checked. If authentication is needed, create a Modal Secret containing `HF_TOKEN`, then use `FISH_HF_SECRET=<secret-name>` on the download command. Keep the token in the Secret, outside source code.

```bash
FISH_COMPILE=1 FISH_MIN_CONTAINERS=1 modal deploy fish-speech/serve.py
```

## Storage and reproducibility

| Modal Volume | Container path | Contents |
| --- | --- | --- |
| `fish-speech-models` | `/models` | Checkpoints grouped by model and immutable HF revision; resumable downloads |
| `fish-speech-references` | `/opt/fish-speech/references` | Voices saved through the reference API |
| `fish-speech-cache` | `/cache` | Torch, Triton, and compilation caches |

The backend and UI use upstream commit `0f8a12e8a651062fdee9c33a894bdccd3a8513f1`, with its frozen `uv.lock` and `package-lock.json`. A small [`webui-layout.patch`](webui-layout.patch) keeps Reference Audio visible alongside Generation Settings on shorter desktop windows and lets the page scroll. Model revisions are pinned in `serve.py`. The downloader checks codec, tokenizer, config, and every indexed weight shard before marking a checkpoint ready. GPU containers use local checkpoints and do not download them during startup. Generated audio is returned to the browser/client; download it to keep a copy.

If startup reports a missing/incomplete checkpoint, rerun the download for the selected model. If you get a Web UI 404, use `/ui` exactly. If an emotion sounds weak, try a shorter sentence, a neutral reference, and an English comparison from the examples file. Drama 3 can also be compared with the same reference.

## Validation

Local checks cover incomplete checkpoints, readiness polling, process failures, and termination. They require the Modal SDK and do not allocate a GPU:

```bash
python -m unittest discover -s fish-speech/tests -v
```

GPU inference and listening quality still need a live Modal run.

## Sources and license

See the official [installation guide](https://speech.fish.audio/install/), [inference and Web UI guide](https://speech.fish.audio/inference/), [S2-Pro model card and tag examples](https://huggingface.co/fishaudio/s2-pro), and [Drama 3 model card](https://huggingface.co/fishaudio/drama-3).

S2-Pro weights use the Fish Audio Research License for research/non-commercial use; commercial use requires a separate license. Consult each model's license before changing the intended use. The current codebase and Drama 3 use the Fish Audio Community License; the exact terms are included in the pinned upstream repositories.
