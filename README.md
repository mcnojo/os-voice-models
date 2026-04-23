# os-voice

A, shrimple, local voice assistant that runs entirely on-device. Speak into your browser, get a spoken response back — no API calls leave your machine.

**Stack:** Faster-Whisper (STT) → Qwen 2.5 0.5B via llama-cpp (LLM) → Kokoro-82M (TTS)

Everything runs on CPU (GPU experiements coming in the future) – the full pipeline is ~1.2GB, you'll have no issue running this on even a very old MacBook, though expect a smidge (1-3s) for the full pipeline to run if so.

## How it works

```
mic audio (WebSocket) -> Faster-Whisper transcription -> Qwen chat completion -> Kokoro speech synthesis -> audio back to browser
```

The browser records audio, ships PCM16 over a WebSocket to a Flask server, and plays back the synthesized response. Conversation history is maintained across turns.

## Setup

### Prerequisites

- Docker (for containerized setup), **or** Python 3.11+ with `espeak-ng` installed
- ~1GB disk for model weights

### 0. (Create env / download reqs.)

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 1. Download the LLM

Grab the GGUF-quantized Qwen model and drop it in `models/`:

```bash
mkdir -p models
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models
```

(Kokoro and Whisper weights are downloaded automatically on first run)

### 2. Configure

Copy the example env and fill in with your values:

```bash
cp .env.example .env
```

The defaults work out of the box, these are the main things you'd want to change:


| Variable         | Default       | Notes                                      |
| ---------------- | ------------- | ------------------------------------------ |
| `WHISPER_MODEL`  | `base`        | `tiny` is faster, `small` is more accurate |
| `KOKORO_VOICE`   | `bf_isabella` | See voice list below                       |
| `LLM_THREADS`    | `8`           | Match to your CPU core count               |
| `LLM_MAX_TOKENS` | `128`         | Keep low for faster voice responses        |


### 3. Run

**With Docker (recommended):**

```bash
./start.sh
# opens at http://localhost:5001
```

**Without Docker:**

```bash
# install espeak-ng (macOS)
brew install espeak-ng

pip install -r requirements.txt
python -m app.main
```

### 4. Try it

Open `http://localhost:5001`, click "Start Recording", speak, click "Stop Recording". You'll see the transcription, the LLM's text response, and hear the spoken reply.

## Project structure

```
app/
  main.py            Flask server + WebSocket handler
  config.py          Env configuration
  stt_service.py     Faster-Whisper wrapper
  llm_service.py     llama-cpp-python wrapper (Qwen GGUF)
  tts_service.py     Kokoro-82M wrapper
  audio_pipeline.py  Wires STT → LLM → TTS together
templates/
  index.html         Browser UI (vanilla JS, records mic, plays audio)
models/              GGUF model files (gitignored)
```

## Available voices

Kokoro ships a bunch of voice options out of the box:

- **American:** `af_heart`, `af_sarah`, `af_nicole`, `af_sky`, `am_adam`, `am_michael`
- **British:** `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`

Set via `KOKORO_VOICE` in `.env`



## Future directions:

1. VAD based Interruption detection
2. More beautiful/functional frontend
3. Integrate as a hypervisor assistant for other ongoign projects (MCP based)

