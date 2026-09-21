# Voice AI Pipeline (`voice-ai-pipeline`)

A modular, real-time voice AI pipeline built in Python 3.11+.

## Pipeline Architecture

```text
Microphone Audio Input (`src/audio/microphone.py`)
        ↓
Silero VAD (`src/vad/silero_vad.py`)              - Voice Activity Detection (ONNX Runtime)
        ↓  (Speech segment numpy array)
Whisper STT (`src/stt/whisper.py`)                - Speech-to-Text (faster-whisper)
        ↓  (Transcribed text string)
Qwen via Ollama (`src/llm/qwen.py`)               - Local LLM Response Generation
        ↓  (AI response text string)
Kokoro TTS (`src/tts/kokoro.py`)                  - Text-to-Speech Audio Synthesis
        ↓  (Synthesized audio array)
Speaker Audio Output (`src/audio/output.py`)     - System Speaker Playback
```

---

## Features & Principles

- **Modular Architecture**: Every AI component (Mic, VAD, STT, LLM, TTS, Speaker) is decoupled behind abstract interface classes (`ABC`) and wired via dependency injection inside `VoicePipeline`.
- **CPU Default / GPU Configurable**: Default settings run efficiently on CPU without requiring CUDA or heavy GPU setup. Device configuration can be switched to GPU via environment variables (`WHISPER_DEVICE=cuda`, `VAD_DEVICE=cuda`).
- **Local Ollama Integration**: Communicates with local Ollama service for Qwen LLM inferencing (`http://localhost:11434`).
- **Independent Component Testing**: Each module includes isolated unit test coverage under `tests/`.

---

## Directory Layout

```text
voice-ai-pipeline/
│
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── microphone.py     # BaseAudioInput interface & Microphone capture implementation
│   │   └── output.py         # BaseAudioOutput interface & Speaker playback implementation
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── pipeline.py       # VoicePipeline orchestrator
│   │
│   ├── vad/
│   │   ├── __init__.py
│   │   └── silero_vad.py     # BaseVAD interface & Silero VAD (ONNX Runtime)
│   │
│   ├── stt/
│   │   ├── __init__.py
│   │   └── whisper.py        # BaseSTT interface & Whisper STT (faster-whisper)
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── qwen.py           # BaseLLM interface & Qwen via Ollama client
│   │
│   └── tts/
│       ├── __init__.py
│       └── kokoro.py         # BaseTTS interface & Kokoro / pyttsx3 / gTTS synthesis
│
├── tests/
│   ├── __init__.py
│   ├── test_audio.py
│   ├── test_vad.py
│   ├── test_stt.py
│   ├── test_llm.py
│   └── test_tts.py
│
├── config/
│   ├── __init__.py
│   └── config.py             # PipelineConfig environment settings
│
├── main.py                   # Application entry point
├── requirements.txt          # Dependency manifest
├── .gitignore
└── README.md
```

---

## Configuration Settings

All model names, devices, and thresholds are configurable via environment variables or `config/config.py`:

| Environment Variable | Default Value | Description |
|----------------------|---------------|-------------|
| `VAD_DEVICE` | `cpu` | Hardware device for Silero VAD (`cpu` or `cuda`) |
| `VAD_THRESHOLD` | `0.5` | Speech detection confidence threshold |
| `VAD_MIN_SPEECH_DURATION` | `0.25` | Min speech segment duration (seconds) |
| `VAD_MIN_SILENCE_DURATION` | `0.5` | Min silence duration to trigger speech end (seconds) |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`) |
| `WHISPER_DEVICE` | `cpu` | Hardware device for Whisper STT (`cpu` or `cuda`) |
| `WHISPER_COMPUTE_TYPE` | `int8` | Inference quantization (`int8`, `float16`, `float32`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama REST API endpoint |
| `OLLAMA_MODEL` | `qwen2.5` | Qwen LLM model tag |
| `KOKORO_VOICE` | `af_heart` | Voice profile for TTS |
| `KOKORO_DEVICE` | `cpu` | Hardware device for TTS synthesis (`cpu` or `cuda`) |
| `SAMPLE_RATE` | `16000` | Audio sampling rate in Hz |

---

## Model & Ollama Setup Guide

### 1. Ollama & Qwen Setup

1. Install [Ollama](https://ollama.com).
2. Start the Ollama service:
   ```bash
   ollama serve
   ```
3. Pull the Qwen model:
   ```bash
   ollama pull qwen2.5
   ```

### 2. Silero VAD & Whisper Models

- **Silero VAD**: Automatically downloads `silero_vad.onnx` to `~/.cache/silero_vad/` on first run.
- **Whisper STT**: Automatically downloads faster-whisper model weights (e.g. `base` or `tiny`) via Hugging Face Hub on first run.

---

## Quickstart & Execution

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests

```bash
python -m unittest discover -s tests
```

### 3. Run the Voice Pipeline

```bash
python main.py
```

---

## Troubleshooting

- **Ollama Offline**: If you see `[LLM] Could not connect to Ollama service`, ensure `ollama serve` is running in another terminal window.
- **Microphone Error**: Ensure your microphone is connected and non-exclusive audio access is enabled.
- **Whisper Memory Error**: If running on CPU with limited RAM, set `WHISPER_MODEL=tiny` in environment variables.
