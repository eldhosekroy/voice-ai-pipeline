# Voice AI Pipeline (`voice-ai-pipeline`)

A modular, real-time voice AI pipeline built in Python 3.11+.

## Target Pipeline Architecture

```text
Microphone Audio Input (STEP 2 - Active)
        ↓
Silero VAD (Voice Activity Detection)
        ↓
Whisper STT (Speech-to-Text)
        ↓
Qwen via Ollama (Local LLM Response)
        ↓
Kokoro TTS (Text-to-Speech)
        ↓
Speaker Audio Output
```

## Features & Principles

- **Modular Design**: Every AI component (Audio Input, VAD, STT, LLM, TTS) is isolated behind clean Abstract Base Class (`ABC`) interfaces.
- **Hardware Agnostic**: Device selection (CPU / GPU) and compute precision are configured via environment variables for easy handoff to team testing.
- **Local Execution**: LLM runs locally through Ollama (Qwen model).

---

## Directory Layout

```text
voice-ai-pipeline/
│
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   └── microphone.py     # BaseAudioInput interface & Microphone capture implementation
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── pipeline.py       # Pipeline orchestrator
│   │
│   ├── vad/
│   │   ├── __init__.py
│   │   └── silero_vad.py     # BaseVAD & SileroVAD placeholder
│   │
│   ├── stt/
│   │   ├── __init__.py
│   │   └── whisper.py        # BaseSTT & WhisperSTT placeholder
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── qwen.py           # BaseLLM & QwenLLM placeholder
│   │
│   └── tts/
│       ├── __init__.py
│       └── kokoro.py         # BaseTTS & KokoroTTS placeholder
│
├── tests/
│   └── __init__.py
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

Settings are defined in `config/config.py` and can be overridden via environment variables:

| Environment Variable | Default Value | Description |
|----------------------|---------------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service endpoint |
| `OLLAMA_MODEL` | `qwen2.5` | Qwen LLM model tag |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, etc.) |
| `WHISPER_DEVICE` | `cpu` | Device hardware (`cpu` or `cuda`) |
| `WHISPER_COMPUTE_TYPE` | `int8` | Precision (`int8`, `float16`, `float32`) |
| `KOKORO_VOICE` | `af_heart` | Voice profile for Kokoro TTS |
| `SAMPLE_RATE` | `16000` | Audio sampling rate in Hz |
| `VAD_THRESHOLD` | `0.5` | Silero VAD speech probability threshold |
| `VAD_MIN_SPEECH_DURATION` | `0.25` | Min speech segment duration (seconds) |
| `VAD_MIN_SILENCE_DURATION` | `0.5` | Min silence duration to end segment (seconds) |

---

## Setup & Running (STEP 2)

### 1. Create Virtual Environment

```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install STEP 2 Dependencies

```bash
pip install sounddevice numpy
```

### 3. Run the Application & Microphone Test

```bash
python main.py
```

Expected log output demonstrating microphone stream initialization, audio chunk capture, and clean shutdown:
```text
[INFO] [AUDIO] Starting microphone stream (sample_rate=16000, chunk_size=512, channels=1)...
[INFO] [AUDIO] Microphone stream started successfully.
[INFO] [AUDIO] Capture test complete: Read 63 chunks (32256 total samples).
[INFO] [AUDIO] Stopping microphone stream...
[INFO] [AUDIO] Microphone stream stopped.
```

---

## Development Roadmap

- [x] **STEP 1**: Project structure, abstract interfaces, configuration & stubs
- [x] **STEP 2**: Microphone audio capture (`src/audio/microphone.py`) & buffer management
- [ ] **STEP 3**: Silero VAD integration & speech segment extraction
- [ ] **STEP 4**: Whisper STT integration & transcription
- [ ] **STEP 5**: Ollama + Qwen LLM integration
- [ ] **STEP 6**: Kokoro TTS audio synthesis
- [ ] **STEP 7**: Speaker audio playback
- [ ] **STEP 8**: Connect components into `VoicePipeline` loop
- [ ] **STEP 9**: Logging & graceful error handling
- [ ] **STEP 10**: Testing & complete end-to-end verification
