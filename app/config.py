import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5001))
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')

    SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', 24000))
    CHUNK_DURATION_MS = int(os.getenv('CHUNK_DURATION_MS', 30))

    # STT (faster-whisper)
    WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
    WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')
    WHISPER_COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'int8')

    # LLM (llama-cpp, currently Qwen 2.5 0.5B GGUF)
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', 128))
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))

    # TTS (Kokoro-82M)
    KOKORO_MODEL_NAME = os.getenv('KOKORO_MODEL_NAME', 'hexgrad/Kokoro-82M')
    KOKORO_VOICE = os.getenv('KOKORO_VOICE', 'af_sarah')
    KOKORO_SPEED = float(os.getenv('KOKORO_SPEED', '1.0'))
    KOKORO_DEVICE = os.getenv('KOKORO_DEVICE', 'cpu')

    HF_LOCAL_READ_TOKEN = os.getenv("HF_LOCAL_READ_TOKEN")

    SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT',
        "You are a helpful voice assistant. Provide concise, natural responses suitable for speech. "
        "Keep answers brief and conversational, as they will be spoken aloud.")

    MAX_HISTORY = int(os.getenv('MAX_HISTORY', 10))

    _LOG_ENV = (os.getenv('APP_ENV') or os.getenv('ENV') or '').strip().lower()
    ENABLE_LOCAL_LOGS = (_LOG_ENV == 'local' or os.getenv('IS_LOCAL', 'false').lower() == 'true')
    LOG_DIR = os.getenv('LOG_DIR', 'logs')
