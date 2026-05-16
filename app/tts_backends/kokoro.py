import logging
from typing import Optional, List

import numpy as np
import torch
from kokoro import KPipeline

from app.config import Config
from app.tts_backends.base import TTSBackend

logger = logging.getLogger(__name__)

VOICE_LANG_MAP = {
    'a': 'a',  # American English
    'b': 'b',  # British English
    'e': 'e',  # Spanish
    'f': 'f',  # French
    'h': 'h',  # Hindi
    'i': 'i',  # Italian
    'j': 'j',  # Japanese (requires misaki[ja])
    'p': 'p',  # Portuguese
    'z': 'z',  # Mandarin (requires misaki[zh])
}


def voice_to_lang_code(voice: str) -> str:
    prefix = (voice or "af_heart").split('_', 1)[0][0]
    return VOICE_LANG_MAP.get(prefix, 'a')


class KokoroBackend(TTSBackend):

    def __init__(self):
        self._config = Config()
        self._pipeline = None
        self._current_lang = None
        self._ensure_pipeline(self._config.KOKORO_VOICE)

    @property
    def sample_rate(self) -> int:
        return 24_000

    def _ensure_pipeline(self, voice: str):
        lang = voice_to_lang_code(voice)
        if self._pipeline is None or self._current_lang != lang:
            logger.info("Initializing Kokoro pipeline (lang_code='%s')", lang)
            self._pipeline = KPipeline(lang_code=lang, repo_id='hexgrad/Kokoro-82M')
            self._current_lang = lang
            logger.info("Kokoro pipeline ready")

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lang: Optional[str] = None,
    ) -> bytes:
        if not text or not text.strip():
            return b""

        voice = voice or self._config.KOKORO_VOICE
        speed = float(speed or self._config.KOKORO_SPEED)
        self._ensure_pipeline(voice)

        logger.info("Synthesizing %d chars, voice='%s', speed=%s", len(text), voice, speed)

        chunks: List[np.ndarray] = []
        try:
            for _, _, audio in self._pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
                audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)
                chunks.append(audio_np)
        except Exception as e:
            logger.error("Kokoro synthesis failed: %s", e)
            return b""

        if not chunks:
            logger.error("Kokoro returned no audio.")
            return b""

        audio = np.clip(np.concatenate(chunks), -1.0, 1.0)
        return (audio * 32_767.0).astype(np.int16).tobytes()

    def get_available_voices(self) -> list[str]:
        return [
            "af_heart", "af_sarah", "af_nicole", "af_sky",
            "am_adam", "am_michael",
            "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
        ]
