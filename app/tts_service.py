import logging
from typing import Optional

import numpy as np
import soundfile as sf

from app.config import Config
from app.tts_backends import create_tts_backend

logger = logging.getLogger(__name__)


class TTSService:
    """Facade over pluggable TTS backends (kokoro, omnivoice, supertonic)."""

    def __init__(self):
        self.config = Config()
        engine = self.config.TTS_ENGINE
        logger.info("Creating TTS backend: %s", engine)
        self._backend = create_tts_backend(engine)
        logger.info("TTS backend ready (engine=%s, sample_rate=%d)", engine, self.sample_rate)

    @property
    def sample_rate(self) -> int:
        return self._backend.sample_rate

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lang: Optional[str] = None,
    ) -> bytes:
        """Synthesize text to PCM16 bytes."""
        return self._backend.synthesize(text, voice=voice, speed=speed, lang=lang)

    def synthesize_to_file(self, text: str, output_path: str, **kwargs) -> bool:
        audio_bytes = self.synthesize(text, **kwargs)
        if not audio_bytes:
            return False
        try:
            sf.write(output_path, np.frombuffer(audio_bytes, dtype=np.int16), self.sample_rate)
            logger.info("Saved audio to %s", output_path)
            return True
        except Exception as e:
            logger.error("Failed to save audio: %s", e)
            return False

    def get_available_voices(self) -> list[str]:
        return self._backend.get_available_voices()
