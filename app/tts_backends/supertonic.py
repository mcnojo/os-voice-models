import logging
from typing import Optional

import numpy as np

from app.config import Config
from app.tts_backends.base import TTSBackend

logger = logging.getLogger(__name__)


class SupertonicBackend(TTSBackend):

    def __init__(self):
        self._config = Config()
        self._tts = None
        self._load_model()

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    def _load_model(self):
        from supertonic import TTS

        logger.info("Loading Supertonic-3 model")
        self._tts = TTS(model="supertonic-3", auto_download=True)
        logger.info("Supertonic-3 model ready (sample_rate=%d)", self._tts.sample_rate)

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lang: Optional[str] = None,
    ) -> bytes:
        if not text or not text.strip():
            return b""

        voice = voice or self._config.SUPERTONIC_VOICE
        speed = float(speed or self._config.SUPERTONIC_SPEED)
        steps = self._config.SUPERTONIC_STEPS
        lang = lang or self._config.SUPERTONIC_LANG

        logger.info(
            "Synthesizing %d chars, voice='%s', speed=%s, steps=%d, lang='%s'",
            len(text), voice, speed, steps, lang,
        )

        style = self._tts.get_voice_style(voice_name=voice)

        try:
            wav, duration = self._tts.synthesize(
                text=text,
                voice_style=style,
                speed=speed,
                total_steps=steps,
                lang=lang,
            )
        except Exception as e:
            logger.error("Supertonic synthesis failed: %s", e)
            return b""

        # wav shape is (1, samples) — squeeze to 1-D
        audio = wav.squeeze()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32_767.0).astype(np.int16).tobytes()

    def get_available_voices(self) -> list[str]:
        return list(self._tts.voice_style_names)
