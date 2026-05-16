import logging
from typing import Optional

import numpy as np

from app.config import Config
from app.tts_backends.base import TTSBackend

logger = logging.getLogger(__name__)

# Preset voice-design instructions keyed by friendly name.
# Users can also pass a raw instruct string as the voice parameter.
VOICE_PRESETS = {
    "female_american": "female, medium pitch, american accent",
    "female_british": "female, medium pitch, british accent",
    "male_american": "male, medium pitch, american accent",
    "male_british": "male, medium pitch, british accent",
    "female_low": "female, low pitch, american accent",
    "male_low": "male, low pitch, american accent",
    "female_high": "female, high pitch, american accent",
    "male_high": "male, high pitch, american accent",
}

DEFAULT_VOICE = "female_american"


class OmniVoiceBackend(TTSBackend):

    def __init__(self):
        self._config = Config()
        self._model = None
        self._load_model()

    @property
    def sample_rate(self) -> int:
        return 24_000

    def _load_model(self):
        import torch
        from omnivoice import OmniVoice

        device = self._config.OMNIVOICE_DEVICE
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but unavailable (e.g. Docker), falling back to CPU")
            device = "cpu"

        logger.info("Loading OmniVoice model (device='%s')", device)
        self._model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=torch.float16,
        )
        logger.info("OmniVoice model ready")

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lang: Optional[str] = None,
    ) -> bytes:
        if not text or not text.strip():
            return b""

        voice = voice or self._config.OMNIVOICE_VOICE
        instruct = VOICE_PRESETS.get(voice, voice)
        speed = float(speed or self._config.OMNIVOICE_SPEED)
        steps = self._config.OMNIVOICE_STEPS

        logger.info(
            "Synthesizing %d chars, instruct='%s', speed=%s, steps=%d",
            len(text), instruct, speed, steps,
        )

        try:
            audio_list = self._model.generate(
                text=text,
                instruct=instruct,
                speed=speed,
                num_step=steps,
            )
        except Exception as e:
            logger.error("OmniVoice synthesis failed: %s", e)
            return b""

        if not audio_list:
            logger.error("OmniVoice returned no audio.")
            return b""

        audio = np.concatenate(audio_list) if len(audio_list) > 1 else audio_list[0]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32_767.0).astype(np.int16).tobytes()

    def get_available_voices(self) -> list[str]:
        return list(VOICE_PRESETS.keys())
