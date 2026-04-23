import logging
from typing import Optional, List
import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline
from app.config import Config

logger = logging.getLogger(__name__)

# first letter of voice name -> Kokoro lang_code
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

SAMPLE_RATE = 24000


def voice_to_lang_code(voice: str) -> str:
    prefix = (voice or "af_heart").split('_', 1)[0][0]
    return VOICE_LANG_MAP.get(prefix, 'a')


class TTSService:
    """Text-to-speech via Kokoro-82M."""

    def __init__(self):
        self.config = Config()
        self.pipeline = None
        self.current_lang = None
        self._ensure_pipeline_for_voice(self.config.KOKORO_VOICE)

    def _ensure_pipeline_for_voice(self, voice: str):
        lang = voice_to_lang_code(voice)
        if self.pipeline is None or self.current_lang != lang:
            logger.info(f"Initializing Kokoro pipeline (lang_code='{lang}')")
            self.pipeline = KPipeline(lang_code=lang, repo_id='hexgrad/Kokoro-82M')
            self.current_lang = lang
            logger.info("Kokoro pipeline ready")

    def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        """Synthesize text to PCM16 bytes."""
        if not text or not text.strip():
            return b""

        voice = voice or self.config.KOKORO_VOICE
        speed = float(speed or self.config.KOKORO_SPEED)
        self._ensure_pipeline_for_voice(voice)

        logger.info(f"Synthesizing {len(text)} chars, voice='{voice}', speed={speed}")

        chunks: List[np.ndarray] = []
        try:
            for _, _, audio in self.pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
                audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)
                chunks.append(audio_np)
        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}")
            return b""

        if not chunks:
            logger.error("Kokoro returned no audio.")
            return b""

        audio = np.clip(np.concatenate(chunks), -1.0, 1.0)
        return (audio * 32767.0).astype(np.int16).tobytes()

    def synthesize_to_file(self, text: str, output_path: str, **kwargs) -> bool:
        audio_bytes = self.synthesize(text, **kwargs)
        if not audio_bytes:
            return False
        try:
            sf.write(output_path, np.frombuffer(audio_bytes, dtype=np.int16), SAMPLE_RATE)
            logger.info(f"Saved audio to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False

    def get_available_voices(self) -> list:
        return [
            "af_heart", "af_sarah", "af_nicole", "af_sky",
            "am_adam", "am_michael",
            "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
        ]
