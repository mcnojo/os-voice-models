import logging
import numpy as np
from faster_whisper import WhisperModel
from app.config import Config
import soundfile as sf

logger = logging.getLogger(__name__)

class STTService:
    """Speech-to-text via faster-whisper (CTranslate2 backend)."""

    def __init__(self):
        self.config = Config()
        logger.info(f"Loading Faster-Whisper model: {self.config.WHISPER_MODEL}")
        self.model = WhisperModel(
            self.config.WHISPER_MODEL,
            device=self.config.WHISPER_DEVICE,
            compute_type=self.config.WHISPER_COMPUTE_TYPE
        )
        logger.info("Faster-Whisper model loaded")

    def transcribe_audio(self, audio_data: bytes, sample_rate: int = None) -> tuple[str, str]:
        """Transcribe raw PCM16 bytes to text.

        Returns (transcription, detected_language) where detected_language
        is an ISO-639-1 code (e.g. 'en', 'de').
        """
        sample_rate = sample_rate or self.config.SAMPLE_RATE
        try:
            audio_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            segments, info = self.model.transcribe(
                audio_float,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            transcription = " ".join(seg.text for seg in segments).strip()
            logger.info(f"Transcription ({info.language}): {transcription}")
            return transcription, info.language
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", "en"

    def transcribe_file(self, file_path: str) -> str:
        """Transcribe an audio file on disk."""
        try:
            audio_data, sample_rate = sf.read(file_path)
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32768).astype(np.int16)
            return self.transcribe_audio(audio_data.tobytes(), sample_rate)
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return ""
