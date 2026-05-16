from abc import ABC, abstractmethod
from typing import Optional


class TTSBackend(ABC):
    """Interface every TTS engine must implement."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lang: Optional[str] = None,
    ) -> bytes:
        """Convert text to PCM-16 audio bytes."""
        ...

    @abstractmethod
    def get_available_voices(self) -> list[str]:
        ...
