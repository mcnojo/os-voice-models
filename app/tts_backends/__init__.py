from app.tts_backends.base import TTSBackend


def create_tts_backend(engine: str) -> TTSBackend:
    """Factory: instantiate the TTS backend selected by engine name."""
    engine = engine.strip().lower()

    if engine == "kokoro":
        from app.tts_backends.kokoro import KokoroBackend
        return KokoroBackend()

    if engine == "omnivoice":
        from app.tts_backends.omnivoice import OmniVoiceBackend
        return OmniVoiceBackend()

    if engine == "supertonic":
        from app.tts_backends.supertonic import SupertonicBackend
        return SupertonicBackend()

    raise ValueError(
        f"Unknown TTS engine '{engine}'. "
        f"Supported: kokoro, omnivoice, supertonic"
    )
