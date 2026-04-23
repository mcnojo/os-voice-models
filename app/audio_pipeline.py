import logging
from typing import Dict, Any
from io import BytesIO

logger = logging.getLogger(__name__)

class AudioPipeline:
    """Wires STT -> LLM -> TTS into a single call."""

    def __init__(self, stt_service, llm_service, tts_service, sample_rate: int = 16000):
        self.stt_service = stt_service
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.sample_rate = sample_rate
        self.audio_buffer = BytesIO()

    def add_audio_chunk(self, audio_bytes: bytes):
        self.audio_buffer.write(audio_bytes)

    def clear_buffer(self):
        self.audio_buffer = BytesIO()

    def process(self) -> Dict[str, Any]:
        """Process buffered audio through the full pipeline. Returns a result dict."""
        result = {
            'success': False,
            'transcription': '',
            'response': '',
            'audio_response': None,
            'error': None
        }

        try:
            audio_data = self.audio_buffer.getvalue()
            if not audio_data:
                result['error'] = "No audio data to process"
                return result

            logger.info(f"Processing {len(audio_data)} bytes of audio")

            transcription = self.stt_service.transcribe_audio(audio_data, self.sample_rate)
            if not transcription or not transcription.strip():
                result['error'] = "No speech detected in audio"
                return result
            result['transcription'] = transcription
            logger.info(f"Transcription: {transcription}")

            response_text = self.llm_service.generate_response(transcription)
            if not response_text or not response_text.strip():
                result['error'] = "Failed to generate response"
                return result
            result['response'] = response_text
            logger.info(f"Response: {response_text}")

            audio_response = self.tts_service.synthesize(response_text)
            if not audio_response:
                result['error'] = "Failed to synthesize speech"
                return result
            result['audio_response'] = audio_response
            logger.info(f"Generated {len(audio_response)} bytes of response audio")

            result['success'] = True

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result['error'] = str(e)

        return result

    def process_text(self, text: str) -> Dict[str, Any]:
        """Skip STT — feed text directly into LLM -> TTS. Useful for testing."""
        result = {'success': False, 'response': '', 'audio_response': None, 'error': None}
        try:
            result['response'] = self.llm_service.generate_response(text)
            result['audio_response'] = self.tts_service.synthesize(result['response'])
            result['success'] = True
        except Exception as e:
            logger.error(f"Text pipeline error: {e}")
            result['error'] = str(e)
        return result
