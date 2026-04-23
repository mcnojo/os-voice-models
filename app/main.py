import logging
import base64
import json
import time
from flask import Flask, render_template
from flask_cors import CORS
from flask_sock import Sock

from app.config import Config
from app.stt_service import STTService
from app.llm_service import LLMService
from app.tts_service import TTSService
from app.audio_pipeline import AudioPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../static', template_folder='../templates')
CORS(app)
sock = Sock(app)

config = Config()

logger.info("Initializing services...")
stt_service = STTService()
llm_service = LLMService()
tts_service = TTSService()
logger.info("All services initialized")

sessions = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return {'status': 'healthy', 'services': 'ready'}, 200


@sock.route('/voice')
def voice_socket(ws):
    """WebSocket endpoint — streams audio in, returns transcription + LLM response + TTS audio."""
    session_id = f"session_{int(time.time() * 1000)}"
    logger.info(f"New voice session: {session_id}")

    pipeline = AudioPipeline(
        stt_service=stt_service,
        llm_service=llm_service,
        tts_service=tts_service,
        sample_rate=config.SAMPLE_RATE
    )
    sessions[session_id] = {'pipeline': pipeline, 'created_at': time.time()}

    try:
        while True:
            message = ws.receive()
            if message is None:
                break

            try:
                data = json.loads(message)
                msg_type = data.get('type')

                if msg_type == 'audio':
                    audio_bytes = base64.b64decode(data.get('audio', ''))
                    pipeline.add_audio_chunk(audio_bytes)

                elif msg_type == 'audio_end':
                    logger.info("Processing complete audio input")
                    result = pipeline.process()

                    if result['success']:
                        ws.send(json.dumps({'type': 'transcription', 'text': result.get('transcription', '')}))
                        ws.send(json.dumps({'type': 'response', 'text': result.get('response', '')}))
                        if result.get('audio_response'):
                            ws.send(json.dumps({
                                'type': 'audio_response',
                                'audio': base64.b64encode(result['audio_response']).decode('utf-8'),
                                'sample_rate': config.SAMPLE_RATE
                            }))
                    else:
                        ws.send(json.dumps({'type': 'error', 'message': result.get('error', 'Processing error')}))

                    pipeline.clear_buffer()

                elif msg_type == 'reset':
                    llm_service.reset_conversation()
                    pipeline.clear_buffer()
                    ws.send(json.dumps({'type': 'reset_complete', 'message': 'Conversation reset'}))

                elif msg_type == 'ping':
                    ws.send(json.dumps({'type': 'pong'}))

            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                ws.send(json.dumps({'type': 'error', 'message': 'Invalid message format'}))
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                ws.send(json.dumps({'type': 'error', 'message': str(e)}))

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        sessions.pop(session_id, None)
        logger.info(f"Voice session ended: {session_id}")


if __name__ == '__main__':
    logger.info(f"Starting voice assistant on {config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False, threaded=True)
