import logging
import os
from typing import List, Dict
from llama_cpp import Llama
from app.config import Config

logger = logging.getLogger(__name__)

class LLMService:
    """LLM served via llama-cpp-python (GGUF quantized, CPU)."""

    def __init__(self):
        self.model = None
        self.config = Config()
        self.conversation_history: List[Dict[str, str]] = []
        self._initialize_model()

    def _initialize_model(self):
        model_path = os.getenv(
            "GGUF_MODEL_PATH",
            "/app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )
        logger.info(f"Loading GGUF model from: {model_path}")

        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=int(os.getenv("LLM_THREADS", "10")),
            n_batch=512,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        logger.info(f"Model loaded — context size: {self.model.n_ctx()} tokens")

    def generate_response(self, user_input: str) -> str:
        """Run a chat turn: append user message, call model, return assistant text."""
        try:
            self.conversation_history.append({"role": "user", "content": user_input})

            # trim to last N turns so we don't blow the context window
            max_turns = self.config.MAX_HISTORY
            if len(self.conversation_history) > max_turns * 2:
                self.conversation_history = self.conversation_history[-(max_turns * 2):]

            messages = [{"role": "system", "content": self.config.SYSTEM_PROMPT}]
            messages.extend(self.conversation_history)

            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=self.config.LLM_MAX_TOKENS,
                temperature=self.config.LLM_TEMPERATURE,
                top_p=0.9,
                stop=["</s>", "<|im_end|>", "<|endoftext|>", "User:", "\n\n\n"],
                stream=False,
            )

            generated = response["choices"][0]["message"]["content"].strip()
            self.conversation_history.append({"role": "assistant", "content": generated})
            logger.info(f"Generated response ({len(generated)} chars): {generated[:100]}...")
            return generated

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm having trouble responding right now."

    def reset_conversation(self):
        self.conversation_history = []
        logger.info("Conversation history reset")
