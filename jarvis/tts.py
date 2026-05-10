import queue
import re
import threading

import numpy as np
import sounddevice as sd
import torch
from kokoro import KPipeline
import unicodedata
from jarvis.config import AssistantConfig


class KokoroTTS:
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.sample_rate = config.tts_sample_rate
        self.tts_queue: queue.Queue[str | np.ndarray | None] = queue.Queue(maxsize=6)
        self.enabled = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        torch.set_grad_enabled(False)

        print("Loading Kokoro TTS...")
        self.pipeline = KPipeline(
            lang_code=config.kokoro_lang_code,
            repo_id=config.kokoro_repo_id,
        )

        self._move_model_to_device()
        self.warm_up()

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove emojis and emoji-like decorative symbols before TTS.

        This is more reliable than only prompting the LLM.
        """
        emoji_ranges = re.compile(
            "["
            "\U0001F1E6-\U0001F1FF"  # flags
            "\U0001F300-\U0001F5FF"  # symbols and pictographs
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F680-\U0001F6FF"  # transport and map
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002700-\U000027BF"  # dingbats
            "\U00002600-\U000026FF"  # misc symbols
            "]+",
            flags=re.UNICODE,
        )

        text = emoji_ranges.sub("", text)

        # Remove emoji joiners and variation selectors.
        text = text.replace("\ufe0f", "")
        text = text.replace("\u200d", "")

        # Remove leftover standalone decorative symbol characters, but keep normal punctuation.
        cleaned_chars = []
        for char in text:
            category = unicodedata.category(char)

            if category == "So":
                continue

            cleaned_chars.append(char)

        return "".join(cleaned_chars)


    @staticmethod
    def remove_machine_timestamps(text: str) -> str:
        """Remove ISO/database-style timestamps before TTS reads them."""

        # Remove full sentences like:
        # "It will end at 2026-05-04T12:56:16."
        text = re.sub(
            r"\b(?:it|this|the timer|timer)\s+"
            r"(?:will\s+)?(?:end|ends|finish|finishes)\s+at\s+"
            r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}(?::\d{2})?"
            r"(?:[+-]\d{2}:\d{2}|Z)?\.?",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove standalone ISO timestamps:
        # 2026-05-04T12:56:16
        # 2026-05-04 12:56
        text = re.sub(
            r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}(?::\d{2})?"
            r"(?:[+-]\d{2}:\d{2}|Z)?\b",
            "",
            text,
        )

        # Remove standalone dates if they look machine-generated:
        # 2026-05-04
        text = re.sub(
            r"\b\d{4}-\d{2}-\d{2}\b",
            "",
            text,
        )

        return text
    def queue_beep(
        self,
        frequency: float = 880.0,
        duration: float = 0.08,
        volume: float = 0.18,
    ) -> None:
        """Queue a short beep through the existing TTS audio stream."""
        if not self.enabled:
            return

        sample_count = int(self.sample_rate * duration)
        t = np.linspace(0, duration, sample_count, endpoint=False)

        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Fade in/out to avoid click noise.
        fade_len = min(200, len(audio) // 4)
        if fade_len > 0:
            fade = np.linspace(0, 1, fade_len).astype(np.float32)
            audio[:fade_len] *= fade
            audio[-fade_len:] *= fade[::-1]

        audio *= volume
        self.tts_queue.put(audio)
    def _move_model_to_device(self) -> None:
        try:
            if hasattr(self.pipeline, "model"):
                self.pipeline.model.to(self.device)
                self.pipeline.model.eval()
                print(f"Kokoro model moved to: {self.device}")
        except Exception as e:
            print(f"Could not explicitly move Kokoro model to GPU: {e}")

    def warm_up(self) -> None:
        try:
            print("Warming up Kokoro TTS...")
            with torch.inference_mode():
                generator = self.pipeline(
                    "Ready.",
                    voice=self.config.kokoro_voice,
                    speed=self.config.kokoro_speed,
                )
                for _, _, _ in generator:
                    break
            print("Kokoro TTS ready.")
        except Exception as e:
            print(f"Kokoro warm-up failed: {e}")

    def clean_text_for_speech(self, text: str) -> str:
        text = text.strip()
        text = self.remove_emojis(text)
        text = self.remove_machine_timestamps(text)


        # Remove fenced code blocks. They are not pleasant to read aloud.
        text = re.sub(r"```.*?```", "I have prepared the code for you.", text, flags=re.DOTALL)

        # Convert numbered instruction lists into natural spoken transitions.
        # This avoids Kokoro reading "2." as "option two" or pausing awkwardly.
        step_words = {
            1: "First,",
            2: "Next,",
            3: "Then,",
            4: "After that,",
            5: "Finally,",
        }

        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()

            numbered_match = re.match(r"^(\d+)[.)]\s*(.*)$", stripped)
            if numbered_match:
                number = int(numbered_match.group(1))
                prefix = step_words.get(number, "Then,")
                stripped = f"{prefix} {numbered_match.group(2).strip()}"

            if stripped.startswith("- ") or stripped.startswith("• "):
                stripped = stripped[2:].strip()

            cleaned_lines.append(stripped)

        text = " ".join(cleaned_lines)

        # Remove inline markdown noise.
        text = text.replace("`", "")
        text = text.replace("*", "")
        text = text.replace("#", "")
        text = text.replace("_", " ")

        # Normalise whitespace without touching normal punctuation.
        text = " ".join(text.split()).strip()
        return text

    @staticmethod
    def audio_to_numpy(audio) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().float().cpu().numpy()
        else:
            audio = np.asarray(audio, dtype=np.float32)

        audio = np.squeeze(audio)

        if audio.ndim != 1:
            audio = audio.reshape(-1)

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def generate_audio(self, text: str):
        text = self.clean_text_for_speech(text)
        if not text:
            return

        with torch.inference_mode():
            generator = self.pipeline(
                text,
                voice=self.config.kokoro_voice,
                speed=self.config.kokoro_speed,
            )

            for _, _, audio in generator:
                yield self.audio_to_numpy(audio)

    def _worker(self) -> None:
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                latency="low",
            ) as stream:
                while True:
                    text = self.tts_queue.get()

                    if text is None:
                        self.tts_queue.task_done()
                        break

                    try:
                        if isinstance(text, np.ndarray):
                            stream.write(text.reshape(-1, 1))
                        else:
                            for audio in self.generate_audio(text):
                                stream.write(audio.reshape(-1, 1))
                    except Exception as e:
                        print(f"Kokoro TTS error: {e}")
                    finally:
                        self.tts_queue.task_done()

        except Exception as e:
            print(f"TTS audio stream error: {e}")

    def queue(self, text: str) -> None:
        if not self.enabled:
            return
        text = text.strip()
        if not text:
            return
        try:
            self.tts_queue.put(text, timeout=1.0)
        except queue.Full:
            print("TTS queue full; dropping speech.")

    def wait(self) -> None:
        self.tts_queue.join()

    def shutdown(self) -> None:
        self.tts_queue.put(None)
        self.tts_queue.join()

    @staticmethod
    def extract_speakable_chunks(buffer: str, force: bool = False) -> tuple[list[str], str]:
        """Split streamed text into natural spoken chunks."""
        chunks = []
        buffer = buffer.strip()

        if not buffer:
            return chunks, ""

        while True:
            match = re.search(r"^(.{45,}?[.!?])\s+", buffer)
            if not match:
                break
            chunk = match.group(1).strip()
            chunks.append(chunk)
            buffer = buffer[match.end():].strip()

        if force and buffer:
            chunks.append(buffer.strip())
            buffer = ""

        return chunks, buffer
