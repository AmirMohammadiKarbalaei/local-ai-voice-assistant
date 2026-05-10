import os
import tempfile
from collections.abc import Callable

import speech_recognition as sr
from faster_whisper import WhisperModel

from jarvis.config import AssistantConfig


class SpeechListener:
    """Speech listener using faster-whisper for better local transcription.

    This keeps speech_recognition for microphone recording, but replaces
    recognize_google() with local Whisper transcription.

    Optional callbacks:
        on_listening_start:
            Called after ambient-noise adjustment and just before recording starts.

        on_listening_end:
            Called immediately after speech_recognition finishes recording audio.

    These callbacks are useful for Alexa-style start/end beeps.
    """

    def __init__(
        self,
        config: AssistantConfig,
        on_listening_start: Callable[[], None] | None = None,
        on_listening_end: Callable[[], None] | None = None,
    ):
        self.config = config
        self.on_listening_start = on_listening_start
        self.on_listening_end = on_listening_end

        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = float(
            os.getenv("SPEECH_PAUSE_THRESHOLD", "0.9")
        )
        self.recognizer.energy_threshold = int(
            os.getenv("SPEECH_ENERGY_THRESHOLD", "300")
        )

        self.microphone = sr.Microphone(device_index=config.mic_device_index)

        self.whisper_model_name = os.getenv("WHISPER_MODEL", "small.en")
        self.whisper_device = os.getenv("WHISPER_DEVICE", "cuda")
        self.whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

        print(
            f"Loading faster-whisper model: {self.whisper_model_name} "
            f"on {self.whisper_device} with {self.whisper_compute_type}"
        )

        self.model = WhisperModel(
            self.whisper_model_name,
            device=self.whisper_device,
            compute_type=self.whisper_compute_type,
        )

        print("Speech recognition model ready.")

    def listen(self, hotwords: list[str] | None = None) -> str | None:
        wav_path: str | None = None

        with self.microphone as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.4)

            self._safe_callback(self.on_listening_start, "listening start")

            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=float(os.getenv("SPEECH_TIMEOUT", "8")),
                    phrase_time_limit=float(
                        os.getenv("SPEECH_PHRASE_TIME_LIMIT", "12")
                    ),
                )

                self._safe_callback(self.on_listening_end, "listening end")

            except sr.WaitTimeoutError:
                return None

        try:
            wav_path = self._audio_to_temp_wav(audio)
            text = self._transcribe_wav(wav_path, hotwords=hotwords)

            if not text:
                print("Could not understand audio.")
                return None

            print(f"You: {text}")
            return text.strip()

        except Exception as e:
            print(f"Speech recognition error: {e}")
            return None

        finally:
            try:
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

    @staticmethod
    def _safe_callback(
        callback: Callable[[], None] | None,
        label: str,
    ) -> None:
        if callback is None:
            return

        try:
            callback()
        except Exception as e:
            print(f"Listening {label} callback error: {e}")

    @staticmethod
    def _audio_to_temp_wav(audio: sr.AudioData) -> str:
        wav_bytes = audio.get_wav_data()

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        with open(path, "wb") as f:
            f.write(wav_bytes)

        return path

    def _transcribe_wav(
        self,
        wav_path: str,
        hotwords: list[str] | None = None,
    ) -> str:
        hotwords = hotwords or []

        base_prompt = (
            "This is a voice command for a personal assistant called jarvis. "
            "The user may say jarvis, weather, temperature, Wikipedia, Ollama, "
            "Python, code, Newcastle, Liverpool, UK, pounds, dollars, time, date."
        )

        if hotwords:
            hotword_text = ", ".join(hotwords[:40])
            initial_prompt = (
                base_prompt
                + " Important possible names and phrases in this conversation: "
                + hotword_text
                + "."
            )
        else:
            hotword_text = None
            initial_prompt = base_prompt

        segments, info = self.model.transcribe(
            wav_path,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 700,
            },
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            hotwords=hotword_text,
        )

        text_parts = []

        for segment in segments:
            text = segment.text.strip()
            if text:
                text_parts.append(text)

        return " ".join(text_parts).strip()