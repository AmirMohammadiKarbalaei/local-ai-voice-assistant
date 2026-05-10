import os
import threading

import numpy as np
import sounddevice as sd


class SoundEffects:
    def __init__(self):
        self.enabled = os.getenv("SOUND_EFFECTS_ENABLED", "true").lower() in {
            "1", "true", "yes", "y", "on"
        }

        self.sample_rate = int(os.getenv("SOUND_EFFECT_SAMPLE_RATE", "24000"))
        self.volume = float(os.getenv("SOUND_EFFECT_VOLUME", "0.18"))

    def play_tone(
        self,
        frequency: float = 880.0,
        duration: float = 0.08,
        blocking: bool = True,
    ) -> None:
        if not self.enabled:
            return

        try:
            t = np.linspace(
                0,
                duration,
                int(self.sample_rate * duration),
                endpoint=False,
            )

            audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

            # Gentle fade in/out to avoid click noise.
            fade_len = min(200, len(audio) // 4)
            if fade_len > 0:
                fade = np.linspace(0, 1, fade_len).astype(np.float32)
                audio[:fade_len] *= fade
                audio[-fade_len:] *= fade[::-1]

            audio *= self.volume

            if blocking:
                sd.play(audio, self.sample_rate)
                sd.wait()
            else:
                threading.Thread(
                    target=lambda: (sd.play(audio, self.sample_rate), sd.wait()),
                    daemon=True,
                ).start()

        except Exception as e:
            print(f"Sound effect error: {e}")

    def listening_start(self) -> None:
        self.play_tone(frequency=880.0, duration=0.07, blocking=True)

    def listening_end(self) -> None:
        self.play_tone(frequency=660.0, duration=0.06, blocking=False)