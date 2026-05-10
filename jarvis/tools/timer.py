import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import sounddevice as sd


@dataclass
class TimerEntry:
    timer_id: str
    label: str
    duration_seconds: int
    warning_seconds: int | None
    started_at: float
    ends_at: float
    cancel_event: threading.Event = field(default_factory=threading.Event)


class TimerManager:
    def __init__(self, notifier=None):
        self.notifier = notifier

        self.timers: dict[str, TimerEntry] = {}
        self.lock = threading.Lock()

        self.alarm_lock = threading.Lock()
        self.alarm_stop_event = threading.Event()
        self.alarm_active = False
        self.alarm_label: str | None = None
        self.alarm_thread: threading.Thread | None = None

    @staticmethod
    def human_duration(seconds: int) -> str:
        seconds = int(seconds)

        if seconds < 60:
            return f"{seconds} second" + ("" if seconds == 1 else "s")

        minutes, sec = divmod(seconds, 60)

        if minutes < 60:
            if sec:
                return (
                    f"{minutes} minute{'s' if minutes != 1 else ''} "
                    f"and {sec} second{'s' if sec != 1 else ''}"
                )
            return f"{minutes} minute" + ("" if minutes == 1 else "s")

        hours, minutes = divmod(minutes, 60)

        if minutes:
            return (
                f"{hours} hour{'s' if hours != 1 else ''} "
                f"and {minutes} minute{'s' if minutes != 1 else ''}"
            )

        return f"{hours} hour" + ("" if hours == 1 else "s")

    @staticmethod
    def _replace_number_words(text: str) -> str:
        """Turn common spoken numbers into digits.

        Examples:
            one minute -> 1 minute
            twenty five minutes -> 25 minutes
            a minute -> 1 minute
            an hour -> 1 hour
        """
        text = text.lower().strip()

        text = re.sub(r"\ban\b", "1", text)
        text = re.sub(r"\ba\b", "1", text)

        number_words = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
        }

        tens = ["twenty", "thirty", "forty", "fifty", "sixty"]
        ones = [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]

        for ten_word in tens:
            for one_word in ones:
                phrase = f"{ten_word} {one_word}"
                value = number_words[ten_word] + number_words[one_word]
                text = re.sub(rf"\b{re.escape(phrase)}\b", str(value), text)

        for word, value in number_words.items():
            text = re.sub(rf"\b{re.escape(word)}\b", str(value), text)

        return text

    @classmethod
    def parse_duration_seconds(
        cls,
        duration_seconds: int | float | None = None,
        duration: str | None = None,
    ) -> int | None:
        if duration_seconds is not None:
            try:
                seconds = int(float(duration_seconds))
                return seconds if seconds > 0 else None
            except (TypeError, ValueError):
                return None

        if not duration:
            return None

        text = cls._replace_number_words(duration)
        fraction_patterns = {
    r"\bhalf an hour\b": 30 * 60,
    r"\bhalf hour\b": 30 * 60,
    r"\bquarter of an hour\b": 15 * 60,
    r"\ba quarter hour\b": 15 * 60,
}

        for pattern, seconds in fraction_patterns.items():
            if re.search(pattern, text):
                return seconds
        # Plain number means seconds.
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return int(float(text))

        total_seconds = 0.0

        matches = re.findall(
            r"(\d+(?:\.\d+)?)\s*"
            r"(seconds?|secs?|sec|s|minutes?|mins?|min|m|hours?|hrs?|hr|h)\b",
            text,
            flags=re.IGNORECASE,
        )

        for amount_text, unit in matches:
            amount = float(amount_text)
            unit = unit.lower()

            if unit in {"second", "seconds", "sec", "secs", "s"}:
                total_seconds += amount
            elif unit in {"minute", "minutes", "min", "mins", "m"}:
                total_seconds += amount * 60
            elif unit in {"hour", "hours", "hr", "hrs", "h"}:
                total_seconds += amount * 3600

        seconds = int(total_seconds)
        return seconds if seconds > 0 else None

    @staticmethod
    def default_warning_seconds(duration_seconds: int) -> int | None:
        """Default near-end warning.

        Simple behaviour:
            5 minutes or more -> warn with 1 minute left
            under 5 minutes -> no warning
        """
        if duration_seconds >= 5 * 60:
            return 60

        return None

    @staticmethod
    def clean_label(label: str | None) -> str:
        label = (label or "timer").strip(" ,.!?").lower()

        # Avoid silly labels like "one minute", "5 minutes", etc.
        duration_like = re.fullmatch(
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
            r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
            r"nineteen|twenty|thirty|forty|fifty|sixty)"
            r"(?:\s+\w+)?\s+(second|seconds|minute|minutes|hour|hours)",
            label,
        )

        if duration_like:
            return "timer"

        bad_labels = {
            "",
            "a",
            "an",
            "the",
            "set",
            "start",
            "create",
            "timer",
            "countdown",
        }

        if label in bad_labels:
            return "timer"

        return label

    def _timer_name(self, entry: TimerEntry) -> str:
        if entry.label.lower() == "timer":
            return "timer"
        return f"{entry.label} timer"

    def _notify(self, message: str) -> None:
        if self.notifier:
            try:
                self.notifier(message)
            except Exception as e:
                print(f"Timer notification error: {e}")

    def set_timer(
        self,
        duration_seconds: int | float | None = None,
        duration: str | None = None,
        label: str | None = None,
        warning_seconds: int | float | None = None,
    ) -> dict:
        seconds = self.parse_duration_seconds(
            duration_seconds=duration_seconds,
            duration=duration,
        )

        if not seconds:
            return {
                "ok": False,
                "error": "I could not understand the timer duration.",
            }
        MAX_TIMER_SECONDS = 24 * 60 * 60
        if seconds > MAX_TIMER_SECONDS:
            return {"ok": False, "error": "I can only set timers up to 24 hours."}
        clean_label = self.clean_label(label)

        if warning_seconds is None:
            warning = self.default_warning_seconds(seconds)
        else:
            try:
                warning = int(float(warning_seconds))
            except (TypeError, ValueError):
                warning = None

        if warning is not None and (warning <= 0 or warning >= seconds):
            warning = None

        timer_id = uuid.uuid4().hex[:8]
        started_at = time.time()
        ends_at = started_at + seconds

        entry = TimerEntry(
            timer_id=timer_id,
            label=clean_label,
            duration_seconds=seconds,
            warning_seconds=warning,
            started_at=started_at,
            ends_at=ends_at,
        )

        with self.lock:
            self.timers[timer_id] = entry

        thread = threading.Thread(
            target=self._timer_worker,
            args=(entry,),
            daemon=True,
        )
        thread.start()

        timer_name = self._timer_name(entry).capitalize()
        message = f"{timer_name} set for {self.human_duration(seconds)}."

        if warning:
            message += f" I will remind you with {self.human_duration(warning)} left."

        return {
            "ok": True,
            "timer_id": timer_id,
            "label": clean_label,
            "duration_seconds": seconds,
            "warning_seconds": warning,
            "ends_at": datetime.fromtimestamp(ends_at).isoformat(timespec="seconds"),
            "message": message,
        }

    def _timer_worker(self, entry: TimerEntry) -> None:
        try:
            if entry.warning_seconds:
                warning_at = entry.ends_at - entry.warning_seconds
                wait_until_warning = max(0, warning_at - time.time())

                if entry.cancel_event.wait(wait_until_warning):
                    return

                self._notify(
                    f"Your {self._timer_name(entry)} has "
                    f"{self.human_duration(entry.warning_seconds)} left."
                )

            wait_until_end = max(0, entry.ends_at - time.time())

            if entry.cancel_event.wait(wait_until_end):
                return

            self._start_alarm(entry)

        finally:
            with self.lock:
                self.timers.pop(entry.timer_id, None)

    def list_timers(self) -> dict:
        now = time.time()

        with self.lock:
            active = []
            expired_ids = []

            for timer_id, entry in self.timers.items():
                remaining = int(entry.ends_at - now)

                if remaining <= 0 or entry.cancel_event.is_set():
                    expired_ids.append(timer_id)
                    continue

                active.append(
                    {
                        "timer_id": entry.timer_id,
                        "label": entry.label,
                        "remaining_seconds": remaining,
                        "remaining": self.human_duration(remaining),
                        "ends_at": datetime.fromtimestamp(entry.ends_at).isoformat(
                            timespec="seconds"
                        ),
                    }
                )

            for timer_id in expired_ids:
                self.timers.pop(timer_id, None)

        return {
            "ok": True,
            "timers": active,
            "count": len(active),
        }

    def cancel_timer(
        self,
        timer_id: str | None = None,
        label: str | None = None,
    ) -> dict:
        with self.lock:
            active = list(self.timers.values())

            if not active:
                return {
                    "ok": False,
                    "error": "There are no active timers.",
                }

            target = self._find_timer(
                active=active,
                timer_id=timer_id,
                label=label,
            )

            if target is None:
                return {
                    "ok": False,
                    "error": "I could not work out which timer to cancel.",
                }

            target.cancel_event.set()
            self.timers.pop(target.timer_id, None)

        return {
            "ok": True,
            "timer_id": target.timer_id,
            "label": target.label,
            "message": f"{self._timer_name(target).capitalize()} cancelled.",
        }

    def _find_timer(
        self,
        active: list[TimerEntry],
        timer_id: str | None = None,
        label: str | None = None,
    ) -> TimerEntry | None:
        if timer_id:
            timer_id = timer_id.lower().strip()
            for entry in active:
                if entry.timer_id.lower().startswith(timer_id):
                    return entry

        if label:
            label_norm = label.lower().strip()
            for entry in active:
                if label_norm in entry.label.lower():
                    return entry

        if len(active) == 1:
            return active[0]

        return None

    def set_timer_warning(
        self,
        remaining_seconds: int | float | None = None,
        remaining: str | None = None,
        label: str | None = None,
        timer_id: str | None = None,
    ) -> dict:
        warning_seconds = self.parse_duration_seconds(
            duration_seconds=remaining_seconds,
            duration=remaining,
        )

        if not warning_seconds:
            return {
                "ok": False,
                "error": "I could not understand when you want the timer warning.",
            }

        with self.lock:
            active = list(self.timers.values())

            if not active:
                return {
                    "ok": False,
                    "error": "There are no active timers.",
                }

            target = self._find_timer(
                active=active,
                timer_id=timer_id,
                label=label,
            )

            if target is None:
                return {
                    "ok": False,
                    "error": "I could not work out which timer you mean.",
                }

            remaining_now = int(target.ends_at - time.time())

            if remaining_now <= 0:
                return {
                    "ok": False,
                    "error": "That timer has already finished.",
                }

            if warning_seconds >= remaining_now:
                return {
                    "ok": False,
                    "error": (
                        f"That timer only has {self.human_duration(remaining_now)} left, "
                        f"so I cannot warn you at {self.human_duration(warning_seconds)} left."
                    ),
                }

            timer_id_to_warn = target.timer_id
            timer_name = self._timer_name(target)
            target_label = target.label

        thread = threading.Thread(
            target=self._extra_warning_worker,
            args=(timer_id_to_warn, warning_seconds),
            daemon=True,
        )
        thread.start()

        return {
            "ok": True,
            "timer_id": timer_id_to_warn,
            "label": target_label,
            "remaining_seconds": warning_seconds,
            "message": (
                f"Okay. I will remind you when your {timer_name} has "
                f"{self.human_duration(warning_seconds)} left."
            ),
        }

    def _extra_warning_worker(self, timer_id: str, warning_seconds: int) -> None:
        while True:
            with self.lock:
                entry = self.timers.get(timer_id)

                if entry is None:
                    return

                wait_seconds = entry.ends_at - warning_seconds - time.time()
                cancel_event = entry.cancel_event

            if wait_seconds <= 0:
                break

            if cancel_event.wait(min(wait_seconds, 1.0)):
                return

        with self.lock:
            entry = self.timers.get(timer_id)

        if entry is None or entry.cancel_event.is_set():
            return

        self._notify(
            f"Your {self._timer_name(entry)} has "
            f"{self.human_duration(warning_seconds)} left."
        )

    def is_alarm_active(self) -> bool:
        with self.alarm_lock:
            return self.alarm_active

    def stop_alarm(self) -> dict:
        with self.alarm_lock:
            if not self.alarm_active:
                return {
                    "ok": False,
                    "message": "There is no alarm ringing.",
                }

            label = self.alarm_label or "timer"
            self.alarm_stop_event.set()
            self.alarm_active = False
            self.alarm_label = None

        return {
            "ok": True,
            "message": f"Okay. I stopped the {label} alarm.",
        }

    def _start_alarm(self, entry: TimerEntry) -> None:
        timer_name = self._timer_name(entry)

        with self.alarm_lock:
            # Stop any existing alarm first.
            self.alarm_stop_event.set()

            self.alarm_stop_event = threading.Event()
            self.alarm_active = True
            self.alarm_label = timer_name

            self.alarm_thread = threading.Thread(
                target=self._alarm_worker,
                args=(timer_name, self.alarm_stop_event),
                daemon=True,
            )
            self.alarm_thread.start()

    def _alarm_worker(self, timer_name: str, stop_event: threading.Event) -> None:
        self._notify(f"Your {timer_name} is done. Say Jarvis stop to silence the alarm.")

        # Give TTS a moment before the ringtone starts.
        if stop_event.wait(1.4):
            return

        try:
            while not stop_event.is_set():
                self._play_alarm_pattern(stop_event)
                stop_event.wait(0.35)

        finally:
            with self.alarm_lock:
                if self.alarm_stop_event is stop_event:
                    self.alarm_active = False
                    self.alarm_label = None

    def _play_alarm_pattern(self, stop_event: threading.Event) -> None:
        """Play a short repeating two-tone ringtone.

        This runs outside TTS, so it does not block Jarvis talking/listening.
        """
        if stop_event.is_set():
            return

        sample_rate = 24000
        volume = 0.22

        def tone(frequency: float, duration: float) -> np.ndarray:
            t = np.linspace(
                0,
                duration,
                int(sample_rate * duration),
                endpoint=False,
            )
            audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

            # Soft fade to avoid clicks.
            fade_len = min(200, len(audio) // 4)
            if fade_len > 0:
                fade = np.linspace(0, 1, fade_len).astype(np.float32)
                audio[:fade_len] *= fade
                audio[-fade_len:] *= fade[::-1]

            return audio

        silence_short = np.zeros(int(sample_rate * 0.08), dtype=np.float32)
        silence_long = np.zeros(int(sample_rate * 0.18), dtype=np.float32)

        audio = np.concatenate(
            [
                tone(880, 0.18),
                silence_short,
                tone(660, 0.18),
                silence_long,
            ]
        )

        audio *= volume

        try:
            sd.play(audio, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Timer alarm sound error: {e}")
            stop_event.wait(1.0)


def schema_set_timer() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": (
                "Set a non-blocking countdown timer. Use this when the user asks "
                "to set, start, or create a timer. The assistant can keep talking "
                "while the timer runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_seconds": {
                        "type": "number",
                        "description": "Timer duration in seconds. For example, 300 for five minutes.",
                    },
                    "duration": {
                        "type": "string",
                        "description": "Natural duration text, for example '5 minutes', '90 seconds', or '1 hour 30 minutes'.",
                    },
                    "label": {
                        "type": "string",
                        "description": (
                            "Optional timer label, for example pasta, tea, laundry, or workout. "
                            "This is what the timer is for, not the duration."
                        ),
                    },
                    "warning_seconds": {
                        "type": "number",
                        "description": "Optional reminder before the timer ends, in seconds.",
                    },
                },
                "required": [],
            },
        },
    }


def schema_list_timers() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "list_timers",
            "description": "List active timers and how long is left.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }


def schema_cancel_timer() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "cancel_timer",
            "description": "Cancel an active timer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timer_id": {
                        "type": "string",
                        "description": "Optional timer id.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional timer label, for example pasta, tea, laundry, or workout.",
                    },
                },
                "required": [],
            },
        },
    }


def schema_set_timer_warning() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "set_timer_warning",
            "description": (
                "Add an extra warning to an active timer. Use this when the user asks "
                "to be reminded when a timer has a certain amount of time left."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "remaining_seconds": {
                        "type": "number",
                        "description": "How much time should be left when Jarvis warns the user. For example, 480 for eight minutes left.",
                    },
                    "remaining": {
                        "type": "string",
                        "description": "Natural time-left text, for example '8 minutes', '30 seconds', or '1 minute'.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional timer label, for example laundry, pasta, tea, or workout.",
                    },
                    "timer_id": {
                        "type": "string",
                        "description": "Optional timer id.",
                    },
                },
                "required": [],
            },
        },
    }


def schema_stop_timer_alarm() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "stop_timer_alarm",
            "description": "Stop the currently ringing timer alarm.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }