import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_csv_env(name: str, default_value: str) -> list[str]:
    raw_value = os.getenv(name, default_value)

    values: list[str] = []
    for item in raw_value.split(","):
        cleaned = item.strip().strip('"').strip("'").lower().strip()
        if cleaned:
            values.append(cleaned)

    return sorted(set(values), key=len, reverse=True)


@dataclass(frozen=True)
class AssistantConfig:
    ollama_url: str = "http://localhost:11434/api/chat"

    fast_model: str = "qwen3:4b-instruct"
    smart_model: str = "qwen3:8b"

    user_timezone: str = "Europe/London"
    user_location: str = "Newcastle, UK"

    user_latitude: float = 54.9783
    user_longitude: float = -1.6178

    enable_web_search: bool = True
    tavily_api_key: str = ""

    mic_device_index: int | None = 2

    start_words: list[str] = field(default_factory=list)
    conversation_stop_words: list[str] = field(default_factory=list)
    shutdown_words: list[str] = field(default_factory=list)

    kokoro_voice: str = "bm_george"
    kokoro_speed: float = 1.08
    kokoro_lang_code: str = "b"
    kokoro_repo_id: str = "hexgrad/Kokoro-82M"
    tts_sample_rate: int = 24000

    @classmethod
    def from_env(cls) -> "AssistantConfig":
        fast_model = os.getenv(
            "FAST_MODEL",
            os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct"),
        )

        mic_index_raw = os.getenv("MIC_DEVICE_INDEX", "2").strip()
        mic_device_index = int(mic_index_raw) if mic_index_raw else None

        return cls(
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat"),
            fast_model=fast_model,
            smart_model=os.getenv("SMART_MODEL", "qwen3:8b"),
            user_timezone=os.getenv("USER_TIMEZONE", "Europe/London"),
            user_location=os.getenv("USER_LOCATION", "Newcastle, UK"),
            user_latitude=float(os.getenv("USER_LATITUDE", "54.9783")),
            user_longitude=float(os.getenv("USER_LONGITUDE", "-1.6178")),
            enable_web_search=env_bool("ENABLE_WEB_SEARCH", True),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            mic_device_index=mic_device_index,
            start_words=load_csv_env(
                "START_WORDS",
                "hey ssp,hi ssp,ok ssp,okay ssp,ssp",
            ),
            conversation_stop_words=load_csv_env(
                "CONVERSATION_STOP_WORDS",
                "bye,goodbye,stop,stop listening,that's all,that is all,thanks ssp,thank you ssp",
            ),
            shutdown_words=load_csv_env(
                "SHUTDOWN_WORDS",
                "ssp shut down,ssp shutdown,ssp sleep,ssp exit,ssp quit,ssp close",
            ),
            kokoro_voice=os.getenv("KOKORO_VOICE", "bm_george"),
            kokoro_speed=float(os.getenv("KOKORO_SPEED", "1.08")),
            kokoro_lang_code=os.getenv("KOKORO_LANG_CODE", "b"),
            kokoro_repo_id=os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M"),
        )