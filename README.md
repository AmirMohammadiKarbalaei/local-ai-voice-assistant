# Jarvis: Local AI Voice Assistant with Tool Routing

Jarvis is a local-first AI voice assistant built in Python. It listens through the microphone, transcribes speech locally with `faster-whisper`, routes user requests through deterministic logic or an LLM-based tool router, executes Python tools, and responds using local Kokoro text-to-speech.

The goal of this project is to explore what a practical local AI assistant needs beyond a normal chatbot: voice interaction, wake-word handling, low-latency responses, reliable tool use, local inference, and spoken output that feels natural in real use.

## What it can do

Jarvis currently supports:

- Wake-word based activation
- Conversation mode after activation
- Stop-listening and shutdown commands
- Local speech-to-text using `faster-whisper`
- Local LLM inference through Ollama
- Streaming LLM responses
- Local Kokoro text-to-speech output
- Short listening start/end beeps
- Deterministic fast paths for common requests
- LLM-based tool routing for more complex commands
- Weather lookup
- Current time and date lookup
- Web search
- Wikipedia summaries
- Calculator tool
- Currency conversion
- Timers, active timer listing, timer cancellation, timer warnings, and alarm stopping

## Architecture

The assistant follows this high-level flow:

```text
Microphone
  ↓
SpeechListener
  ↓
faster-whisper local transcription
  ↓
Wake-word / conversation-state handling
  ↓
Request handling
  ├── deterministic fast paths
  └── LLM-based ToolRouter
        ↓
      ToolRegistry
        ↓
      Python tools
        ↓
Response generation with Ollama
  ↓
Kokoro TTS
  ↓
Spoken audio output
```

The core orchestration is handled by `JarvisAssistant`. It manages configuration, the speech listener, Ollama client, tool registry, tool router, conversation state, response streaming, and TTS playback.

## Key design ideas

### Local-first voice pipeline

Jarvis uses `speech_recognition` for microphone capture, then transcribes the recorded audio locally with `faster-whisper`. This avoids relying on cloud speech recognition for the core voice input pipeline.

### Ollama-based local reasoning

The assistant talks to Ollama through a small HTTP client. It supports both one-shot chat calls and streamed chat responses.

The default model setup is:

```env
FAST_MODEL=qwen3:4b-instruct
SMART_MODEL=qwen3:8b
```

The fast model is used for most voice interactions to keep latency low. The smart model can be used for more complex requests.

### Hybrid request handling

Jarvis does not send every request directly to the LLM.

It first checks for deterministic/direct tool responses, such as simple weather, time, or Wikipedia requests. This keeps common voice interactions faster and more reliable.

For broader or ambiguous requests, Jarvis uses an LLM-based `ToolRouter`.

### LLM tool router

The `ToolRouter` asks the local model to return strict JSON describing:

```json
{
  "use_tool": true,
  "tool_name": "get_weather",
  "arguments": {
    "location": "Liverpool"
  },
  "confidence": 0.92,
  "reason": "User asked for current weather."
}
```

The router also repairs common speech-to-text mistakes and argument issues. For example:

- `"what timer is it"` may mean `"what time is it"`
- `"weather in little pool"` may mean `"weather in Liverpool"`
- `"set a pasta time for ten minutes"` may mean `"set a pasta timer for ten minutes"`
- `"search restaurants round me"` is converted into a location-aware web search query

### Modular tools

Tools are registered through `ToolRegistry`. The current tool set includes:

```text
get_current_time
get_weather
web_search
wiki_summary
calculate
convert_currency
set_timer
list_timers
cancel_timer
stop_timer_alarm
set_timer_warning
```

This makes it easier to add new assistant capabilities without rewriting the core conversation loop.

### Spoken-response optimization

Jarvis is designed for speech, not just text.

The system prompt keeps answers short and natural. The TTS layer also cleans text before speaking by removing emojis, markdown, code blocks, machine timestamps, and awkward numbered-list formatting.

Streaming responses are split into speakable chunks so Jarvis can start talking before the full LLM response is finished.

## Project structure

```text
main.py
jarvis/
  __init__.py

  assistant.py
    Main assistant orchestration loop.
    Handles wake words, conversation state, direct tools, tool routing,
    Ollama streaming, and TTS playback.

  config.py
    Loads environment variables and central assistant settings.
    Includes model names, location, timezone, microphone device,
    wake words, stop words, shutdown words, web search settings,
    and Kokoro TTS configuration.

  prompts.py
    System prompt for Jarvis.
    Optimised for concise spoken responses.

  speech.py
    Microphone recording and local speech recognition.
    Uses speech_recognition for audio capture and faster-whisper
    for local transcription.

  tts.py
    Kokoro TTS wrapper.
    Handles GPU setup, text cleaning, audio queueing,
    beep sounds, and streamed speech playback.

  ollama_client.py
    HTTP client for Ollama.
    Supports normal chat calls and streaming chat responses.

  tool_router.py
    LLM-based tool routing layer.
    Decides whether a tool is needed, chooses the tool,
    validates JSON, repairs arguments, and filters unsupported inputs.

  timezones.py
    Maps common place names to IANA time zones.

  sounds.py
    Simple sound effect helpers for listening feedback.

  tools/
    __init__.py

    registry.py
      Central registry for all available tools and tool schemas.

    current_time.py
      Current date/time lookup using timezone-aware datetime.

    weather.py
      Weather lookup using Open-Meteo and geocoding.

    web_search.py
      Web search using Tavily when configured, with DuckDuckGo fallback.

    wiki_summary.py
      Wikipedia summary lookup.

    timer.py
      Non-blocking threaded timer system with labels, warnings,
      cancellation, active timer listing, and alarm playback.

    calculate.py
      Calculator tool.

    currency.py
      Currency conversion tool.
```

## Requirements

This project is designed for a local Python environment with Ollama installed.

Core dependencies include:

```text
python-dotenv
requests
numpy
sounddevice
torch
kokoro
speechrecognition
faster-whisper
```

Depending on your machine, you may also need system audio dependencies for microphone input and speaker output.

## Ollama setup

Install Ollama and pull the models you want to use.

Example:

```bash
ollama pull qwen3:4b-instruct
ollama pull qwen3:8b
```

Make sure Ollama is running locally before starting Jarvis.

Default Ollama endpoint:

```env
OLLAMA_URL=http://localhost:11434/api/chat
```

## Environment configuration

Create a `.env` file in the project root.

Example:

```env
# Ollama
OLLAMA_URL=http://localhost:11434/api/chat
FAST_MODEL=qwen3:4b-instruct
SMART_MODEL=qwen3:8b

# User location
USER_TIMEZONE=Europe/London
USER_LOCATION=Newcastle, UK
USER_LATITUDE=54.9783
USER_LONGITUDE=-1.6178

# Microphone
MIC_DEVICE_INDEX=2
SPEECH_TIMEOUT=8
SPEECH_PHRASE_TIME_LIMIT=12
SPEECH_PAUSE_THRESHOLD=0.9
SPEECH_ENERGY_THRESHOLD=300

# faster-whisper
WHISPER_MODEL=small.en
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

# Tool router
ENABLE_LLM_TOOL_ROUTER=true
ROUTER_CONFIDENCE_THRESHOLD=0.65
ROUTER_CONFIRM_THRESHOLD=0.80

# Web search
ENABLE_WEB_SEARCH=true
TAVILY_API_KEY=

# Wake / stop / shutdown words
START_WORDS=hey jarvis,hi jarvis,ok jarvis,okay jarvis,jarvis
CONVERSATION_STOP_WORDS=bye,goodbye,stop,stop listening,that's all,that is all,thanks jarvis,thank you jarvis
SHUTDOWN_WORDS=jarvis shut down,jarvis shutdown,jarvis sleep,jarvis exit,jarvis quit,jarvis close

# Kokoro TTS
KOKORO_VOICE=bm_george
KOKORO_SPEED=1.08
KOKORO_LANG_CODE=b
KOKORO_REPO_ID=hexgrad/Kokoro-82M
```

The wake-word loader accepts comma-separated values and strips quotes, so this also works:

```env
START_WORDS="hey jarvis", "hi jarvis", "ok jarvis", "jarvis"
```

## Running the assistant

Start Ollama first, then run:

```bash
python main.py
```

Jarvis will start in idle mode and wait for a wake word.

Example:

```text
Jarvis is online.
Say "Jarvis" when you need it.
```

## Example voice interactions

```text
You: Jarvis, what time is it?
Jarvis: It is 10:42 AM in Newcastle, UK.
```

```text
You: Jarvis, what is the weather in Liverpool?
Jarvis: It is currently 14 degrees Celsius in Liverpool, and it feels like 11 degrees. Today's range is 10 to 15 degrees, with a low chance of rain.
```

```text
You: Jarvis, set a pasta timer for ten minutes.
Jarvis: Pasta timer set for 10 minutes. I will remind you with 1 minute left.
```

```text
You: How long is left?
Jarvis: Pasta timer has 8 minutes and 20 seconds left.
```

```text
You: Stop the alarm.
Jarvis: Alarm stopped.
```

```text
You: Jarvis, search restaurants near me.
Jarvis: Here are a few nearby options...
```

## Timer system

The timer system is non-blocking, so Jarvis can keep answering while timers run.

Timer features include:

- Spoken duration parsing
- Labels such as pasta, tea, laundry, workout, or break
- Default warning for longer timers
- Extra warning support
- Active timer listing
- Cancellation by timer ID or label
- Alarm playback
- Voice-controlled alarm stopping

## Web search

Jarvis can use Tavily if a `TAVILY_API_KEY` is provided.

If no Tavily key is available, it falls back to DuckDuckGo Instant Answer. The fallback is useful, but less reliable than a full search API.

## Notes on local performance

This project is designed around local inference, so performance depends heavily on your hardware.

Recommended setup:

- NVIDIA GPU for faster-whisper and Kokoro TTS
- Ollama running locally
- A small fast model for voice UX
- A larger smart model only when deeper reasoning is needed

The default configuration is tuned for low-latency voice interaction rather than long-form chatbot output.

## Current limitations

- The assistant is currently optimized for English speech.
- Tool routing depends on the local model returning valid JSON, although the router includes JSON extraction and repair logic.
- Web search quality depends on whether Tavily is configured.
- Microphone device index may need to be changed for your machine.
- Some text responses may still need further cleaning for perfect speech output.
- The current implementation is a local prototype rather than a packaged desktop or mobile app.

## Why this project matters

This project explores the product and engineering challenges behind useful AI assistants:

- How to reduce latency in voice interaction
- How to decide when an LLM should use a tool
- How to recover from imperfect speech transcription
- How to keep assistant responses short and spoken-friendly
- How to combine local models, real-time APIs, and deterministic logic
- How to build a modular assistant that can grow over time

Jarvis is not just a chatbot. It is a local AI assistant architecture that connects voice input, local reasoning, tool execution, and spoken output into one working system.
