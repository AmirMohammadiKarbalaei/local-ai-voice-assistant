# Jarvis Assistant Refactor

This is the same Jarvis assistant split into smaller files.

## Structure

```text
main.py
jarvis/
  __init__.py
  assistant.py       # main orchestration loop
  config.py          # .env loading and app config
  prompts.py         # system prompt
  timezones.py       # location to timezone aliases
  tools.py           # time and web-search tools
  tts.py             # Kokoro TTS queue, cleaning, streaming
  speech.py          # microphone and speech recognition
  ollama_client.py   # Ollama HTTP client
.env.example
```

## Run

Copy `.env.example` to `.env`, edit values as needed, then run:

```bash
python main.py
```

## Wake words

Use this clean format:

```env
WAKE_WORDS=hey jarvis,hi jarvis,ok jarvis,okay jarvis,jarvis,what,where,who,when,how,why,i,do,can
```

This quoted style also works because the loader strips quotes:

```env
WAKE_WORDS="hey jarvis", "hi jarvis", "ok jarvis", "jarvis"
```
