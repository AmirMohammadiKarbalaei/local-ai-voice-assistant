import json
import re
from jarvis.tool_router import ToolRouter
import requests
import os
from jarvis.config import AssistantConfig
from jarvis.ollama_client import OllamaClient
from jarvis.prompts import SYSTEM_PROMPT
from jarvis.speech import SpeechListener
from jarvis.tools import ToolRegistry
from jarvis.tts import KokoroTTS


class JarvisAssistant:
    """Local voice assistant using Ollama + Kokoro TTS.

    Conversation behaviour:

    Idle mode:
        Requires a start word, such as "jarvis" or "Hey jarvis".

    Conversation mode:
        User can speak normally without saying "jarvis" every time.

    Stop conversation:
        User says "bye", "stop", "stop listening", etc.

    Shutdown program:
        User says "jarvis shut down", "jarvis sleep", etc.
    """

    def __init__(self):
        self.config = AssistantConfig.from_env()

        self.fast_model = self.config.fast_model
        self.smart_model = self.config.smart_model
        self.model = self.fast_model

        self.tts = KokoroTTS(self.config)
        self.listener = SpeechListener(
            self.config,
            on_listening_start=lambda: self.tts.queue_beep(
                frequency=880.0,
                duration=0.07,
                volume=0.18,
            ),
            on_listening_end=lambda: self.tts.queue_beep(
                frequency=660.0,
                duration=0.06,
                volume=0.14,
            ),
        )
        self.ollama = OllamaClient(self.config.ollama_url)
        self.tools = ToolRegistry(
            self.config,
            notifier=self.speak,
        )

        self.in_conversation = False
        self.tool_router = ToolRouter(
            ollama=self.ollama,
            config=self.config,
            tools=self.tools,
        )
        self.enable_llm_tool_router = os.getenv("ENABLE_LLM_TOOL_ROUTER", "true").lower() in {
            "1", "true", "yes", "y", "on"
        }
        self.router_confidence_threshold = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.65"))
        self.router_confirm_threshold = float(os.getenv("ROUTER_CONFIRM_THRESHOLD", "0.80"))
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        print(f"Start words loaded: {self.config.start_words}")
        print(f"Conversation stop words loaded: {self.config.conversation_stop_words}")
        print(f"Shutdown words loaded: {self.config.shutdown_words}")
        print(f"Fast model: {self.fast_model}")
        print(f"Smart model: {self.smart_model}")
        print(f"User location: {self.config.user_location}")
        print(f"User timezone: {self.config.user_timezone}")
    @staticmethod
    def log_direct_tool_call(tool_name: str, arguments: dict, result: dict) -> None:
        print("\n" + "=" * 60)
        print("DIRECT PYTHON TOOL ROUTE")
        print(f"Tool name: {tool_name}")
        print(f"Arguments: {json.dumps(arguments, ensure_ascii=False, indent=2)}")
        print("Tool result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 60 + "\n")
    # -------------------------
    # Speaking
    # -------------------------
    @staticmethod
    def is_weather_query(user_message: str) -> bool:
        text = user_message.lower()

        weather_patterns = [
            r"\bweather\b",
            r"\bforecast\b",
            r"\btemperature\b",
            r"\bis it raining\b",
            r"\bwill it rain\b",
            r"\bdo i need an umbrella\b",
            r"\bhow cold is it\b",
            r"\bhow hot is it\b",
        ]
        return any(re.search(p, text) for p in weather_patterns)
    def is_alarm_stop_command(self, text: str) -> bool:
        text_norm = self.normalise_command_text(text)

        stop_phrases = [
            "stop",
            "stop it",
            "stop alarm",
            "stop the alarm",
            "silence alarm",
            "silence the alarm",
            "turn off alarm",
            "turn off the alarm",
            "cancel alarm",
            "cancel the alarm",
            "jarvis stop",
            "jarvis stop it",
            "jarvis stop alarm",
            "jarvis stop the alarm",
        ]

        return any(phrase in text_norm for phrase in stop_phrases)
    def should_consider_tool_router(self, user_message: str) -> bool:
        """Cheap gate before spending an LLM call on tool routing.

        Direct tools run first. This catches messy STT or broader tool requests.
        """
        if not self.enable_llm_tool_router:
            return False

        text = self.normalise_command_text(user_message)

        if not text:
            return False

        simple_chat = {
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "good morning",
            "good afternoon",
            "good evening",
        }

        if text in simple_chat:
            return False

        tool_keywords = [
            "time",
            "timer",
            "date",
            "today",
            "weather",
            "temperature",
            "forecast",
            "rain",
            "wind",
            "umbrella",
            "search",
            "look up",
            "find",
            "near me",
            "nearby",
            "around me",
            "online",
            "latest",
            "current",
            "news",
            "price",
            "stock",
            "score",
            "wikipedia",
            "wiki",
            "calculate",
            "percent",
            "percentage",
            "convert",
            "currency",
            "exchange rate",
            "pounds",
            "dollars",
            "euros",
            "yuan",
            "timer",
            "timers",
            "countdown",
            "remind me in",
            "set a timer",
            "start a timer",
            "cancel timer",
            "stop timer",
            "how long left",
        ]

        if any(keyword in text for keyword in tool_keywords):
            return True

        question_or_action_starters = (
            "what ",
            "where ",
            "who ",
            "when ",
            "how ",
            "can ",
            "could ",
            "do ",
            "does ",
            "is ",
            "are ",
            "will ",
            "should ",
            "please ",
            "show ",
            "tell me ",
        )

        return text.startswith(question_or_action_starters)
    def ask_with_generic_tool_router(self, user_message: str, model: str) -> str | None:
        if not self.should_consider_tool_router(user_message):
            return None

        # Use fast model for routing even if the final answer uses smart model.
        route = self.tool_router.route(user_message=user_message, model=self.fast_model)

        print("\n" + "-" * 60)
        print("GENERIC TOOL ROUTER DECISION")
        print(json.dumps(route, ensure_ascii=False, indent=2))
        print("-" * 60 + "\n")

        if not route.get("use_tool"):
            return None

        confidence = float(route.get("confidence", 0.0) or 0.0)

        if confidence < self.router_confidence_threshold:
            print(f"Router confidence too low: {confidence}")
            return None

        tool_name = route["tool_name"]
        arguments = route.get("arguments") or {}

        safe_information_tools = {
            "get_current_time",
            "get_weather",
            "web_search",
            "wiki_summary",
            "calculate",
            "convert_currency",
        }

        if tool_name not in safe_information_tools and confidence < self.router_confirm_threshold:
            reply = "I think that needs an action, but I am not fully sure. Could you repeat that?"
            self.speak(reply)
            return reply

        try:
            result = self.tools.available_tools[tool_name](**arguments)

            self.log_direct_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
            )

            simple_reply = self.format_routed_tool_response(
                user_message=user_message,
                tool_name=tool_name,
                tool_result=result,
            )

            if simple_reply:
                print(f"jarvis: {simple_reply}")
                self.tts.queue(simple_reply)
                return simple_reply

            return self.generate_final_answer_from_tool(
                user_message=user_message,
                tool_name=tool_name,
                tool_result=result,
                model=model,
            )

        except Exception as e:
            error = f"I tried to use {tool_name}, but something went wrong: {e}"
            self.speak(error)
            return error
    def format_routed_tool_response(
        self,
        user_message: str,
        tool_name: str,
        tool_result: dict,
    ) -> str | None:
        """Fast spoken replies for simple tool results.

        This avoids a second LLM call after obvious tools.
        """
        if not tool_result.get("ok"):
            return None

        lowered = user_message.lower()

        if tool_name == "get_current_time":
            if "date" in lowered or "day" in lowered:
                return f"It is {tool_result['date']} in {tool_result['location']}."
            return f"It is {tool_result['time_12h']} in {tool_result['location']}."

        if tool_name == "get_weather":
            return self.format_weather_response(tool_result)

        if tool_name == "wiki_summary":
            return self.format_wikipedia_response(tool_result)

        if tool_name == "calculate":
            return f"The answer is {tool_result['result']}."

        if tool_name == "convert_currency":
            return (
                f"{tool_result['amount']} {tool_result['from_currency']} is about "
                f"{tool_result['converted']} {tool_result['to_currency']}."
            )

        return None

        if tool_name in {"set_timer", "cancel_timer"}:
            return tool_result.get("message")

        if tool_name == "list_timers":
            timers = tool_result.get("timers") or []
            if not timers:
                return "You do not have any active timers."
            spoken = [f"{timer['label']} has {timer['remaining']} left" for timer in timers[:3]]
            return ". ".join(spoken) + "."

        return None
    def generate_final_answer_from_tool(
        self,
        user_message: str,
        tool_name: str,
        tool_result: dict,
        model: str,
    ) -> str:
        """Generate a final spoken answer from a Python tool result.

        Important:
        Do not pass tool results as role='tool' here unless this is part of a real
        model tool-call chain. For this generic router path, pass the tool result
        as structured context inside a normal user message.
        """

        if tool_name == "web_search":
            system_prompt = (
                "You are jarvis, a concise local voice assistant. "
                "The user asked for online information. Your job is to answer the user's exact question using the search result. "
                "Do not say the user shared or provided content. The content is search data from your own web_search tool. "
                "Do not summarize the search result as an article. "
                "Do not mention URLs unless the user asks. "
                "Do not use emojis, markdown, headings, bullet points, numbered lists, or decorative symbols. "
                "Use natural spoken English. "
                "For 'what can I do today' or recommendations, give three to five specific options with a short reason for each. "
                "Prefer events or activities that look current, dated, or suitable for today. "
                "If the results are weak or uncertain, say so briefly."
            )

            context = self.build_web_search_answer_context(tool_result)

            user_content = (
                f"User question: {user_message}\n\n"
                f"Search result from web_search tool:\n{context}\n\n"
                "Answer the user directly. Start with a useful recommendation, not a summary of the search data."
            )

            # Optional. Use smart model for web synthesis only if you can tolerate slower replies.
            # model = self.smart_model

        else:
            system_prompt = (
                "You are jarvis, a concise local voice assistant. "
                "Answer the user's exact question using only the tool result. "
                "Do not say the user shared or provided content. "
                "Do not use emojis, markdown, headings, bullet points, numbered lists, or decorative symbols. "
                "Use one to three short spoken sentences. "
                "If the tool failed or has no useful data, say that briefly. "
                "Do not invent missing information."
            )

            user_content = (
                f"User question: {user_message}\n\n"
                f"Tool name: {tool_name}\n"
                f"Tool result:\n{json.dumps(tool_result, ensure_ascii=False, indent=2)}\n\n"
                "Answer the user directly."
            )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        reply = self.ollama_chat_stream(messages=messages, model=model)
        return reply.strip()
    
    def build_web_search_answer_context(self, tool_result: dict) -> str:
        """Turn a web_search result into clean context for the final answer model."""

        if not tool_result.get("ok"):
            return json.dumps(tool_result, ensure_ascii=False, indent=2)

        lines = []

        query = tool_result.get("query")
        if query:
            lines.append(f"Search query: {query}")

        answer = tool_result.get("answer")
        if answer:
            lines.append(f"Search engine short answer: {answer}")

        # Prefer your prebuilt context field if your updated web_search.py returns it.
        context = tool_result.get("context")
        if context:
            lines.append("Search context:")
            lines.append(str(context))
            return "\n".join(lines)

        results = tool_result.get("results") or []
        for index, item in enumerate(results[:5], start=1):
            title = item.get("title") or "Untitled source"
            content = item.get("content") or ""

            # Keep the final LLM context readable and not too huge.
            content = " ".join(content.split())
            if len(content) > 900:
                content = content[:900].rsplit(" ", 1)[0] + "."

            lines.append(
                f"Source {index}: {title}\n"
                f"Content: {content}"
            )

        return "\n\n".join(lines)
    def process_text_input(self, text: str) -> dict:
        """Process one text command without using the laptop microphone.

        Used by the phone/web UI.
        """
        text = (text or "").strip()

        if not text:
            return {
                "ok": False,
                "reply": "",
                "state": "empty",
                "ignored": True,
            }

        if self.is_shutdown_command(text):
            self.in_conversation = False
            return {
                "ok": True,
                "reply": "Going to sleep. Tap the core when you need me again.",
                "state": "sleeping",
                "ignored": False,
            }

        if not self.in_conversation:
            if not self.is_start_command(text):
                return {
                    "ok": True,
                    "reply": "",
                    "state": "waiting_for_wake_word",
                    "ignored": True,
                }

            self.in_conversation = True
            command = self.remove_start_word(text)

            if not command:
                return {
                    "ok": True,
                    "reply": "Yes?",
                    "state": "conversation",
                    "ignored": False,
                }

        else:
            command = text.strip()

        if self.is_conversation_stop_command(command):
            self.in_conversation = False
            return {
                "ok": True,
                "reply": "Okay. Say jarvis when you need me.",
                "state": "waiting_for_wake_word",
                "ignored": False,
            }

        reply = self.ask_ollama_streaming(command)
        self.tts.wait()

        return {
            "ok": True,
            "reply": reply or "",
            "state": "conversation" if self.in_conversation else "waiting_for_wake_word",
            "ignored": False,
        }
    def extract_weather_location(self, user_message: str) -> str | None:
        """Extract a simple weather location.

        Examples:
            "what is the temperature in Liverpool UK" -> "Liverpool UK"
            "weather for London" -> "London"
            "I'm in Liverpool UK" -> "Liverpool UK"
        """
        text = user_message.strip()
        normalised = self.normalise_command_text(text)

        patterns = [
            r"\bi am in\s+(.+)$",
            r"\bi m in\s+(.+)$",
            r"\bim in\s+(.+)$",
            r"\bin\s+(.+)$",
            r"\bfor\s+(.+)$",
            r"\bnear\s+(.+)$",
            r"\bat\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalised)
            if match:
                location = match.group(1).strip()

                # Remove filler words from speech recognition errors.
                location = re.sub(r"^(play|please|now)\s+", "", location).strip()

                if location:
                    return location

        return None


    @staticmethod
    def format_weather_response(result: dict) -> str:
        if not result.get("ok"):
            return "I could not get the weather right now."

        location = result.get("location", "your location")
        temp = result.get("temperature_c")
        feels_like = result.get("feels_like_c")
        max_temp = result.get("max_temp_c")
        min_temp = result.get("min_temp_c")
        rain_chance = result.get("precipitation_probability_percent")
        wind = result.get("wind_speed_kmh")

        parts = []

        if temp is not None:
            parts.append(f"It is currently {temp} degrees Celsius in {location}")

        if feels_like is not None:
            parts.append(f"and it feels like {feels_like} degrees")

        sentence = " ".join(parts).strip()
        if sentence:
            sentence += "."

        extra = []

        if min_temp is not None and max_temp is not None:
            extra.append(f"Today's range is {min_temp} to {max_temp} degrees")

        if rain_chance is not None:
            extra.append(f"with a {rain_chance} percent chance of rain")

        if wind is not None:
            extra.append(f"and wind around {wind} kilometres per hour")

        if extra:
            sentence += " " + ", ".join(extra) + "."

        return sentence or "I got the weather, but the temperature was not available."
    def speak(self, text: str) -> None:
        print(f"\njarvis: {text}\n")
        self.tts.queue(text)

    def speak_blocking(self, text: str) -> None:
        
        print(f"\njarvis: {text}\n")
        self.tts.queue(text)
        self.tts.wait()
    @staticmethod
    def is_wikipedia_query(user_message: str) -> bool:
        text = user_message.lower()
        return "wikipedia" in text or "wiki" in text


    def extract_wikipedia_query(self, user_message: str) -> str:
        text = user_message.strip()

        # Normalise common spoken phrases.
        replacements = [
            "can you check on wikipedia",
            "can you check wikipedia",
            "check on wikipedia",
            "check wikipedia",
            "search on wikipedia",
            "search wikipedia",
            "look up on wikipedia",
            "look up wikipedia",
            "wikipedia",
            "wiki",
        ]

        cleaned = text

        for phrase in replacements:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            cleaned = pattern.sub("", cleaned)

        cleaned = cleaned.strip(" ,.!?")

        # Remove leading filler words.
        cleaned = re.sub(
            r"^(what is|who is|tell me about|for|about)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" ,.!?")

        return cleaned or text


    @staticmethod
    def format_wikipedia_response(result: dict) -> str:
        if not result.get("ok"):
            return "I could not find a useful Wikipedia result for that."

        title = result.get("title") or "That topic"
        summary = result.get("summary") or ""

        if not summary:
            return f"I found {title} on Wikipedia, but there was no useful summary."

        # Keep it short for voice.
        sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
        short_summary = " ".join(sentences[:2]).strip()

        return f"According to Wikipedia, {title}: {short_summary}"
        # -------------------------
        # Conversation mode helpers
        # -------------------------

    @staticmethod
    def normalise_command_text(text: str) -> str:
        """Normalise speech text for reliable command matching."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def starts_with_any(self, text: str, phrases: list[str]) -> bool:
        text_norm = self.normalise_command_text(text)

        for phrase in phrases:
            phrase_norm = self.normalise_command_text(phrase)

            if text_norm == phrase_norm:
                return True

            if text_norm.startswith(phrase_norm + " "):
                return True

        return False

    def equals_any(self, text: str, phrases: list[str]) -> bool:
        text_norm = self.normalise_command_text(text)

        for phrase in phrases:
            phrase_norm = self.normalise_command_text(phrase)

            if text_norm == phrase_norm:
                return True

        return False

    def is_start_command(self, text: str) -> bool:
        return self.starts_with_any(text, self.config.start_words)

    def is_conversation_stop_command(self, text: str) -> bool:
        return self.equals_any(text, self.config.conversation_stop_words)

    def is_shutdown_command(self, text: str) -> bool:
        return self.equals_any(text, self.config.shutdown_words)

    def remove_start_word(self, text: str) -> str:
        """Remove the jarvis start phrase from the beginning of a command.

        Example:
            "jarvis what time is it" -> "what time is it"
            "Hey jarvis open VS Code" -> "open VS Code"
            "jarvis" -> ""
        """
        original = text.strip()
        original_words = original.split()
        text_norm = self.normalise_command_text(text)

        for start_word in self.config.start_words:
            start_norm = self.normalise_command_text(start_word)

            if text_norm == start_norm:
                return ""

            if text_norm.startswith(start_norm + " "):
                start_word_count = len(start_norm.split())
                remaining_words = original_words[start_word_count:]
                return " ".join(remaining_words).strip(" ,.!?")

        return original

    # -------------------------
    # Direct tool helpers
    # -------------------------

    @staticmethod
    def is_time_query(user_message: str) -> bool:
        text = user_message.lower().strip()
        time_phrases = [
            "what's the time",
            "what is the time",
            "what time is it",
            "current time",
            "time now",
            "time is it in",
            "time in",
            "what day is it",
            "what's the date",
            "today's date",
            "date today",
        ]
        return any(phrase in text for phrase in time_phrases)

    @staticmethod
    def extract_time_location(user_message: str) -> str | None:
        """Extract a simple location from phrases like 'time in Liverpool'."""
        text = user_message.strip()
        text_lower = text.lower().strip()

        markers = ["time is it in ", "time in ", "date in "]
        for marker in markers:
            index = text_lower.find(marker)
            if index != -1:
                location_start = index + len(marker)
                location = text[location_start:].strip(" ,.!?")
                return location or None

        return None

    def try_direct_tool_response(self, user_message: str) -> str | None:
        """Handle obvious tool questions without asking the LLM to choose a tool."""

        if self.is_time_query(user_message):
            location = self.extract_time_location(user_message)

            result = self.tools.get_current_time(location=location)

            self.log_direct_tool_call(
                tool_name="get_current_time",
                arguments={"location": location},
                result=result,
            )

            if not result.get("ok"):
                return "I could not work out that timezone. Try asking for a city, like London or Tokyo."

            if "date" in user_message.lower() or "day" in user_message.lower():
                return f"It is {result['date']} in {result['location']}."

            return f"It is {result['time_12h']} in {result['location']}."

        if self.is_weather_query(user_message):
            location = self.extract_weather_location(user_message)

            result = self.tools.get_weather(location=location)

            self.log_direct_tool_call(
                tool_name="get_weather",
                arguments={"location": location},
                result=result,
            )

            return self.format_weather_response(result)

        if self.is_wikipedia_query(user_message):
            query = self.extract_wikipedia_query(user_message)

            result = self.tools.wiki_summary(query=query)

            self.log_direct_tool_call(
                tool_name="wiki_summary",
                arguments={"query": query},
                result=result,
            )

            return self.format_wikipedia_response(result)
    @staticmethod
    def should_use_tools(user_message: str) -> bool:
        text = user_message.lower()
        tool_keywords = [
            "latest",
            "current",
            "news",
            "search",
            "look up",
            "google",
            "online",

            "wikipedia",
            "wiki",

            "weather",
            "temperature",
            "temp",
            "forecast",
            "rain",
            "umbrella",

            "national holiday",
            "public holiday",
            "bank holiday",
            "holiday this week",
            "holiday today",
            "next holiday",

            "price",
            "stock",
            "who won",
            "score",
        ]
        return any(keyword in text for keyword in tool_keywords)

    def select_model(self, user_message: str) -> str:
        """Keep this conservative. Voice UX is better with the fast model."""
        text = user_message.lower()
        smart_keywords = [
            "debug this code",
            "analyse this code",
            "analyze this code",
            "think carefully",
            "complex",
            "architecture",
        ]
        if any(keyword in text for keyword in smart_keywords):
            return self.smart_model
        return self.fast_model

    def trim_messages(self) -> None:
        if len(self.messages) > 18:
            self.messages = [self.messages[0]] + self.messages[-14:]

    # -------------------------
    # Ollama chat helpers
    # -------------------------

    def ollama_chat_stream(self, messages: list[dict], model: str) -> str:
        full_reply = ""
        speech_buffer = ""

        print("\njarvis: ", end="", flush=True)

        for token in self.ollama.stream_chat(messages=messages, model=model):
            print(token, end="", flush=True)
            full_reply += token
            speech_buffer += token

            chunks, speech_buffer = self.tts.extract_speakable_chunks(speech_buffer)
            for chunk in chunks:
                self.tts.queue(chunk)

        final_chunks, _ = self.tts.extract_speakable_chunks(speech_buffer, force=True)
        for chunk in final_chunks:
            self.tts.queue(chunk)

        print("\n")
        return full_reply.strip()

    def ask_ollama_streaming(self, user_message: str) -> str:
        """Ask Ollama, using deterministic tools first when possible."""
        model = self.select_model(user_message)

        try:
            direct_reply = self.try_direct_tool_response(user_message)
            if direct_reply:
                print(f"jarvis: {direct_reply}")
                self.tts.queue(direct_reply)
                self.messages.append({"role": "user", "content": user_message})
                self.messages.append({"role": "assistant", "content": direct_reply})
                self.trim_messages()
                return direct_reply

            router_reply = self.ask_with_generic_tool_router(
                user_message=user_message,
                model=model,
            )

            if router_reply:
                self.messages.append({"role": "user", "content": user_message})
                self.messages.append({"role": "assistant", "content": router_reply})
                self.trim_messages()
                return router_reply

            self.messages.append({"role": "user", "content": user_message})
            full_reply = self.ollama_chat_stream(messages=self.messages, model=model)

            if full_reply:
                self.messages.append({"role": "assistant", "content": full_reply})

            self.trim_messages()
            return full_reply

        except requests.exceptions.ConnectionError:
            error = "I cannot connect to Ollama. Please make sure Ollama is running."
            self.speak(error)
            return error
        except requests.exceptions.Timeout:
            error = "Ollama took too long to reply."
            self.speak(error)
            return error
        except Exception as e:
            error = f"Something went wrong when talking to Ollama: {e}"
            self.speak(error)
            return error

    def ask_ollama_with_tools(self, user_message: str, model: str) -> str:
        """Single-round tool calling: model chooses tool, Python executes, model speaks final."""
        self.messages.append({"role": "user", "content": user_message})

        first_response = self.ollama.chat_once(
            messages=self.messages,
            model=model,
            tools=self.tools.build_tool_schemas(),
        )

        assistant_message = first_response.get("message", {})
        tool_calls = assistant_message.get("tool_calls") or []
        print("\n" + "-" * 60)
        print(f"Model selected for tool decision: {model}")
        print("Raw assistant tool decision:")
        print(json.dumps(assistant_message, ensure_ascii=False, indent=2))
        print("-" * 60 + "\n")
        # If the model decides no tool is needed, speak the direct answer.
        if not tool_calls:
            reply = assistant_message.get("content", "").strip()
            print(f"\njarvis: {reply}\n")
            if reply:
                self.tts.queue(reply)
                self.messages.append({"role": "assistant", "content": reply})
            self.trim_messages()
            return reply

        # Keep the model's tool-call message in the conversation.
        self.messages.append(
            {
                "role": "assistant",
                "content": assistant_message.get("content", ""),
                "tool_calls": tool_calls,
            }
        )

        for tool_call in tool_calls:
            tool_name, tool_result = self.tools.execute_tool_call(tool_call)
            print(f"Tool used: {tool_name} -> {tool_result}")

            self.messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )

        final_reply = self.ollama_chat_stream(messages=self.messages, model=model)

        if final_reply:
            self.messages.append({"role": "assistant", "content": final_reply})

        self.trim_messages()
        return final_reply

    # -------------------------
    # Main loop
    # -------------------------

    def run(self) -> None:
        self.speak_blocking("jarvis is online my lord. Say jarvis when you need me.")

        while True:
            text = self.listener.listen()

            if not text:
                continue
            # If a timer alarm is ringing, "stop" should silence it before any normal
# shutdown/conversation-stop logic runs.
            if hasattr(self.tools, "has_active_alarm") and self.tools.has_active_alarm():
                if self.is_alarm_stop_command(text):
                    result = self.tools.stop_timer_alarm()
                    self.speak_blocking(result.get("message", "Alarm stopped."))
                    continue
            # Shutdown should work from both idle mode and conversation mode.
            # It requires a clear full phrase like "jarvis shut down".
            if self.is_shutdown_command(text):
                self.speak_blocking("It was great serving you my lord. Goodbye.")
                self.tts.shutdown()
                break

            # -------------------------
            # Idle mode
            # -------------------------
            if not self.in_conversation:
                if not self.is_start_command(text):
                    print("Start word not detected. Ignoring.")
                    continue

                self.in_conversation = True
                command = self.remove_start_word(text)

                if not command:
                    self.speak_blocking("Yes? How can I help you?")
                    continue

            # -------------------------
            # Conversation mode
            # -------------------------
            else:
                command = text.strip()

            # User leaves conversation mode, but Jarvis keeps running.
            if self.is_conversation_stop_command(command):
                self.in_conversation = False
                self.speak_blocking("You can call me again anytime by saying jarvis.")
                continue

            self.ask_ollama_streaming(command)

            # Avoid Jarvis hearing itself.
            self.tts.wait()