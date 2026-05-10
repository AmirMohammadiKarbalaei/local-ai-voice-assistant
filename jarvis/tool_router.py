import inspect
import json
import re


class ToolRouter:
    """LLM-based generic tool router for Jarvis.

    It chooses whether a tool is needed, which tool to use, and what arguments
    to pass. It also repairs common argument mistakes before execution.

    This version supports:
    - time/date
    - weather
    - web search
    - Wikipedia
    - calculator
    - currency conversion
    - timers
    """

    TOOL_ALIASES = {
        # Web search
        "search_web": "web_search",
        "search": "web_search",
        "google": "web_search",
        "online_search": "web_search",

        # Weather
        "weather": "get_weather",
        "forecast": "get_weather",
        "temperature": "get_weather",

        # Time/date
        "time": "get_current_time",
        "date": "get_current_time",
        "current_time": "get_current_time",

        # Wikipedia
        "wiki": "wiki_summary",
        "wikipedia": "wiki_summary",

        # Calculation
        "calc": "calculate",
        "calculator": "calculate",
        "math": "calculate",

        # Currency
        "currency": "convert_currency",
        "exchange_rate": "convert_currency",
        "currency_conversion": "convert_currency",

        # Timers
        "timer": "set_timer",
        "set_timer": "set_timer",
        "start_timer": "set_timer",
        "create_timer": "set_timer",
        "countdown": "set_timer",

        "timers": "list_timers",
        "list_timer": "list_timers",
        "list_timers": "list_timers",
        "active_timers": "list_timers",

        "cancel_timer": "cancel_timer",
        "stop_timer": "cancel_timer",
        "delete_timer": "cancel_timer",
        "remove_timer": "cancel_timer",

        "stop_alarm": "stop_timer_alarm",
        "silence_alarm": "stop_timer_alarm",
        "stop_timer_alarm": "stop_timer_alarm",
    }

    def __init__(self, ollama, config, tools):
        self.ollama = ollama
        self.config = config
        self.tools = tools

    def route(self, user_message: str, model: str) -> dict:
        tool_schemas = self.tools.build_tool_schemas()
        available_tool_names = sorted(self.tools.available_tools.keys())

        prompt = f"""
You are a tool-routing layer for a local voice assistant called Jarvis.

Your job is only to decide whether the user's message needs a tool.
The user's message comes from speech-to-text, so it may contain transcription mistakes.

Correct obvious STT mistakes when choosing the tool and arguments.

Common STT examples:
- "what timer is it" may mean "what time is it"
- "weather in little pool" may mean "weather in Liverpool"
- "search restaurants round me" may mean "search restaurants around me"
- "set a pasta time for ten minutes" may mean "set a pasta timer for ten minutes"
- "how long left" may mean list_timers
- "stop the pasta time" may mean cancel the pasta timer

Return JSON only.
Do not answer the user.
Do not use markdown.
Do not include explanations outside the JSON.

User default location:
{self.config.user_location}

User timezone:
{self.config.user_timezone}

Available tool names:
{json.dumps(available_tool_names, ensure_ascii=False)}

Available tools with their exact required arguments:
{json.dumps(tool_schemas, ensure_ascii=False, indent=2)}

Tool-selection rules:
- Only choose a tool from the available tool names.
- Use get_current_time for time, date, current day, or time in a location.
- Use get_weather for weather, temperature, forecast, rain, wind, hot, cold, or umbrella questions.
- Use web_search for current, live, recent, online, local, price, stock, news, sports, recommendation, or search questions.
- Use wiki_summary only for Wikipedia/wiki or non-current encyclopedia-style summaries.
- Use calculate for arithmetic, percentages, and numeric calculations.
- Use convert_currency for money conversion or exchange-rate conversion.
- Use set_timer when the user wants to set, start, create, or run a timer/countdown.
- Use list_timers when the user asks how long is left, what timers are active, or asks to check timers.
- Use cancel_timer when the user asks to cancel, stop, remove, or delete a timer.

Argument rules:
- For web_search, provide "query", not "location".
- For web_search local searches, convert "near me", "around me", "nearby", "round me", or "around here" into the user's default location inside the query.
- For get_current_time, use "location" only if the user asks for time in a place.
- For get_weather, use "location" only if the user mentions a city or place.
- For wiki_summary, use "query".
- For calculate, use "expression".
- For convert_currency, use "amount", "from_currency", and "to_currency".
- For set_timer, provide "duration_seconds" if you can calculate it.
- For set_timer, provide "duration" as natural text if you cannot safely calculate seconds.
- For set_timer, label means what the timer is for, not the duration. 
- Examples: pasta, tea, laundry, oven, break, workout.
- Do not use duration phrases like "one minute" or "five minutes" as the label.
- For set_timer, use "label" if the user names the timer, such as pasta, tea, laundry, workout, break, or oven.
- For set_timer, "warning_seconds" is optional. Only provide it if the user asks for a specific warning. Otherwise omit it and let the tool choose.
- For list_timers, use no arguments.
- For cancel_timer, use "label" if the user names the timer.

Use a tool when the user asks for:
- current, live, recent, online, or local information
- weather or temperature
- time or date
- web search
- Wikipedia or wiki
- currency conversion
- calculation
- prices, stocks, sports scores, news, or recommendations that need online information
- setting a timer
- checking timers
- cancelling timers

If no tool is needed, return use_tool false.

Return exactly this JSON shape:
{{
  "use_tool": true,
  "tool_name": "tool name here",
  "arguments": {{}},
  "confidence": 0.0,
  "reason": "short reason"
}}

Or:
{{
  "use_tool": false,
  "tool_name": null,
  "arguments": {{}},
  "confidence": 0.0,
  "reason": "short reason"
}}

User message:
{user_message}
""".strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON tool router. "
                    "Return valid JSON only. "
                    "No markdown. No prose. "
                    "Do not answer the user."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        try:
            response = self._chat_router(
                messages=messages,
                model=model,
            )

            content = response.get("message", {}).get("content", "").strip()
            data = self._extract_json(content)
            route = self._validate_route(data)

            if route.get("use_tool"):
                route = self._repair_arguments(route, user_message)

            return route

        except Exception as e:
            return {
                "use_tool": False,
                "tool_name": None,
                "arguments": {},
                "confidence": 0.0,
                "reason": f"Tool router failed: {e}",
            }

    def _chat_router(self, messages: list[dict], model: str) -> dict:
        """Call Ollama for routing.

        This supports both versions of your OllamaClient:
        - newer chat_once with temperature/response_format args
        - older chat_once with only messages/model/tools
        """
        try:
            return self.ollama.chat_once(
                messages=messages,
                model=model,
                tools=None,
                temperature=0,
                num_ctx=2048,
                num_predict=200,
                response_format="json",
                keep_alive="10m",
            )
        except TypeError:
            return self.ollama.chat_once(
                messages=messages,
                model=model,
                tools=None,
            )

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        text = (
            text.replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find balanced JSON objects and try the last valid one.
        objects = []
        start = None
        depth = 0
        in_string = False
        escape = False

        for index, char in enumerate(text):
            if escape:
                escape = False
                continue

            if char == "\\" and in_string:
                escape = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                if depth == 0:
                    start = index
                depth += 1

            elif char == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start:index + 1])
                    start = None

        for candidate in reversed(objects):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise ValueError(f"No valid JSON found in router output: {text[:200]}")

    def _validate_route(self, data: dict) -> dict:
        if not isinstance(data, dict):
            data = {}

        use_tool = bool(data.get("use_tool"))
        tool_name = data.get("tool_name")
        tool_name = self.TOOL_ALIASES.get(tool_name, tool_name)

        arguments = data.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}

        reason = data.get("reason") or ""

        default_confidence = 0.75 if use_tool else 0.0
        try:
            confidence = float(data.get("confidence", default_confidence) or default_confidence)
        except (TypeError, ValueError):
            confidence = default_confidence

        confidence = max(0.0, min(confidence, 1.0))

        if not use_tool:
            return {
                "use_tool": False,
                "tool_name": None,
                "arguments": {},
                "confidence": confidence,
                "reason": reason,
            }

        if tool_name not in self.tools.available_tools:
            return {
                "use_tool": False,
                "tool_name": None,
                "arguments": {},
                "confidence": 0.0,
                "reason": f"Unknown tool selected: {tool_name}",
            }

        return {
            "use_tool": True,
            "tool_name": tool_name,
            "arguments": arguments,
            "confidence": confidence,
            "reason": reason,
        }

    def _repair_arguments(self, route: dict, user_message: str) -> dict:
        """Repair common LLM argument mistakes.

        This is not hard-coding every user phrase. It normalises arguments so
        they match the Python function signatures.
        """
        tool_name = route.get("tool_name")
        arguments = route.get("arguments") or {}

        if tool_name == "web_search":
            arguments = self._repair_web_search_args(arguments, user_message)

        elif tool_name == "wiki_summary":
            if "query" not in arguments:
                arguments["query"] = user_message

        elif tool_name == "get_weather":
            if "location" not in arguments and "query" in arguments:
                arguments["location"] = arguments.pop("query")

        elif tool_name == "get_current_time":
            if "location" not in arguments and "query" in arguments:
                arguments["location"] = arguments.pop("query")

        elif tool_name == "calculate":
            if "expression" not in arguments and "query" in arguments:
                arguments["expression"] = arguments.pop("query")

        elif tool_name == "convert_currency":
            arguments = self._repair_currency_args(arguments, user_message)

        elif tool_name == "set_timer":
            arguments = self._repair_timer_args(arguments, user_message)

        elif tool_name == "cancel_timer":
            if "label" not in arguments and "query" in arguments:
                arguments["label"] = arguments.pop("query")

            if "label" not in arguments:
                label = self._guess_timer_label(user_message)
                if label:
                    arguments["label"] = label

        elif tool_name == "list_timers":
            arguments = {}

        route["arguments"] = self._filter_arguments_for_tool(tool_name, arguments)
        return route

    def _repair_web_search_args(self, arguments: dict, user_message: str) -> dict:
        """Make sure web_search receives query and max_results."""
        location = (
            arguments.get("location")
            or arguments.get("place")
            or arguments.get("city")
            or self.config.user_location
        )

        query = arguments.get("query") or user_message
        query = self._make_location_aware_query(
            query=query,
            location=location,
            user_message=user_message,
        )

        try:
            max_results = int(arguments.get("max_results", 5) or 5)
        except (TypeError, ValueError):
            max_results = 5

        return {
            "query": query,
            "max_results": max(3, min(max_results, 5)),
        }

    def _make_location_aware_query(
        self,
        query: str,
        location: str,
        user_message: str,
    ) -> str:
        query = str(query).strip(" ,.!?")

        local_phrases = [
            "around me",
            "near me",
            "nearby",
            "close to me",
            "in my area",
            "around here",
            "round me",
        ]

        combined = f"{query} {user_message}".lower()
        local_requested = any(phrase in combined for phrase in local_phrases)

        for phrase in local_phrases:
            query = re.sub(
                re.escape(phrase),
                f"in {location}",
                query,
                flags=re.IGNORECASE,
            )

        if local_requested and location.lower() not in query.lower():
            query = f"{query} in {location}"

        query = re.sub(
            r"^(can you|could you|please|find me|find|search for|look up|show me)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        ).strip(" ,.!?")

        return query

    def _repair_currency_args(self, arguments: dict, user_message: str) -> dict:
        if "amount" not in arguments:
            match = re.search(r"\b(\d+(?:\.\d+)?)\b", user_message)
            if match:
                arguments["amount"] = float(match.group(1))

        if "from_currency" in arguments:
            arguments["from_currency"] = str(arguments["from_currency"]).upper().strip()

        if "to_currency" in arguments:
            arguments["to_currency"] = str(arguments["to_currency"]).upper().strip()

        return arguments

    def _repair_timer_args(self, arguments: dict, user_message: str) -> dict:
        """Repair timer arguments.

        The timer tool can accept either:
        - duration_seconds
        - duration as natural text
        """
        if "duration_seconds" not in arguments and "duration" not in arguments:
            seconds = self._extract_duration_seconds(user_message)

            if seconds:
                arguments["duration_seconds"] = seconds
            else:
                arguments["duration"] = user_message

        if "duration_seconds" in arguments:
            try:
                arguments["duration_seconds"] = int(float(arguments["duration_seconds"]))
            except (TypeError, ValueError):
                arguments.pop("duration_seconds", None)
                arguments["duration"] = user_message

        if "warning_seconds" in arguments:
            try:
                arguments["warning_seconds"] = int(float(arguments["warning_seconds"]))
            except (TypeError, ValueError):
                arguments.pop("warning_seconds", None)

        if "label" not in arguments:
            label = self._guess_timer_label(user_message)
            if label:
                arguments["label"] = label

        return arguments

    def _extract_duration_seconds(self, text: str) -> int | None:
        text = text.lower().strip()

        word_numbers = {
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

        for word, number in word_numbers.items():
            text = re.sub(rf"\b{word}\b", str(number), text)

        total_seconds = 0.0

        matches = re.findall(
            r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|sec|s|minutes?|mins?|min|m|hours?|hrs?|hr|h)\b",
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

    def _guess_timer_label(self, user_message: str) -> str | None:
        text = user_message.lower().strip()

        # "set a pasta timer for 10 minutes" -> pasta
        match = re.search(
            r"\b(?:set|start|create)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+timer\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            label = self._clean_timer_label(match.group(1))
            if label:
                return label

        # "set a timer for pasta for 10 minutes" -> pasta
        match = re.search(
            r"\btimer\s+for\s+(.+?)(?:\s+for\s+\d|\s+in\s+\d|$)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            label = self._clean_timer_label(match.group(1))
            if label:
                return label

        # "cancel pasta timer" -> pasta
        match = re.search(
            r"\b(?:cancel|stop|delete|remove)\s+(?:the\s+|a\s+)?(.+?)\s+timer\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            label = self._clean_timer_label(match.group(1))
            if label:
                return label

        return None

    @staticmethod
    def _clean_timer_label(label: str) -> str | None:
        label = label.strip(" ,.!?")
        label = re.sub(
            r"\b(for|in|after|lasting|duration|called|named)\b.*$",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip(" ,.!?")

        bad_labels = {
            "",
            "timer",
            "a",
            "an",
            "the",
            "set",
            "start",
            "create",
            "cancel",
            "stop",
            "delete",
            "remove",
        }

        if label.lower() in bad_labels:
            return None

        return label

    def _filter_arguments_for_tool(self, tool_name: str, arguments: dict) -> dict:
        """Remove arguments that the actual Python function does not accept."""
        if tool_name not in self.tools.available_tools:
            return arguments

        function = self.tools.available_tools[tool_name]
        signature = inspect.signature(function)
        allowed = set(signature.parameters.keys())

        return {
            key: value
            for key, value in arguments.items()
            if key in allowed
        }