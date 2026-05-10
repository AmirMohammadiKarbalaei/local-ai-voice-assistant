import json

from jarvis.config import AssistantConfig

from . import calculate as calculate_tool
from . import currency as currency_tool
from . import current_time as current_time_tool
from . import timer as timer_tool
from . import weather as weather_tool
from . import web_search as web_search_tool
from . import wiki_summary as wiki_summary_tool


class ToolRegistry:
    def __init__(self, config: AssistantConfig, notifier=None):
        self.config = config
        self.notifier = notifier
        self.timer_manager = timer_tool.TimerManager(notifier=notifier)

        self.available_tools = {
            "get_current_time": self.get_current_time,
            "web_search": self.web_search,
            "calculate": self.calculate,
            "convert_currency": self.convert_currency,
            "wiki_summary": self.wiki_summary,
            "get_weather": self.get_weather,
            "set_timer": self.set_timer,
            "list_timers": self.list_timers,
            "cancel_timer": self.cancel_timer,
            "stop_timer_alarm": self.stop_timer_alarm,
            "set_timer_warning": self.set_timer_warning,
        }

    def build_tool_schemas(self) -> list[dict]:
        return [
            current_time_tool.schema(),
            web_search_tool.schema(),
            calculate_tool.schema(),
            currency_tool.schema(),
            wiki_summary_tool.schema(),
            weather_tool.schema(),
            timer_tool.schema_set_timer(),
            timer_tool.schema_list_timers(),
            timer_tool.schema_cancel_timer(),
            timer_tool.schema_stop_timer_alarm(),
            timer_tool.schema_set_timer_warning(),
        ]
    def set_timer_warning(
        self,
        remaining_seconds=None,
        remaining=None,
        label=None,
        timer_id=None,
    ) -> dict:
        return self.timer_manager.set_timer_warning(
            remaining_seconds=remaining_seconds,
            remaining=remaining,
            label=label,
            timer_id=timer_id,
    )
    def get_current_time(
        self,
        location: str | None = None,
        timezone: str | None = None,
    ) -> dict:
        return current_time_tool.get_current_time(
            config=self.config,
            location=location,
            timezone=timezone,
        )

    def web_search(self, query: str, max_results: int = 3) -> dict:
        return web_search_tool.web_search(
            config=self.config,
            query=query,
            max_results=max_results,
        )

    def calculate(self, expression: str) -> dict:
        return calculate_tool.calculate(expression=expression)

    def convert_currency(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
    ) -> dict:
        return currency_tool.convert_currency(
            amount=amount,
            from_currency=from_currency,
            to_currency=to_currency,
        )

    def wiki_summary(self, query: str) -> dict:
        return wiki_summary_tool.wiki_summary(query=query)

    def geocode_location(self, location: str) -> dict:
        return weather_tool.geocode_location(location=location)

    def get_weather(
        self,
        location: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict:
        return weather_tool.get_weather(
            config=self.config,
            location=location,
            latitude=latitude,
            longitude=longitude,
        )
    def stop_timer_alarm(self) -> dict:
        return self.timer_manager.stop_alarm()


    def has_active_alarm(self) -> bool:
        return self.timer_manager.is_alarm_active()
    def set_timer(
        self,
        duration_seconds: int | float | None = None,
        duration: str | None = None,
        label: str | None = None,
        warning_seconds: int | float | None = None,
    ) -> dict:
        return self.timer_manager.set_timer(
            duration_seconds=duration_seconds,
            duration=duration,
            label=label,
            warning_seconds=warning_seconds,
        )

    def list_timers(self) -> dict:
        return self.timer_manager.list_timers()

    def cancel_timer(
        self,
        timer_id: str | None = None,
        label: str | None = None,
    ) -> dict:
        return self.timer_manager.cancel_timer(
            timer_id=timer_id,
            label=label,
        )

    def execute_tool_call(self, tool_call: dict) -> tuple[str, dict]:
        function = tool_call.get("function", {})
        name = function.get("name")
        arguments = function.get("arguments", {}) or {}

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        print("\n" + "=" * 60)
        print("MODEL TOOL CALL")
        print(f"Tool name: {name}")
        print(f"Arguments: {json.dumps(arguments, ensure_ascii=False, indent=2)}")
        print("=" * 60)

        if name not in self.available_tools:
            result = {
                "ok": False,
                "error": f"Unknown tool: {name}",
            }

            print("TOOL RESULT")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("=" * 60 + "\n")

            return name or "unknown_tool", result

        try:
            result = self.available_tools[name](**arguments)

            print("TOOL RESULT")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("=" * 60 + "\n")

            return name, result

        except Exception as e:
            result = {
                "ok": False,
                "error": str(e),
            }

            print("TOOL ERROR")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("=" * 60 + "\n")

            return name, result