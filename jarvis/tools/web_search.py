import requests

from jarvis.config import AssistantConfig


def schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current or recent information. Use this for latest news, "
                "current facts, live information, recent product/company facts, or anything that may have changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The web search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Use 3 by default.",
                    },
                },
                "required": ["query"],
            },
        },
    }


def web_search(config: AssistantConfig, query: str, max_results: int = 5) -> dict:
    max_results = max(1, min(int(max_results or 3), 5))

    if not config.enable_web_search:
        return {"ok": False, "error": "Web search is disabled. Set ENABLE_WEB_SEARCH=true in .env to enable it."}

    if config.tavily_api_key:
        return _tavily_search(config=config, query=query, max_results=max_results)

    return _duckduckgo_instant_answer(query=query, max_results=max_results)


def _tavily_search(config: AssistantConfig, query: str, max_results: int = 5) -> dict:
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": config.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": max_results,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for index, item in enumerate(data.get("results", [])[:max_results], start=1):
            results.append(
                {
                    "index": index,
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),
                }
            )

        context_lines = []

        if data.get("answer"):
            context_lines.append(f"Search engine answer: {data.get('answer')}")

        for item in results:
            title = item.get("title") or "Untitled source"
            content = item.get("content") or ""
            url = item.get("url") or ""

            if len(content) > 700:
                content = content[:700].rsplit(" ", 1)[0] + "."

            context_lines.append(
                f"Source {item['index']}: {title}\nURL: {url}\nContent: {content}"
            )

        return {
            "ok": True,
            "source": "Tavily",
            "query": query,
            "answer": data.get("answer"),
            "results": results,
            "context": "\n\n".join(context_lines),
        }

    except Exception as e:
        return {
            "ok": False,
            "source": "Tavily",
            "query": query,
            "error": str(e),
        }


def _duckduckgo_instant_answer(query: str, max_results: int = 3) -> dict:
    """No-key fallback. Useful, but not as reliable as a real search API."""
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()

        results = []

        abstract = data.get("AbstractText") or data.get("Answer")
        abstract_url = data.get("AbstractURL")
        if abstract:
            results.append(
                {
                    "title": data.get("Heading") or "DuckDuckGo answer",
                    "url": abstract_url,
                    "content": abstract,
                }
            )

        def collect_related(items: list[dict]) -> None:
            for item in items:
                if len(results) >= max_results:
                    return
                if "Topics" in item:
                    collect_related(item.get("Topics", []))
                elif item.get("Text"):
                    results.append(
                        {
                            "title": item.get("Text", "")[:80],
                            "url": item.get("FirstURL"),
                            "content": item.get("Text"),
                        }
                    )

        collect_related(data.get("RelatedTopics", []))

        if not results:
            return {
                "ok": False,
                "source": "DuckDuckGo Instant Answer",
                "query": query,
                "error": "No useful instant-answer result was returned. Add TAVILY_API_KEY for stronger web search.",
            }

        return {
            "ok": True,
            "source": "DuckDuckGo Instant Answer",
            "query": query,
            "answer": abstract,
            "results": results[:max_results],
        }
    except Exception as e:
        return {"ok": False, "source": "DuckDuckGo Instant Answer", "query": query, "error": str(e)}
