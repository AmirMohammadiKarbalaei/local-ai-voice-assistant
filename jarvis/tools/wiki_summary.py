from urllib.parse import quote

import requests


def schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "wiki_summary",
            "description": (
                "Get a short factual summary from Wikipedia. Use this when the user explicitly says "
                "Wikipedia or wiki, or asks for a general factual summary about people, places, "
                "concepts, technologies, and historical topics. Do not use this for live or current facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic to search on Wikipedia."}
                },
                "required": ["query"],
            },
        },
    }


def wiki_summary(query: str) -> dict:
    try:
        search_response = requests.get(
            "https://en.wikipedia.org/w/rest.php/v1/search/page",
            params={"q": query, "limit": 1},
            headers={"User-Agent": "SSPAssistant/1.0"},
            timeout=12,
        )
        search_response.raise_for_status()
        search_data = search_response.json()

        pages = search_data.get("pages", [])
        if not pages:
            return {"ok": False, "query": query, "error": "No Wikipedia page found."}

        title = pages[0]["title"]
        encoded_title = quote(title.replace(" ", "_"))

        summary_response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}",
            headers={"User-Agent": "SSPAssistant/1.0"},
            timeout=12,
        )
        summary_response.raise_for_status()
        summary_data = summary_response.json()

        return {
            "ok": True,
            "query": query,
            "title": summary_data.get("title"),
            "description": summary_data.get("description"),
            "summary": summary_data.get("extract"),
            "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page"),
        }

    except Exception as e:
        return {"ok": False, "query": query, "error": str(e)}
