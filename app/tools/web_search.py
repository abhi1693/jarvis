from __future__ import annotations

from typing import Any

from duckduckgo_search import DDGS


class WebSearchTool:
    def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        try:
            with DDGS() as client:
                results = list(client.text(query, max_results=max_results))
        except Exception as error:
            return {"ok": False, "error": str(error)}

        return {
            "ok": True,
            "query": query,
            "results": [
                {
                    "title": result.get("title", ""),
                    "href": result.get("href", ""),
                    "body": result.get("body", ""),
                }
                for result in results
            ],
        }
