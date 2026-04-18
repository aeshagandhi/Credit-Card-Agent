"""
Tool Registry for the Research Assistant Agent.
Each tool has:
  - A JSON schema (for Claude's tool_use API)
  - A Python implementation
  - An executor function called by the agent loop
"""

import json
import re
import math
import datetime
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import arxiv


TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo. Returns a list of results with "
            "title, URL, and snippet. Use this to find current information, "
            "news, or general knowledge not available in training data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up."
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_webpage",
        "description": (
            "Fetch and extract the main text content from a webpage URL. "
            "Use this to read the full content of a search result or any URL. "
            "Returns cleaned text (first ~3000 words)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch."
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "arxiv_search",
        "description": (
            "Search ArXiv for academic papers. Returns papers with title, "
            "authors, abstract, publication date, and ArXiv URL. "
            "Use for scientific, ML, or technical research queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The academic search query."
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of papers to return (default 5, max 10).",
                    "default": 5
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "Sort order for results.",
                    "default": "relevance"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression safely. Supports arithmetic, "
            "exponents (**), sqrt, log, sin, cos, tan, pi, e, and basic stats. "
            "Use for any numeric calculation needed during research."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python-style math expression, e.g. '2**10', 'sqrt(144)', 'log(100, 10)'."
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "save_research_note",
        "description": (
            "Save an important finding, quote, or summary to the research notebook. "
            "Notes are persisted for the session and included in the final report. "
            "Use this to bookmark key facts, quotes, or conclusions as you research."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title for this note."
                },
                "content": {
                    "type": "string",
                    "description": "The note content — a finding, quote, or summary."
                },
                "source": {
                    "type": "string",
                    "description": "URL or citation for where this came from (optional).",
                    "default": ""
                }
            },
            "required": ["title", "content"]
        }
    }
]

def _to_openai_schema(s):
    return {"type": "function", "function": {"name": s["name"], "description": s["description"], "parameters": s["input_schema"]}}

TOOL_SCHEMAS_OPENAI = [_to_openai_schema(s) for s in TOOL_SCHEMAS]


def tool_web_search(query: str, num_results: int = 5) -> dict:
    """DuckDuckGo web search."""
    num_results = min(int(num_results), 10)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        if not results:
            return {"error": "No results found.", "results": []}
        cleaned = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            }
            for r in results
        ]
        return {"query": query, "results": cleaned, "count": len(cleaned)}
    except Exception as e:
        return {"error": str(e), "results": []}


def tool_fetch_webpage(url: str) -> dict:
    """Fetch and clean webpage content."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style/nav/footer noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        # Get main text
        text = soup.get_text(separator="\n", strip=True)
        # Collapse whitespace
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)
        # Limit to ~3000 words
        words = text.split()
        if len(words) > 3000:
            text = " ".join(words[:3000]) + "\n\n[... content truncated ...]"
        title = soup.title.string.strip() if soup.title else url
        return {"url": url, "title": title, "content": text, "word_count": len(words)}
    except Exception as e:
        return {"error": str(e), "url": url, "content": ""}


def tool_arxiv_search(query: str, num_results: int = 5, sort_by: str = "relevance") -> dict:
    """Search ArXiv for papers."""
    num_results = min(int(num_results), 10)
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }
    criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=num_results,
            sort_by=criterion
        )
        papers = []
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "authors": [a.name for a in result.authors[:5]],
                "abstract": result.summary[:500] + ("..." if len(result.summary) > 500 else ""),
                "published": result.published.strftime("%Y-%m-%d"),
                "url": result.entry_id,
                "pdf_url": result.pdf_url
            })
        return {"query": query, "results": papers, "count": len(papers)}
    except Exception as e:
        return {"error": str(e), "results": []}


def tool_calculator(expression: str) -> dict:
    """Safe math expression evaluator."""
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
    if re.search(r'[^0-9\s\+\-\*\/\(\)\.\,\_a-zA-Z\%\^]', expression):
        return {"error": "Expression contains disallowed characters."}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "expression": expression}

_research_notes: list = []

def tool_save_research_note(title: str, content: str, source: str = "") -> dict:
    """Save a note to the research notebook."""
    note = {
        "id": len(_research_notes) + 1,
        "title": title,
        "content": content,
        "source": source,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
    }
    _research_notes.append(note)
    return {"success": True, "note_id": note["id"], "total_notes": len(_research_notes)}

def get_all_notes() -> list:
    return _research_notes.copy()

def clear_notes():
    _research_notes.clear()


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatch tool call and return JSON string result."""
    try:
        if tool_name == "web_search":
            result = tool_web_search(**tool_input)
        elif tool_name == "fetch_webpage":
            result = tool_fetch_webpage(**tool_input)
        elif tool_name == "arxiv_search":
            result = tool_arxiv_search(**tool_input)
        elif tool_name == "calculator":
            result = tool_calculator(**tool_input)
        elif tool_name == "save_research_note":
            result = tool_save_research_note(**tool_input)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except TypeError as e:
        result = {"error": f"Invalid tool arguments: {e}"}
    return json.dumps(result, indent=2, default=str)
