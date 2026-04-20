from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAI

try:
    from planning import SpendingProfile
except ImportError:  # pragma: no cover
    from src.planning import SpendingProfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tool_registry import (
    TOOL_SCHEMAS_OPENAI,
    tool_calculator,
    tool_fetch_webpage,
    tool_save_research_note,
    tool_web_search,
)


ALLOWED_TOOL_NAMES = {
    "web_search",
    "fetch_webpage",
    "calculator",
    "save_research_note",
}


@dataclass
class CardRecommendation:
    primary_card: str
    primary_issuer: str
    primary_estimated_value: float
    runner_up_card: str
    runner_up_issuer: str
    runner_up_estimated_value: float
    explanation: str
    caveats: list[str]
    card_rankings: list[dict[str, object]]
    sources: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "primary_recommendation": {
                "card_name": self.primary_card,
                "issuer": self.primary_issuer,
                "estimated_annual_value": self.primary_estimated_value,
            },
            "runner_up_recommendation": {
                "card_name": self.runner_up_card,
                "issuer": self.runner_up_issuer,
                "estimated_annual_value": self.runner_up_estimated_value,
            },
            "explanation": self.explanation,
            "caveats": self.caveats,
            "card_rankings": self.card_rankings,
            "sources": self.sources,
        }


class CreditCardRecommender:
    """Shared control stage using an LLM plus tool_registry web tools."""

    def __init__(
        self,
        model: str | None = None,
        max_tool_rounds: int = 10,
    ) -> None:
        load_dotenv(_find_dotenv_path(), override=False)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tool_rounds = max_tool_rounds

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY was not found in the environment or .env file.")

        self.client = OpenAI(api_key=api_key)
        self.tools = [
            schema
            for schema in TOOL_SCHEMAS_OPENAI
            if schema["function"]["name"] in ALLOWED_TOOL_NAMES
        ]
        self.tool_implementations = {
            "web_search": tool_web_search,
            "fetch_webpage": tool_fetch_webpage,
            "calculator": tool_calculator,
            "save_research_note": tool_save_research_note,
        }

    def recommend_card(self, spending_profile: SpendingProfile) -> CardRecommendation:
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            {
                "role": "user",
                "content": self._build_user_prompt(spending_profile),
            },
        ]

        used_tools: set[str] = set()
        for _ in range(self.max_tool_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )

            message = response.choices[0].message
            if message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in message.tool_calls
                        ],
                    }
                )

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    used_tools.add(tool_name)
                    tool_result = self._execute_tool_call(
                        tool_name=tool_name,
                        raw_arguments=tool_call.function.arguments,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result),
                        }
                    )
                continue

            content = message.content or "{}"
            if not {"web_search", "fetch_webpage"}.issubset(used_tools):
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You have not yet completed enough live research. "
                            "Use web_search and fetch_webpage before finalizing your recommendation."
                        ),
                    }
                )
                continue

            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Your last response was not valid JSON. "
                            "Return the same recommendation again as valid JSON only."
                        ),
                    }
                )
                continue
            return self._normalize_payload(payload)

        raise RuntimeError(
            "The control agent did not finish within the allowed tool rounds."
        )

    def _system_prompt(self) -> str:
        today = dt.date.today().isoformat()
        return (
            "You are the shared control-phase agent for a credit-card recommendation project. "
            f"Today's date is {today}. "
            "Your job is to recommend the best current credit card and a runner-up based on a spending profile. "
            "You must research live information before answering. "
            "Always use web_search to discover relevant current card options and fetch_webpage to inspect promising sources. "
            "Prefer official issuer pages when possible, but reputable comparison sites are acceptable for discovery. "
            "Use calculator if it helps estimate annual reward value. "
            "Be efficient: usually 1-2 search queries and 1-3 webpage fetches are enough before finalizing. "
            "Return valid JSON only. "
            "The JSON must include these keys exactly: "
            "primary_recommendation, runner_up_recommendation, explanation, caveats, card_rankings, sources. "
            "primary_recommendation must match card_rankings[0]. "
            "runner_up_recommendation must match card_rankings[1]. "
            "Keep caveats short and practical. "
            "Each source must include title, url, and why_it_matters."
        )

    def _build_user_prompt(self, spending_profile: SpendingProfile) -> str:
        top_categories = [
            {"category": category, "amount": amount}
            for category, amount in sorted(
                spending_profile.category_totals.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if amount > 0
        ]

        search_hints = [
            f"best credit cards for {entry['category']} rewards"
            for entry in top_categories[:3]
        ]
        if not search_hints:
            search_hints = ["best flat-rate cash back credit cards"]

        return json.dumps(
            {
                "task": (
                    "Use the spending profile to research current credit cards and recommend "
                    "the best one plus a runner-up."
                ),
                "workflow": [
                    "Search for current cards relevant to the top spending categories.",
                    "Open promising pages to verify reward rates, fees, and positioning.",
                    "Estimate the likely annual value for the user based on their spending profile.",
                    "Return a concise structured recommendation with cited sources.",
                    "Keep tool use efficient and finalize once you have enough evidence.",
                ],
                "suggested_search_queries": search_hints,
                "spending_profile": spending_profile.as_dict(),
                "response_schema": {
                    "primary_recommendation": {
                        "card_name": "string",
                        "issuer": "string",
                        "estimated_annual_value": "number",
                    },
                    "runner_up_recommendation": {
                        "card_name": "string",
                        "issuer": "string",
                        "estimated_annual_value": "number",
                    },
                    "explanation": "string",
                    "caveats": ["string"],
                    "card_rankings": [
                        {
                            "card_name": "string",
                            "issuer": "string",
                            "estimated_annual_value": "number",
                            "reason": "string",
                        }
                    ],
                    "sources": [
                        {
                            "title": "string",
                            "url": "string",
                            "why_it_matters": "string",
                        }
                    ],
                },
            },
            indent=2,
        )

    def _execute_tool_call(self, tool_name: str, raw_arguments: str) -> dict[str, object]:
        if tool_name not in self.tool_implementations:
            return {"error": f"Unknown tool requested: {tool_name}"}

        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError as exc:
            return {"error": f"Could not parse tool arguments: {exc}"}

        try:
            result = self.tool_implementations[tool_name](**arguments)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as exc:  # pragma: no cover - defensive tool wrapper
            return {"error": str(exc), "tool_name": tool_name}

    def _normalize_payload(self, payload: dict[str, object]) -> CardRecommendation:
        primary = payload.get("primary_recommendation", {})
        runner_up = payload.get("runner_up_recommendation", {})
        caveats = payload.get("caveats", [])
        rankings = payload.get("card_rankings", [])
        sources = payload.get("sources", [])

        if not isinstance(primary, dict) or not primary:
            raise ValueError("LLM response missing primary_recommendation.")
        if not isinstance(runner_up, dict) or not runner_up:
            raise ValueError("LLM response missing runner_up_recommendation.")
        if not isinstance(caveats, list):
            caveats = []
        if not isinstance(rankings, list):
            rankings = []
        if not isinstance(sources, list):
            sources = []

        normalized_rankings = [
            {
                "card_name": str(item.get("card_name", "")),
                "issuer": str(item.get("issuer", "")),
                "estimated_annual_value": float(item.get("estimated_annual_value", 0.0)),
                "reason": str(item.get("reason", "")),
            }
            for item in rankings
            if isinstance(item, dict)
        ]

        normalized_sources = [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "why_it_matters": str(item.get("why_it_matters", "")),
            }
            for item in sources
            if isinstance(item, dict)
        ]

        return CardRecommendation(
            primary_card=str(primary.get("card_name", "")),
            primary_issuer=str(primary.get("issuer", "")),
            primary_estimated_value=float(primary.get("estimated_annual_value", 0.0)),
            runner_up_card=str(runner_up.get("card_name", "")),
            runner_up_issuer=str(runner_up.get("issuer", "")),
            runner_up_estimated_value=float(runner_up.get("estimated_annual_value", 0.0)),
            explanation=str(payload.get("explanation", "")),
            caveats=[str(item) for item in caveats],
            card_rankings=normalized_rankings,
            sources=normalized_sources,
        )


def _find_dotenv_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".env"
