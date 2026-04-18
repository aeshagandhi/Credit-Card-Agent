from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:
    from planning import SpendingProfile
except ImportError:  # pragma: no cover
    from src.planning import SpendingProfile


@dataclass
class CardProfile:
    name: str
    issuer: str
    annual_fee: float
    rewards: dict[str, float]
    notes: list[str] = field(default_factory=list)


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
        }


DEFAULT_CARD_CATALOG = [
    CardProfile(
        name="Blue Cash Preferred",
        issuer="American Express",
        annual_fee=95.0,
        rewards={
            "groceries": 0.06,
            "gas": 0.03,
            "dining": 0.01,
            "travel": 0.01,
            "entertainment": 0.01,
            "shopping": 0.01,
            "healthcare": 0.01,
            "other": 0.01,
        },
        notes=[
            "Strong grocery rewards but includes an annual fee.",
        ],
    ),
    CardProfile(
        name="Amex Gold",
        issuer="American Express",
        annual_fee=325.0,
        rewards={
            "groceries": 0.04,
            "dining": 0.04,
            "travel": 0.03,
            "gas": 0.01,
            "entertainment": 0.01,
            "shopping": 0.01,
            "healthcare": 0.01,
            "other": 0.01,
        },
        notes=[
            "Good fit for dining and grocery-heavy profiles if spending is high enough.",
        ],
    ),
    CardProfile(
        name="Chase Sapphire Preferred",
        issuer="Chase",
        annual_fee=95.0,
        rewards={
            "travel": 0.02,
            "dining": 0.03,
            "groceries": 0.01,
            "gas": 0.01,
            "entertainment": 0.01,
            "shopping": 0.01,
            "healthcare": 0.01,
            "other": 0.01,
        },
        notes=[
            "Travel-oriented option with better value when travel and dining are important.",
        ],
    ),
    CardProfile(
        name="Citi Custom Cash",
        issuer="Citi",
        annual_fee=0.0,
        rewards={
            "groceries": 0.05,
            "dining": 0.05,
            "gas": 0.05,
            "travel": 0.05,
            "entertainment": 0.05,
            "shopping": 0.05,
            "healthcare": 0.05,
            "other": 0.01,
        },
        notes=[
            "Works best when one spending category clearly dominates.",
        ],
    ),
    CardProfile(
        name="Capital One SavorOne",
        issuer="Capital One",
        annual_fee=0.0,
        rewards={
            "dining": 0.03,
            "entertainment": 0.03,
            "groceries": 0.03,
            "travel": 0.01,
            "gas": 0.01,
            "shopping": 0.01,
            "healthcare": 0.01,
            "other": 0.01,
        },
        notes=[
            "Balanced no-fee option for groceries, dining, and entertainment.",
        ],
    ),
    CardProfile(
        name="Wells Fargo Active Cash",
        issuer="Wells Fargo",
        annual_fee=0.0,
        rewards={
            "groceries": 0.02,
            "dining": 0.02,
            "travel": 0.02,
            "gas": 0.02,
            "entertainment": 0.02,
            "shopping": 0.02,
            "healthcare": 0.02,
            "other": 0.02,
        },
        notes=[
            "Flat-rate option that can be better if spending is spread across many categories.",
        ],
    ),
]


class CreditCardRecommender:
    """LLM-based control module for card recommendation."""

    def __init__(
        self,
        card_catalog: list[CardProfile] | None = None,
        model: str | None = None,
    ) -> None:
        load_dotenv(_find_dotenv_path(), override=False)
        self.card_catalog = card_catalog or DEFAULT_CARD_CATALOG
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY was not found in the environment or .env file.")

        self.client = OpenAI(api_key=api_key)

    def recommend_card(self, spending_profile: SpendingProfile) -> CardRecommendation:
        prompt = self._build_user_prompt(spending_profile)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        raw_content = response.choices[0].message.content or "{}"
        payload = json.loads(raw_content)
        return self._normalize_payload(payload)

    def _system_prompt(self) -> str:
        return (
            "You are a credit card recommendation agent for a class project. "
            "You receive a structured spending profile extracted from a receipt. "
            "Choose ONLY from the provided card catalog. "
            "Return valid JSON only. "
            "Be conservative and practical. "
            "Estimated annual value should be a reasonable NET dollar estimate after subtracting the annual fee when one exists. "
            "The primary recommendation must match the first item in card_rankings. "
            "The runner-up recommendation must match the second item in card_rankings. "
            "The JSON must include these top-level keys exactly: "
            "primary_recommendation, runner_up_recommendation, explanation, caveats, card_rankings."
        )

    def _build_user_prompt(self, spending_profile: SpendingProfile) -> str:
        catalog = [asdict(card) for card in self.card_catalog]
        return json.dumps(
            {
                "task": "Recommend the best credit card and a runner-up from the given catalog.",
                "instructions": [
                    "Use the spending profile as the main signal.",
                    "Use the reward rates and annual fees in the card catalog.",
                    "Prefer cards that realistically fit the spend pattern.",
                    "Treat estimated_annual_value as net value after annual fee, not gross rewards.",
                    "Return card_rankings as a list sorted from best to worst.",
                    "Make primary_recommendation equal to card_rankings[0].",
                    "Make runner_up_recommendation equal to card_rankings[1].",
                    "Each entry in card_rankings should include card_name, issuer, estimated_annual_value, and reason.",
                    "Keep caveats short and useful.",
                ],
                "spending_profile": spending_profile.as_dict(),
                "card_catalog": catalog,
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
                },
            },
            indent=2,
        )

    def _normalize_payload(self, payload: dict[str, object]) -> CardRecommendation:
        primary = payload.get("primary_recommendation", {})
        runner_up = payload.get("runner_up_recommendation", {})
        caveats = payload.get("caveats", [])
        rankings = payload.get("card_rankings", [])

        if not isinstance(primary, dict) or not primary:
            raise ValueError("LLM response missing primary_recommendation.")
        if not isinstance(runner_up, dict) or not runner_up:
            raise ValueError("LLM response missing runner_up_recommendation.")
        if not isinstance(caveats, list):
            caveats = []
        if not isinstance(rankings, list):
            rankings = []

        return CardRecommendation(
            primary_card=str(primary.get("card_name", "")),
            primary_issuer=str(primary.get("issuer", "")),
            primary_estimated_value=float(primary.get("estimated_annual_value", 0.0)),
            runner_up_card=str(runner_up.get("card_name", "")),
            runner_up_issuer=str(runner_up.get("issuer", "")),
            runner_up_estimated_value=float(runner_up.get("estimated_annual_value", 0.0)),
            explanation=str(payload.get("explanation", "")),
            caveats=[str(item) for item in caveats],
            card_rankings=[
                {
                    "card_name": str(item.get("card_name", "")),
                    "issuer": str(item.get("issuer", "")),
                    "estimated_annual_value": float(item.get("estimated_annual_value", 0.0)),
                    "reason": str(item.get("reason", "")),
                }
                for item in rankings
                if isinstance(item, dict)
            ],
        )


def _find_dotenv_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".env"
