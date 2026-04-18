from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - optional dependency at runtime
    pipeline = None


CATEGORIES = [
    "groceries",
    "dining",
    "travel",
    "gas",
    "entertainment",
    "shopping",
    "healthcare",
    "other",
]


DEFAULT_MERCHANT_LOOKUP = {
    "whole foods": "groceries",
    "trader joe": "groceries",
    "costco": "groceries",
    "safeway": "groceries",
    "kroger": "groceries",
    "publix": "groceries",
    "target": "shopping",
    "walmart": "shopping",
    "amazon": "shopping",
    "macy": "shopping",
    "starbucks": "dining",
    "mcdonald": "dining",
    "chipotle": "dining",
    "doordash": "dining",
    "uber eats": "dining",
    "shell": "gas",
    "chevron": "gas",
    "exxon": "gas",
    "delta": "travel",
    "united": "travel",
    "airbnb": "travel",
    "marriott": "travel",
    "netflix": "entertainment",
    "spotify": "entertainment",
    "walgreens": "healthcare",
    "cvs": "healthcare",
    "rite aid": "healthcare",
}


CATEGORY_KEYWORDS = {
    "groceries": [
        "milk",
        "bread",
        "eggs",
        "banana",
        "produce",
        "grocery",
        "vegetable",
        "fruit",
        "meat",
        "cheese",
    ],
    "dining": [
        "burger",
        "pizza",
        "coffee",
        "latte",
        "restaurant",
        "sandwich",
        "fries",
        "taco",
        "meal",
        "cafe",
    ],
    "travel": [
        "hotel",
        "flight",
        "airlines",
        "airbnb",
        "uber",
        "lyft",
        "rental",
        "booking",
        "trip",
    ],
    "gas": [
        "fuel",
        "gas",
        "diesel",
        "unleaded",
        "gallon",
        "pump",
        "ev charging",
    ],
    "entertainment": [
        "movie",
        "cinema",
        "theater",
        "concert",
        "streaming",
        "game",
        "ticket",
    ],
    "shopping": [
        "shirt",
        "shoes",
        "apparel",
        "clothing",
        "electronics",
        "retail",
        "housewares",
        "merchandise",
    ],
    "healthcare": [
        "pharmacy",
        "prescription",
        "medicine",
        "vitamin",
        "clinic",
        "medical",
        "doctor",
    ],
}


CATEGORY_DESCRIPTIONS = {
    "groceries": "food staples, supermarket items, household groceries, produce, packaged food",
    "dining": "restaurants, cafes, coffee shops, takeout, fast food, prepared meals",
    "travel": "airfare, hotels, rideshare, car rentals, transit, travel bookings",
    "gas": "fuel, gas stations, diesel, pump purchases, charging stops",
    "entertainment": "movies, games, concerts, streaming, events, tickets",
    "shopping": "general retail, clothing, electronics, home goods, merchandise",
    "healthcare": "pharmacy, medicine, clinic, medical products, wellness items",
}


PRICE_PATTERN = re.compile(r"(?<!\d)(\d{1,4}\.\d{2})(?!\d)")
SUMMARY_LINE_KEYWORDS = [
    "subtotal",
    "sub total",
    "total",
    "tax",
    "balance",
    "change",
    "cash",
    "visa",
    "mastercard",
    "debit",
    "credit",
    "amount due",
    "tender",
    "payment",
]


@dataclass
class LineItem:
    description: str
    amount: float
    category: str
    score: float | None = None


@dataclass
class SpendingProfile:
    merchant: str | None
    category_totals: dict[str, float]
    line_items: list[LineItem] = field(default_factory=list)
    uncategorized_lines: list[str] = field(default_factory=list)
    planner_version: str = "v1"
    planner_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_amount(self) -> float:
        return round(sum(self.category_totals.values()), 2)

    def as_dict(self) -> dict[str, object]:
        return {
            "merchant": self.merchant,
            "planner_version": self.planner_version,
            "planner_metadata": self.planner_metadata,
            "category_totals": self.category_totals,
            "total_amount": self.total_amount,
            "line_items": [
                {
                    "description": item.description,
                    "amount": item.amount,
                    "category": item.category,
                    "score": item.score,
                }
                for item in self.line_items
            ],
            "uncategorized_lines": self.uncategorized_lines,
        }


class ReceiptPlanner:
    """Planning module with Version 1 classical logic and Version 2 DL classification."""

    DEFAULT_V2_MODEL = "typeform/distilbert-base-uncased-mnli"

    def __init__(
        self,
        merchant_lookup: dict[str, str] | None = None,
        category_keywords: dict[str, list[str]] | None = None,
        default_version: str = "v1",
        v2_model_name_or_path: str | None = None,
        v2_score_threshold: float = 0.35,
    ) -> None:
        self.merchant_lookup = merchant_lookup or DEFAULT_MERCHANT_LOOKUP
        self.category_keywords = category_keywords or CATEGORY_KEYWORDS
        self.default_version = default_version
        self.v2_model_name_or_path = v2_model_name_or_path or self._default_v2_model()
        self.v2_score_threshold = v2_score_threshold
        self._v2_classifier = None

    def build_spending_profile(
        self,
        receipt_text: str,
        version: str | None = None,
    ) -> SpendingProfile:
        selected_version = (version or self.default_version).lower()
        if selected_version == "v1":
            return self.build_spending_profile_v1(receipt_text)
        if selected_version == "v2":
            return self.build_spending_profile_v2(receipt_text)
        raise ValueError("planning version must be 'v1' or 'v2'")

    def build_spending_profile_v1(self, receipt_text: str) -> SpendingProfile:
        lines = self._normalize_lines(receipt_text)
        merchant = self._detect_merchant(lines)
        default_category = self._lookup_merchant_category(merchant) if merchant else None

        category_totals = {category: 0.0 for category in CATEGORIES}
        line_items: list[LineItem] = []
        uncategorized_lines: list[str] = []

        for line in lines:
            amount = self._extract_amount(line)
            if amount is None:
                continue

            description = self._strip_price_from_line(line)
            if self._is_summary_line(description):
                continue

            category = self._categorize_line_v1(description, merchant, default_category)

            if not description:
                uncategorized_lines.append(line)
                description = "unknown_item"

            line_items.append(
                LineItem(
                    description=description,
                    amount=amount,
                    category=category,
                    score=None,
                )
            )
            category_totals[category] += amount

        return self._build_profile(
            merchant=merchant,
            category_totals=category_totals,
            line_items=line_items,
            uncategorized_lines=uncategorized_lines,
            planner_version="v1",
            planner_metadata={
                "classification_method": "merchant_lookup_and_keyword_rules",
            },
        )

    def build_spending_profile_v2(self, receipt_text: str) -> SpendingProfile:
        lines = self._normalize_lines(receipt_text)
        merchant = self._detect_merchant(lines)
        default_category = self._semantic_default_category(merchant)
        candidates = self._candidate_lines(lines)

        category_totals = {category: 0.0 for category in CATEGORIES}
        line_items: list[LineItem] = []
        uncategorized_lines: list[str] = []

        if not candidates:
            return self._build_profile(
                merchant=merchant,
                category_totals=category_totals,
                line_items=[],
                uncategorized_lines=[],
                planner_version="v2",
                planner_metadata={
                    "classification_method": "transformer_semantic_classifier",
                    "model_name_or_path": self.v2_model_name_or_path,
                    "score_threshold": self.v2_score_threshold,
                },
            )

        predictions = self._classify_candidate_lines_v2(candidates, merchant, default_category)

        for candidate, prediction in zip(candidates, predictions):
            description = candidate["description"]
            if not description:
                uncategorized_lines.append(candidate["original_line"])
                description = "unknown_item"

            category = prediction["category"]
            score = prediction["score"]

            line_items.append(
                LineItem(
                    description=description,
                    amount=candidate["amount"],
                    category=category,
                    score=score,
                )
            )
            category_totals[category] += candidate["amount"]

        return self._build_profile(
            merchant=merchant,
            category_totals=category_totals,
            line_items=line_items,
            uncategorized_lines=uncategorized_lines,
            planner_version="v2",
            planner_metadata={
                "classification_method": "transformer_semantic_classifier",
                "model_name_or_path": self.v2_model_name_or_path,
                "score_threshold": self.v2_score_threshold,
                "merchant_default_category": default_category,
            },
        )

    def _candidate_lines(self, lines: list[str]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for line in lines:
            amount = self._extract_amount(line)
            if amount is None:
                continue

            description = self._strip_price_from_line(line)
            if self._is_summary_line(description):
                continue

            candidates.append(
                {
                    "original_line": line,
                    "description": description,
                    "amount": amount,
                }
            )
        return candidates

    def _classify_candidate_lines_v2(
        self,
        candidates: list[dict[str, Any]],
        merchant: str | None,
        default_category: str | None,
    ) -> list[dict[str, Any]]:
        classifier = self._load_v2_classifier()
        texts = [self._classification_text(item["description"], merchant) for item in candidates]
        labels = [category for category in CATEGORIES if category != "other"]

        raw_results = classifier(
            texts,
            candidate_labels=labels,
            hypothesis_template="This transaction belongs to the {} category.",
            multi_label=False,
        )
        if isinstance(raw_results, dict):
            raw_results = [raw_results]

        predictions: list[dict[str, Any]] = []
        for result, candidate in zip(raw_results, candidates):
            ordered_labels = [str(label).lower() for label in result.get("labels", [])]
            ordered_scores = [float(score) for score in result.get("scores", [])]

            if ordered_labels:
                category = ordered_labels[0]
                score = ordered_scores[0] if ordered_scores else 0.0
            else:
                category = default_category or "other"
                score = 0.0

            if score < self.v2_score_threshold:
                fallback = default_category or self._lookup_merchant_category(merchant) or "other"
                category = fallback

            if not candidate["description"]:
                category = default_category or "other"

            predictions.append(
                {
                    "category": category,
                    "score": round(score, 4),
                }
            )

        return predictions

    def _semantic_default_category(self, merchant: str | None) -> str | None:
        if not merchant:
            return None

        merchant_lookup_category = self._lookup_merchant_category(merchant)
        if merchant_lookup_category is not None:
            return merchant_lookup_category

        classifier = self._load_v2_classifier()
        labels = [category for category in CATEGORIES if category != "other"]
        result = classifier(
            merchant,
            candidate_labels=labels,
            hypothesis_template="This merchant most often belongs to the {} category.",
            multi_label=False,
        )

        ordered_labels = [str(label).lower() for label in result.get("labels", [])]
        ordered_scores = [float(score) for score in result.get("scores", [])]
        if not ordered_labels:
            return None

        if ordered_scores and ordered_scores[0] >= self.v2_score_threshold:
            return ordered_labels[0]
        return None

    def _load_v2_classifier(self):
        if self._v2_classifier is not None:
            return self._v2_classifier

        if pipeline is None:
            raise ImportError(
                "Transformers is required for planning version 2. "
                "Install transformers and torch to use the DL planning pipeline."
            )

        self._v2_classifier = pipeline(
            "zero-shot-classification",
            model=self.v2_model_name_or_path,
        )
        return self._v2_classifier

    def _default_v2_model(self) -> str:
        local_model_dir = Path(__file__).resolve().parent.parent / "models" / "planning_v2"
        if local_model_dir.exists() and any(local_model_dir.iterdir()):
            return str(local_model_dir)
        return self.DEFAULT_V2_MODEL

    def _build_profile(
        self,
        merchant: str | None,
        category_totals: dict[str, float],
        line_items: list[LineItem],
        uncategorized_lines: list[str],
        planner_version: str,
        planner_metadata: dict[str, Any],
    ) -> SpendingProfile:
        rounded_totals = {
            category: round(total, 2) for category, total in category_totals.items()
        }
        return SpendingProfile(
            merchant=merchant,
            category_totals=rounded_totals,
            line_items=line_items,
            uncategorized_lines=uncategorized_lines,
            planner_version=planner_version,
            planner_metadata=planner_metadata,
        )

    def _normalize_lines(self, receipt_text: str) -> list[str]:
        return [line.strip() for line in receipt_text.splitlines() if line.strip()]

    def _detect_merchant(self, lines: list[str]) -> str | None:
        if not lines:
            return None

        candidate_window = lines[:5]
        for line in candidate_window:
            normalized = line.lower()
            for merchant_key in self.merchant_lookup:
                if merchant_key in normalized:
                    return line

        return candidate_window[0]

    def _lookup_merchant_category(self, merchant: str | None) -> str | None:
        if not merchant:
            return None

        normalized = merchant.lower()
        for merchant_key, category in self.merchant_lookup.items():
            if merchant_key in normalized:
                return category
        return None

    def _categorize_line_v1(
        self,
        description: str,
        merchant: str | None,
        default_category: str | None,
    ) -> str:
        text = f"{merchant or ''} {description}".lower()

        for merchant_key, category in self.merchant_lookup.items():
            if merchant_key in text:
                return category

        category_scores = {category: 0 for category in CATEGORIES}
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    category_scores[category] += 1

        best_category = max(category_scores, key=category_scores.get)
        if category_scores[best_category] > 0:
            return best_category

        if default_category is not None:
            return default_category
        return "other"

    def _classification_text(self, description: str, merchant: str | None) -> str:
        merchant_text = merchant or "unknown merchant"
        category_guide = "; ".join(
            f"{name}: {description_text}"
            for name, description_text in CATEGORY_DESCRIPTIONS.items()
        )
        return (
            f"Merchant: {merchant_text}\n"
            f"Transaction line: {description}\n"
            f"Possible categories: {category_guide}"
        )

    def _extract_amount(self, line: str) -> float | None:
        matches = PRICE_PATTERN.findall(line)
        if not matches:
            return None

        value = matches[-1]
        try:
            return float(value)
        except ValueError:
            return None

    def _strip_price_from_line(self, line: str) -> str:
        cleaned = PRICE_PATTERN.sub("", line)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" -:$")

    def _is_summary_line(self, description: str) -> bool:
        normalized = description.lower()
        return any(keyword in normalized for keyword in SUMMARY_LINE_KEYWORDS)
