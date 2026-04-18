from __future__ import annotations

from dataclasses import dataclass, field
import re


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
]


@dataclass
class LineItem:
    description: str
    amount: float
    category: str


@dataclass
class SpendingProfile:
    merchant: str | None
    category_totals: dict[str, float]
    line_items: list[LineItem] = field(default_factory=list)
    uncategorized_lines: list[str] = field(default_factory=list)

    @property
    def total_amount(self) -> float:
        return round(sum(self.category_totals.values()), 2)

    def as_dict(self) -> dict[str, object]:
        return {
            "merchant": self.merchant,
            "category_totals": self.category_totals,
            "total_amount": self.total_amount,
            "line_items": [
                {
                    "description": item.description,
                    "amount": item.amount,
                    "category": item.category,
                }
                for item in self.line_items
            ],
            "uncategorized_lines": self.uncategorized_lines,
        }


class ReceiptPlanner:
    """Simple planning module that converts OCR text into spending categories."""

    def __init__(
        self,
        merchant_lookup: dict[str, str] | None = None,
        category_keywords: dict[str, list[str]] | None = None,
    ) -> None:
        self.merchant_lookup = merchant_lookup or DEFAULT_MERCHANT_LOOKUP
        self.category_keywords = category_keywords or CATEGORY_KEYWORDS

    def build_spending_profile(self, receipt_text: str) -> SpendingProfile:
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
            category = self._categorize_line(description, merchant, default_category)

            if not description:
                uncategorized_lines.append(line)
                description = "unknown_item"

            line_items.append(
                LineItem(
                    description=description,
                    amount=amount,
                    category=category,
                )
            )
            category_totals[category] += amount

        rounded_totals = {
            category: round(total, 2) for category, total in category_totals.items()
        }
        return SpendingProfile(
            merchant=merchant,
            category_totals=rounded_totals,
            line_items=line_items,
            uncategorized_lines=uncategorized_lines,
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

    def _categorize_line(
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
