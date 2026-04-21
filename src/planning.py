from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None

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
    "total due",
    "tax",
    "vat",
    "balance",
    "change",
    "change due",
    "cash",
    "visa",
    "mastercard",
    "debit",
    "credit",
    "amount due",
    "tender",
    "tendered",
    "payment",
    "tip",
    "gratuity",
    "service charge",
    "discount",
    "rounding",
    "fee",
    "fees",
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
    DEFAULT_RECEIPT_PARSER_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        merchant_lookup: dict[str, str] | None = None,
        category_keywords: dict[str, list[str]] | None = None,
        default_version: str = "v1",
        v2_model_name_or_path: str | None = None,
        v2_score_threshold: float = 0.35,
        llm_receipt_parser_enabled: bool = True,
        llm_receipt_parser_model: str | None = None,
    ) -> None:
        if load_dotenv is not None:
            load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
        self.merchant_lookup = merchant_lookup or DEFAULT_MERCHANT_LOOKUP
        self.category_keywords = category_keywords or CATEGORY_KEYWORDS
        self.default_version = default_version
        self.v2_model_name_or_path = v2_model_name_or_path or self._default_v2_model()
        self.v2_score_threshold = v2_score_threshold
        self.llm_receipt_parser_enabled = llm_receipt_parser_enabled
        self.llm_receipt_parser_model = (
            llm_receipt_parser_model
            or os.getenv("OPENAI_RECEIPT_PARSER_MODEL")
            or os.getenv("OPENAI_MODEL")
            or self.DEFAULT_RECEIPT_PARSER_MODEL
        )
        self._llm_receipt_parser_client = None
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
        merchant, candidates, summary_lines, parser_metadata = self._candidate_lines_v2(
            receipt_text=receipt_text,
            lines=lines,
            merchant=merchant,
        )
        default_category = self._semantic_default_category(merchant)

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
                    **parser_metadata,
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

        summary_adjustment_metadata = self._apply_summary_line_adjustments(
            category_totals=category_totals,
            line_items=line_items,
            summary_lines=summary_lines,
            default_category=default_category,
        )

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
                **parser_metadata,
                **summary_adjustment_metadata,
            },
        )

    def _candidate_lines(self, lines: list[str]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            amount = self._extract_amount(line)
            if amount is None:
                continue

            description = self._strip_price_from_line(line)
            if description:
                if self._is_summary_line(description):
                    continue

                candidates.append(
                    {
                        "original_line": line,
                        "description": description,
                        "amount": amount,
                    }
                )
                continue

            paired_description = self._pair_amount_only_line(lines, index)
            if paired_description is None:
                continue

            candidates.append(
                {
                    "original_line": f"{paired_description} | {line}",
                    "description": paired_description,
                    "amount": amount,
                }
            )
        return candidates

    def _candidate_lines_v2(
        self,
        receipt_text: str,
        lines: list[str],
        merchant: str | None,
    ) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        parsed_receipt = self._parse_receipt_structure_with_llm(receipt_text, merchant)
        if parsed_receipt is not None and parsed_receipt["items"]:
            resolved_merchant = parsed_receipt["merchant"] or merchant
            return (
                resolved_merchant,
                parsed_receipt["items"],
                parsed_receipt["summary_lines"],
                {
                    "receipt_parser": "llm_structured_parser",
                    "receipt_parser_model": self.llm_receipt_parser_model,
                    "parsed_item_count": len(parsed_receipt["items"]),
                    "parsed_summary_lines": parsed_receipt["summary_lines"][:8],
                    "ignored_line_count": len(parsed_receipt["ignored_lines"]),
                },
            )

        heuristic_candidates = self._candidate_lines(lines)
        heuristic_summary_lines = self._extract_summary_lines_heuristic(lines)
        return (
            merchant,
            heuristic_candidates,
            heuristic_summary_lines,
            {
                "receipt_parser": "heuristic_regex_parser",
                "parsed_item_count": len(heuristic_candidates),
                "parsed_summary_lines": heuristic_summary_lines[:8],
                "ignored_line_count": 0,
            },
        )

    def _parse_receipt_structure_with_llm(
        self,
        receipt_text: str,
        merchant: str | None,
    ) -> dict[str, Any] | None:
        client = self._load_llm_receipt_parser_client()
        if client is None:
            return None

        trimmed_text = receipt_text.strip()[:12000]
        if not trimmed_text:
            return None

        try:
            response = client.chat.completions.create(
                model=self.llm_receipt_parser_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a receipt understanding parser. "
                            "Your job is to separate actual purchased items from summary lines such as subtotal, tax, tip, total, payment, and change. "
                            "OCR text may be noisy or split across lines. "
                            "Return JSON only with these keys exactly: merchant, purchase_items, summary_lines, ignored_lines. "
                            "purchase_items must include only true purchased items or services that should count toward spend. "
                            "summary_lines must include totals, subtotals, taxes, fees, discounts, tips, payment lines, and change lines. "
                            "If a line is clearly metadata such as date, time, cashier, table, receipt number, or phone number, put it in ignored_lines. "
                            "Amounts must be numbers, not strings."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "detected_merchant_hint": merchant,
                                "ocr_text": trimmed_text,
                                "output_schema": {
                                    "merchant": "string or null",
                                    "purchase_items": [
                                        {
                                            "description": "string",
                                            "amount": "number",
                                            "source_line": "string",
                                        }
                                    ],
                                    "summary_lines": [
                                        {
                                            "description": "string",
                                            "amount": "number or null",
                                            "line_type": "subtotal|tax|tip|discount|total|payment|change|other_summary",
                                            "source_line": "string",
                                        }
                                    ],
                                    "ignored_lines": ["string"],
                                },
                            },
                            indent=2,
                        ),
                    },
                ],
            )
        except Exception:
            return None

        content = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None

        return self._normalize_structured_receipt_payload(payload)

    def _normalize_structured_receipt_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        merchant = str(payload.get("merchant", "")).strip() or None

        purchase_items = payload.get("purchase_items", [])
        if not isinstance(purchase_items, list):
            purchase_items = []

        summary_lines = payload.get("summary_lines", [])
        if not isinstance(summary_lines, list):
            summary_lines = []

        ignored_lines = payload.get("ignored_lines", [])
        if not isinstance(ignored_lines, list):
            ignored_lines = []

        normalized_items: list[dict[str, Any]] = []
        for item in purchase_items:
            if not isinstance(item, dict):
                continue

            description = str(item.get("description", "")).strip()
            amount = self._coerce_amount(item.get("amount"))
            source_line = str(item.get("source_line", description)).strip() or description

            if not description or amount is None or amount <= 0:
                continue
            if self._is_summary_line(description):
                continue

            normalized_items.append(
                {
                    "original_line": source_line,
                    "description": description,
                    "amount": amount,
                }
            )

        normalized_summary_lines: list[dict[str, Any]] = []
        for line in summary_lines:
            if not isinstance(line, dict):
                continue
            normalized_summary_lines.append(
                {
                    "description": str(line.get("description", "")).strip(),
                    "amount": self._coerce_amount(line.get("amount")),
                    "line_type": str(line.get("line_type", "other_summary")).strip() or "other_summary",
                    "source_line": str(line.get("source_line", "")).strip(),
                }
            )

        return {
            "merchant": merchant,
            "items": normalized_items,
            "summary_lines": normalized_summary_lines,
            "ignored_lines": [str(line).strip() for line in ignored_lines if str(line).strip()],
        }

    def _apply_summary_line_adjustments(
        self,
        category_totals: dict[str, float],
        line_items: list[LineItem],
        summary_lines: list[dict[str, Any]],
        default_category: str | None,
    ) -> dict[str, Any]:
        applied_adjustments: list[dict[str, Any]] = []
        skipped_summary_lines: list[dict[str, Any]] = []

        for summary_line in summary_lines:
            description = str(summary_line.get("description", "")).strip() or "summary_line"
            amount = self._coerce_amount(summary_line.get("amount"))
            line_type = str(summary_line.get("line_type", "other_summary")).strip() or "other_summary"

            if amount is None:
                skipped_summary_lines.append(
                    {"description": description, "line_type": line_type, "reason": "missing_amount"}
                )
                continue

            if not self._summary_line_counts_toward_spend(line_type):
                skipped_summary_lines.append(
                    {"description": description, "line_type": line_type, "reason": "non_spend_summary"}
                )
                continue

            signed_amount = self._signed_summary_amount(amount, line_type, description)
            if signed_amount == 0:
                continue

            allocation = self._allocate_summary_amount(
                amount=signed_amount,
                category_totals=category_totals,
                default_category=default_category,
            )

            for index, (category, allocated_amount) in enumerate(allocation.items()):
                category_totals[category] += allocated_amount
                description_suffix = "allocated" if len(allocation) > 1 else "included"
                line_items.append(
                    LineItem(
                        description=f"{description} [{description_suffix}]",
                        amount=allocated_amount,
                        category=category,
                        score=None,
                    )
                )
                applied_adjustments.append(
                    {
                        "description": description,
                        "line_type": line_type,
                        "category": category,
                        "amount": allocated_amount,
                    }
                )

        return {
            "applied_summary_adjustments": applied_adjustments[:12],
            "skipped_summary_lines": skipped_summary_lines[:12],
        }

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

    def _load_llm_receipt_parser_client(self):
        if not self.llm_receipt_parser_enabled or OpenAI is None:
            return None

        if self._llm_receipt_parser_client is not None:
            return self._llm_receipt_parser_client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        self._llm_receipt_parser_client = OpenAI(api_key=api_key)
        return self._llm_receipt_parser_client

    def _default_v2_model(self) -> str:
        local_model_dir = Path(__file__).resolve().parent.parent / "models" / "planning_v2"
        if local_model_dir.exists() and any(local_model_dir.iterdir()):
            return str(local_model_dir)
        return self.DEFAULT_V2_MODEL

    def _extract_summary_lines_heuristic(self, lines: list[str]) -> list[dict[str, Any]]:
        summary_lines: list[dict[str, Any]] = []
        used_indices: set[int] = set()

        for index, line in enumerate(lines):
            if index in used_indices:
                continue

            stripped = line.strip()
            if not stripped:
                continue

            description = self._strip_price_from_line(stripped)
            amount = self._extract_amount(stripped)
            source_line = stripped

            if description and self._is_summary_line(description):
                if amount is None and index + 1 < len(lines) and self._is_amount_only_line(lines[index + 1]):
                    amount = self._extract_amount(lines[index + 1])
                    source_line = f"{stripped} | {lines[index + 1].strip()}"
                    used_indices.add(index + 1)

                summary_lines.append(
                    {
                        "description": description,
                        "amount": amount,
                        "line_type": self._classify_summary_line_type(description),
                        "source_line": source_line,
                    }
                )
                used_indices.add(index)

        return summary_lines

    def _summary_line_counts_toward_spend(self, line_type: str) -> bool:
        return line_type in {"tax", "tip", "discount", "fee", "other_summary"}

    def _signed_summary_amount(self, amount: float, line_type: str, description: str) -> float:
        normalized_description = description.lower()
        if line_type == "discount" or "discount" in normalized_description:
            return -abs(amount)
        return abs(amount)

    def _allocate_summary_amount(
        self,
        amount: float,
        category_totals: dict[str, float],
        default_category: str | None,
    ) -> dict[str, float]:
        positive_categories = {
            category: value
            for category, value in category_totals.items()
            if value > 0 and category != "other"
        }
        if not positive_categories:
            positive_categories = {
                category: value
                for category, value in category_totals.items()
                if value > 0
            }

        if positive_categories:
            total = sum(positive_categories.values())
            categories = list(positive_categories.items())
            allocation: dict[str, float] = {}
            running_total = 0.0
            for index, (category, value) in enumerate(categories):
                if index == len(categories) - 1:
                    allocated = round(amount - running_total, 2)
                else:
                    allocated = round(amount * (value / total), 2)
                    running_total += allocated
                allocation[category] = allocated
            return allocation

        target_category = default_category or "other"
        return {target_category: round(amount, 2)}

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

    def _coerce_amount(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return round(float(value), 2)

        cleaned = str(value).strip()
        if not cleaned:
            return None

        cleaned = cleaned.replace("$", "").replace(",", "")
        cleaned = re.sub(r"[^\d.\-]", "", cleaned)
        if cleaned.count(".") > 1:
            return None

        try:
            return round(float(cleaned), 2)
        except ValueError:
            return None

    def _strip_price_from_line(self, line: str) -> str:
        cleaned = PRICE_PATTERN.sub("", line)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" -:$")

    def _is_summary_line(self, description: str) -> bool:
        normalized = description.lower()
        return any(keyword in normalized for keyword in SUMMARY_LINE_KEYWORDS)

    def _classify_summary_line_type(self, description: str) -> str:
        normalized = description.lower()
        if "subtotal" in normalized or "sub total" in normalized:
            return "subtotal"
        if "tax" in normalized or "vat" in normalized:
            return "tax"
        if "tip" in normalized or "gratuity" in normalized:
            return "tip"
        if "discount" in normalized:
            return "discount"
        if "service charge" in normalized or "rounding" in normalized or "fee" in normalized:
            return "fee"
        if "change" in normalized:
            return "change"
        if any(keyword in normalized for keyword in ["visa", "mastercard", "debit", "credit", "cash", "payment", "tender"]):
            return "payment"
        if "total" in normalized or "balance" in normalized or "amount due" in normalized:
            return "total"
        return "other_summary"

    def _pair_amount_only_line(self, lines: list[str], index: int) -> str | None:
        current_line = lines[index]
        if not self._is_amount_only_line(current_line):
            return None

        search_start = max(0, index - 2)
        for candidate_index in range(index - 1, search_start - 1, -1):
            candidate_line = lines[candidate_index].strip()
            if not candidate_line:
                continue
            if self._extract_amount(candidate_line) is not None:
                continue
            if self._is_summary_line(candidate_line):
                return None
            if self._looks_like_item_description(candidate_line):
                return candidate_line
        return None

    def _is_amount_only_line(self, line: str) -> bool:
        normalized = PRICE_PATTERN.sub("", line)
        normalized = re.sub(r"[\s:$\-]", "", normalized)
        return bool(normalized == "")

    def _looks_like_item_description(self, text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return False
        if self._is_summary_line(normalized):
            return False
        if re.search(r"\b(date|time|receipt|check|order|invoice|table|server|cashier|phone|tel)\b", normalized):
            return False
        return bool(re.search(r"[a-zA-Z]", normalized))
