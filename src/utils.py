from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from functools import lru_cache

try:
    from control import CreditCardRecommender
    from perception import OCRResult, ReceiptPerception
    from planning import CATEGORIES, ReceiptPlanner, SpendingProfile
except ImportError:  # pragma: no cover
    from src.control import CreditCardRecommender
    from src.perception import OCRResult, ReceiptPerception
    from src.planning import CATEGORIES, ReceiptPlanner, SpendingProfile


def resolve_pipeline_settings(
    pipeline_version: str | None = None,
    ocr_method: str | None = None,
    planning_version: str | None = None,
) -> tuple[str, str]:
    if pipeline_version == "v1":
        perception_method = ocr_method or "tesseract"
        selected_planning_version = planning_version or "v1"
    elif pipeline_version == "v2":
        perception_method = ocr_method or "paddleocr"
        selected_planning_version = planning_version or "v2"
    else:
        perception_method = ocr_method or "tesseract"
        selected_planning_version = planning_version or "v1"

    return perception_method, selected_planning_version


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def list_receipt_images(receipts_dir: str | Path) -> list[Path]:
    receipts_dir = Path(receipts_dir)
    patterns = ("*.png", "*.jpg", "*.jpeg")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(receipts_dir.glob(pattern)))
    return sorted(paths)


def preferred_receipts_dir() -> Path:
    root = project_root()
    candidate_dirs = [
        root / "data" / "receipt_dataset" / "ds0" / "img",
        root / "data" / "receipts_nano",
        root / "data" / "receipts",
    ]
    for candidate_dir in candidate_dirs:
        if candidate_dir.exists() and list_receipt_images(candidate_dir):
            return candidate_dir
    return candidate_dirs[0]


def preferred_labeled_receipts_dir() -> Path:
    root = project_root()
    candidate_dirs = [
        root / "data" / "receipt_dataset" / "ds0" / "img",
        root / "data" / "receipts",
    ]
    for candidate_dir in candidate_dirs:
        if candidate_dir.exists() and list_receipt_images(candidate_dir):
            return candidate_dir
    return candidate_dirs[0]


def labeled_receipt_dirs() -> list[Path]:
    root = project_root()
    return [
        root / "data" / "receipt_dataset" / "ds0" / "img",
        root / "data" / "receipts",
    ]


def labeled_receipt_images() -> list[Path]:
    paths: list[Path] = []
    for candidate_dir in labeled_receipt_dirs():
        if candidate_dir.exists():
            paths.extend(list_receipt_images(candidate_dir))
    deduplicated: dict[str, Path] = {str(path.resolve()): path for path in paths}
    return list(deduplicated.values())


def has_reference_labels(image_path: str | Path) -> bool:
    image_path = Path(image_path)
    root = project_root()
    local_json_candidates = [
        root / "data" / "receipt_dataset" / "ds0" / "ann" / f"{image_path.name}.json",
        root / "data" / "receipt_dataset" / "ds0" / "ann" / f"{image_path.stem}.json",
    ]
    if any(candidate.exists() for candidate in local_json_candidates):
        return True

    sroie_labels = root / "data" / "labels" / "sroie_labels.json"
    if not sroie_labels.exists():
        return False

    try:
        import json

        index = json.loads(sroie_labels.read_text())
    except Exception:
        return False

    return image_path.stem in index


@lru_cache(maxsize=1)
def _labeled_receipt_hash_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for image_path in labeled_receipt_images():
        try:
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        except OSError:
            continue
        index[digest] = str(image_path)
    return index


def resolve_reference_labeled_image(
    image_path: str | Path | None = None,
    file_name: str | None = None,
    file_bytes: bytes | None = None,
) -> Path | None:
    if image_path is not None:
        candidate = Path(image_path)
        if candidate.exists() and has_reference_labels(candidate):
            return candidate

    if file_bytes:
        digest = hashlib.sha256(file_bytes).hexdigest()
        indexed_path = _labeled_receipt_hash_index().get(digest)
        if indexed_path is not None:
            candidate = Path(indexed_path)
            if candidate.exists() and has_reference_labels(candidate):
                return candidate

    if file_name:
        for candidate_dir in labeled_receipt_dirs():
            exact_match = candidate_dir / Path(file_name).name
            if exact_match.exists() and has_reference_labels(exact_match):
                return exact_match

    return None


def run_receipt_pipeline(
    image_path: str | Path,
    perception: ReceiptPerception | None = None,
    planner: ReceiptPlanner | None = None,
    recommender: CreditCardRecommender | None = None,
    pipeline_version: str | None = None,
    ocr_method: str | None = None,
    planning_version: str | None = None,
    run_control: bool = True,
) -> dict[str, Any]:
    perception_method, selected_planning_version = resolve_pipeline_settings(
        pipeline_version=pipeline_version,
        ocr_method=ocr_method,
        planning_version=planning_version,
    )

    perception = perception or ReceiptPerception()
    planner = planner or ReceiptPlanner(default_version=selected_planning_version)

    ocr_result = perception.extract_text(image_path=image_path, method=perception_method)
    spending_profile = planner.build_spending_profile(
        ocr_result.text,
        version=selected_planning_version,
    )

    recommendation = None
    if run_control:
        recommender = recommender or CreditCardRecommender()
        recommendation = recommender.recommend_card(spending_profile)

    return {
        "pipeline": {
            "perception_method": perception_method,
            "planning_version": selected_planning_version,
        },
        "ocr_result": ocr_result,
        "spending_profile": spending_profile,
        "recommendation": recommendation,
    }


def merge_spending_profiles(
    profiles: list[SpendingProfile],
    source_names: list[str] | None = None,
) -> SpendingProfile:
    category_totals = {category: 0.0 for category in CATEGORIES}
    line_items = []
    uncategorized_lines: list[str] = []
    merchants: list[str] = []

    for profile in profiles:
        if profile.merchant:
            merchants.append(profile.merchant)

        for category, amount in profile.category_totals.items():
            category_totals[category] += amount

        line_items.extend(profile.line_items)
        uncategorized_lines.extend(profile.uncategorized_lines)

    distinct_merchants = sorted(set(merchants))
    if not distinct_merchants:
        merchant = None
    elif len(distinct_merchants) == 1:
        merchant = distinct_merchants[0]
    else:
        merchant = "Multiple merchants"

    planner_versions = sorted({profile.planner_version for profile in profiles if profile.planner_version})
    merged_metadata = {
        "source_receipt_count": len(profiles),
        "source_receipts": source_names or [],
        "source_planner_versions": planner_versions,
    }

    rounded_totals = {
        category: round(amount, 2)
        for category, amount in category_totals.items()
    }

    return SpendingProfile(
        merchant=merchant,
        category_totals=rounded_totals,
        line_items=line_items,
        uncategorized_lines=uncategorized_lines,
        planner_version="aggregate",
        planner_metadata=merged_metadata,
    )


def save_uploaded_bytes(file_name: str, file_bytes: bytes, suffix: str | None = None) -> Path:
    effective_suffix = suffix or Path(file_name).suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=effective_suffix) as handle:
        handle.write(file_bytes)
        return Path(handle.name)


def preview_text(text: str, limit: int = 1400) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "\n..."


def ocr_result_as_dict(result: OCRResult) -> dict[str, Any]:
    return {
        "method": result.method,
        "image_path": result.image_path,
        "text": result.text,
        "raw_text": result.raw_text,
        "confidence": result.confidence,
        "metadata": result.metadata,
    }
