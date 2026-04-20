from __future__ import annotations

import argparse
from pathlib import Path

try:
    from perception import ReceiptPerception
    from planning import ReceiptPlanner
    from utils import list_receipt_images, preferred_receipts_dir
except ImportError:  # pragma: no cover
    from src.perception import ReceiptPerception
    from src.planning import ReceiptPlanner
    from src.utils import list_receipt_images, preferred_receipts_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the receipt pipeline on a few receipt images from a folder."
    )
    parser.add_argument(
        "--receipts-dir",
        default=None,
        help="Directory containing receipt images.",
    )
    parser.add_argument(
        "--pipeline-version",
        choices=["v1", "v2"],
        default=None,
        help="Convenience preset. v1 = tesseract + planning v1. v2 = paddleocr + planning v2.",
    )
    parser.add_argument(
        "--ocr-method",
        choices=["tesseract", "trocr", "paddleocr", "labels"],
        default=None,
        help="Perception method to use. Overrides the pipeline preset if provided.",
    )
    parser.add_argument(
        "--planning-version",
        choices=["v1", "v2"],
        default=None,
        help="Planning version to use. Overrides the pipeline preset if provided.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of receipts to test.",
    )
    return parser.parse_args()


def resolve_pipeline_settings(args: argparse.Namespace) -> tuple[str, str]:
    if args.pipeline_version == "v1":
        perception_method = args.ocr_method or "tesseract"
        planning_version = args.planning_version or "v1"
    elif args.pipeline_version == "v2":
        perception_method = args.ocr_method or "paddleocr"
        planning_version = args.planning_version or "v2"
    else:
        perception_method = args.ocr_method or "tesseract"
        planning_version = args.planning_version or "v1"

    return perception_method, planning_version


def main() -> None:
    args = parse_args()
    receipts_dir = Path(args.receipts_dir) if args.receipts_dir else preferred_receipts_dir()
    if not receipts_dir.exists():
        raise FileNotFoundError(f"Receipts directory not found: {receipts_dir}")

    receipt_paths = list_receipt_images(receipts_dir)[: args.limit]
    if not receipt_paths:
        raise FileNotFoundError(f"No receipt images found in: {receipts_dir}")

    perception_method, planning_version = resolve_pipeline_settings(args)
    perception = ReceiptPerception()
    planner = ReceiptPlanner(default_version=planning_version)

    for receipt_path in receipt_paths:
        print(f"\n{'=' * 80}")
        print(f"Receipt: {receipt_path.name}")
        print(f"Pipeline: perception={perception_method}, planning={planning_version}")

        ocr_result = perception.extract_text(receipt_path, method=perception_method)
        spending_profile = planner.build_spending_profile(
            ocr_result.text,
            version=planning_version,
        )

        print("\nOCR preview:")
        preview_lines = ocr_result.text.splitlines()[:12]
        print("\n".join(preview_lines) if preview_lines else "[No text extracted]")

        print("\nCategory totals:")
        for category, amount in spending_profile.category_totals.items():
            if amount > 0:
                print(f"  {category}: {amount:.2f}")

        print("\nParsed line items:")
        for item in spending_profile.line_items[:8]:
            score_text = f" [score={item.score:.2f}]" if item.score is not None else ""
            print(f"  {item.description} -> {item.category} (${item.amount:.2f}){score_text}")

        if not spending_profile.line_items:
            print("  [No line items parsed]")


if __name__ == "__main__":
    main()
