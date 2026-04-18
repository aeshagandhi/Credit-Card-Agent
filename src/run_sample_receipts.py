from __future__ import annotations

import argparse
from pathlib import Path

try:
    from perception import ReceiptPerception
    from planning import ReceiptPlanner
except ImportError:  # pragma: no cover
    from src.perception import ReceiptPerception
    from src.planning import ReceiptPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR + planning on a few receipt images from a folder."
    )
    parser.add_argument(
        "--receipts-dir",
        default="data/receipts",
        help="Directory containing receipt images.",
    )
    parser.add_argument(
        "--ocr-method",
        choices=["tesseract", "trocr", "labels"],
        default="tesseract",
        help="OCR method to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of receipts to test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipts_dir = Path(args.receipts_dir)
    if not receipts_dir.exists():
        raise FileNotFoundError(f"Receipts directory not found: {receipts_dir}")

    receipt_paths = sorted(receipts_dir.glob("*.jpg"))[: args.limit]
    if not receipt_paths:
        raise FileNotFoundError(f"No .jpg files found in: {receipts_dir}")

    perception = ReceiptPerception()
    planner = ReceiptPlanner()

    for receipt_path in receipt_paths:
        print(f"\n{'=' * 80}")
        print(f"Receipt: {receipt_path.name}")

        ocr_result = perception.extract_text(receipt_path, method=args.ocr_method)
        spending_profile = planner.build_spending_profile(ocr_result.text)

        print("\nOCR preview:")
        preview_lines = ocr_result.text.splitlines()[:12]
        print("\n".join(preview_lines) if preview_lines else "[No text extracted]")

        print("\nCategory totals:")
        for category, amount in spending_profile.category_totals.items():
            if amount > 0:
                print(f"  {category}: {amount:.2f}")

        print("\nParsed line items:")
        for item in spending_profile.line_items[:8]:
            print(f"  {item.description} -> {item.category} (${item.amount:.2f})")

        if not spending_profile.line_items:
            print("  [No line items parsed]")


if __name__ == "__main__":
    main()
