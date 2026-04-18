from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from control import CreditCardRecommender
    from perception import ReceiptPerception
    from planning import ReceiptPlanner
except ImportError:  # pragma: no cover
    from src.control import CreditCardRecommender
    from src.perception import ReceiptPerception
    from src.planning import ReceiptPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run receipt OCR and planning on a single receipt image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the receipt image file.",
    )
    parser.add_argument(
        "--ocr-method",
        choices=["tesseract", "trocr", "labels"],
        default="tesseract",
        help="OCR method to use.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save the combined OCR + planning output as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    perception = ReceiptPerception()
    planner = ReceiptPlanner()
    recommender = CreditCardRecommender()

    ocr_result = perception.extract_text(image_path=image_path, method=args.ocr_method)
    spending_profile = planner.build_spending_profile(ocr_result.text)
    recommendation = recommender.recommend_card(spending_profile)

    output = {
        "ocr": {
            "method": ocr_result.method,
            "image_path": ocr_result.image_path,
            "confidence": ocr_result.confidence,
            "metadata": ocr_result.metadata,
            "text": ocr_result.text,
        },
        "planning": spending_profile.as_dict(),
        "control": recommendation.as_dict(),
    }

    print("\n=== OCR Text ===")
    print(ocr_result.text if ocr_result.text else "[No text extracted]")

    print("\n=== Spending Profile ===")
    print(json.dumps(spending_profile.as_dict(), indent=2))

    print("\n=== Recommendation ===")
    print(json.dumps(recommendation.as_dict(), indent=2))

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    main()
