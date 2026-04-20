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
        description="Run the receipt pipeline on a single receipt image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the receipt image file.",
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
        "--compare-ocr",
        action="store_true",
        help="Run Tesseract, TrOCR, and PaddleOCR side by side on the same image.",
    )
    parser.add_argument(
        "--planning-version",
        choices=["v1", "v2"],
        default=None,
        help="Planning version to use. Overrides the pipeline preset if provided.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save the combined perception + planning + control output as JSON.",
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
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    perception = ReceiptPerception()
    if args.compare_ocr:
        comparison_methods = ["tesseract", "trocr", "paddleocr"]
        comparison_output: dict[str, object] = {
            "image_path": str(image_path),
            "results": {},
        }

        print("\n=== OCR Comparison ===")
        for method in comparison_methods:
            try:
                result = perception.extract_text(image_path=image_path, method=method)
                comparison_output["results"][method] = {
                    "method": result.method,
                    "confidence": result.confidence,
                    "metadata": result.metadata,
                    "text": result.text,
                }
                print(f"\n--- {method.upper()} ---")
                print(f"Confidence: {result.confidence}")
                print(result.text if result.text else "[No text extracted]")
            except Exception as exc:
                comparison_output["results"][method] = {"error": str(exc)}
                print(f"\n--- {method.upper()} ---")
                print(f"Failed: {exc}")

        if args.save_json:
            output_path = Path(args.save_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(comparison_output, indent=2))
            print(f"\nSaved output to: {output_path}")
        return

    perception_method, planning_version = resolve_pipeline_settings(args)
    planner = ReceiptPlanner(default_version=planning_version)
    recommender = CreditCardRecommender()

    ocr_result = perception.extract_text(image_path=image_path, method=perception_method)
    spending_profile = planner.build_spending_profile(
        ocr_result.text,
        version=planning_version,
    )
    recommendation = recommender.recommend_card(spending_profile)

    output = {
        "pipeline": {
            "perception_method": perception_method,
            "planning_version": planning_version,
        },
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

    print("\n=== Pipeline Settings ===")
    print(json.dumps(output["pipeline"], indent=2))

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
