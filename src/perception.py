from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
import pytesseract

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional dependency at runtime
    PaddleOCR = None


@dataclass
class OCRResult:
    method: str
    image_path: str
    text: str
    raw_text: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ReceiptPerception:
    """Receipt OCR wrapper for the three supported perception paths."""

    def __init__(self, tesseract_psm: int = 6) -> None:
        self.tesseract_psm = tesseract_psm
        self._paddle_ocr = None

    def preprocess_receipt(self, image_path: str | Path) -> np.ndarray:
        """Apply lightweight preprocessing for the classical OCR path."""
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        angle = self._estimate_skew_angle(binary)
        if abs(angle) > 0.5:
            return self._rotate_image(binary, angle)
        return binary

    def run_tesseract_ocr(self, image_path: str | Path) -> OCRResult:
        """Classical OCR path using Tesseract on a preprocessed image."""
        processed = self.preprocess_receipt(image_path)
        config = f"--oem 3 --psm {self.tesseract_psm}"
        raw_text = pytesseract.image_to_string(processed, config=config)
        confidence = self._mean_tesseract_confidence(processed, config)

        return OCRResult(
            method="tesseract",
            image_path=str(image_path),
            text=self._clean_text(raw_text),
            raw_text=raw_text,
            confidence=confidence,
            metadata={
                "psm": self.tesseract_psm,
                "preprocessing": [
                    "grayscale",
                    "gaussian_blur",
                    "adaptive_threshold",
                    "deskew",
                ],
            },
        )

    def run_paddleocr(self, image_path: str | Path) -> OCRResult:
        """Deep-learning OCR path using PaddleOCR on the original image."""
        if PaddleOCR is None:
            raise ImportError(
                "PaddleOCR dependencies are not available. Install paddleocr "
                "and the appropriate paddle runtime for your platform to use PaddleOCR."
            )

        try:
            self._load_paddle_ocr()
            results = self._paddle_ocr.predict(str(image_path))
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR could not initialize or run. If this is the first time using it, "
                "the model weights may still need to download on a machine with network access. "
                f"Original error: {exc}"
            ) from exc

        lines: list[str] = []
        confidences: list[float] = []
        if results:
            for page in results:
                rec_texts = page.get("rec_texts", [])
                rec_scores = page.get("rec_scores", [])
                for text, confidence in zip(rec_texts, rec_scores):
                    cleaned = str(text).strip()
                    if cleaned:
                        lines.append(cleaned)
                        confidences.append(float(confidence))

        raw_text = "\n".join(lines)
        mean_confidence = None
        if confidences:
            mean_confidence = round(sum(confidences) / len(confidences) * 100, 2)

        return OCRResult(
            method="paddleocr",
            image_path=str(image_path),
            text=self._clean_text(raw_text),
            raw_text=raw_text,
            confidence=mean_confidence,
            metadata={
                "preprocessing": [
                    "original_image",
                    "paddle_doc_orientation",
                    "paddle_detection",
                    "paddle_recognition",
                ],
                "angle_classification": True,
            },
        )

    def run_dataset_labels(self, image_path: str | Path) -> OCRResult:
        """Reference-text path using the local receipt_dataset annotation JSON."""
        image_path = Path(image_path)
        annotation_path = self._find_local_annotation_path(image_path)
        if annotation_path is None:
            raise FileNotFoundError(
                "No matching annotation JSON was found in data/receipt_dataset/ds0/ann "
                f"for receipt '{image_path.name}'."
            )

        annotation = json.loads(annotation_path.read_text())
        raw_text, categories = self._reconstruct_text_from_local_annotation(annotation)

        return OCRResult(
            method="labels",
            image_path=str(image_path),
            text=self._clean_text(raw_text),
            raw_text=raw_text,
            confidence=100.0,
            metadata={
                "source": "receipt_dataset annotation json",
                "annotation_path": str(annotation_path),
                "categories": categories,
            },
        )

    def extract_text(self, image_path: str | Path, method: str = "tesseract") -> OCRResult:
        """Public entry point for the rest of the pipeline."""
        method = method.lower()
        if method == "tesseract":
            return self.run_tesseract_ocr(image_path)
        if method == "paddleocr":
            return self.run_paddleocr(image_path)
        if method == "labels":
            return self.run_dataset_labels(image_path)
        raise ValueError("method must be 'tesseract', 'paddleocr', or 'labels'")

    def _load_paddle_ocr(self) -> None:
        if self._paddle_ocr is not None:
            return

        self._paddle_ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=True,
            use_textline_orientation=True,
        )

    def _find_local_annotation_path(self, image_path: Path) -> Path | None:
        annotations_dir = (
            Path(__file__).resolve().parent.parent / "data" / "receipt_dataset" / "ds0" / "ann"
        )
        candidate_paths = [
            annotations_dir / f"{image_path.name}.json",
            annotations_dir / f"{image_path.stem}.json",
        ]
        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path
        return None

    def _reconstruct_text_from_local_annotation(
        self,
        annotation: dict[str, Any],
    ) -> tuple[str, list[str]]:
        objects = annotation.get("objects", [])
        entries: list[dict[str, float | str]] = []
        heights: list[float] = []
        categories: set[str] = set()

        for obj in objects:
            transcription = None
            category = None
            for tag in obj.get("tags", []):
                if tag.get("name") == "Transcription":
                    transcription = str(tag.get("value", "")).strip()
                elif tag.get("name") == "Category":
                    category = str(tag.get("value", "")).strip()

            if not transcription:
                continue

            exterior = obj.get("points", {}).get("exterior", [])
            if len(exterior) != 2:
                continue

            (x1, y1), (x2, y2) = exterior
            heights.append(max(float(y2) - float(y1), 1.0))
            if category:
                categories.add(category)
            entries.append(
                {
                    "text": transcription,
                    "x1": float(x1),
                    "y_center": (float(y1) + float(y2)) / 2,
                }
            )

        if not entries:
            return "", sorted(categories)

        line_threshold = max(8.0, median(heights) * 0.7)
        entries.sort(key=lambda item: (float(item["y_center"]), float(item["x1"])))

        lines: list[list[dict[str, float | str]]] = []
        for entry in entries:
            if not lines:
                lines.append([entry])
                continue

            current_line = lines[-1]
            current_y = sum(float(item["y_center"]) for item in current_line) / len(current_line)
            if abs(float(entry["y_center"]) - current_y) <= line_threshold:
                current_line.append(entry)
            else:
                lines.append([entry])

        text_lines: list[str] = []
        for line in lines:
            ordered = [str(item["text"]) for item in sorted(line, key=lambda item: float(item["x1"]))]
            text_lines.append(" ".join(ordered))

        return "\n".join(text_lines), sorted(categories)

    @staticmethod
    def _clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    @staticmethod
    def _mean_tesseract_confidence(image: np.ndarray, config: str) -> float | None:
        data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        confidences: list[float] = []
        for value in data.get("conf", []):
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if score >= 0:
                confidences.append(score)

        if not confidences:
            return None
        return round(sum(confidences) / len(confidences), 2)

    @staticmethod
    def _estimate_skew_angle(binary_image: np.ndarray) -> float:
        coords = np.column_stack(np.where(binary_image < 255))
        if len(coords) == 0:
            return 0.0

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        return -angle

    @staticmethod
    def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        (height, width) = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
