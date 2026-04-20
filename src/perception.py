from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from PIL import Image
import pytesseract

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:  # pragma: no cover - optional dependency at runtime
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency at runtime
    load_dataset = None

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
    """Receipt OCR wrapper with classical and DL-based methods."""

    TROCR_MODEL_NAME = "microsoft/trocr-base-printed"

    def __init__(self, tesseract_psm: int = 6) -> None:
        self.tesseract_psm = tesseract_psm
        self._trocr_processor = None
        self._trocr_model = None
        self._paddle_ocr = None
        self._sroie_index: dict[str, dict[str, Any]] | None = None

    def preprocess_receipt(self, image_path: str | Path) -> np.ndarray:
        """Apply simple cleanup steps to improve OCR quality."""
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
        deskewed = self._rotate_image(binary, angle) if abs(angle) > 0.5 else binary
        return deskewed

    def run_tesseract_ocr(self, image_path: str | Path) -> OCRResult:
        """Classical OCR path using Tesseract."""
        processed = self.preprocess_receipt(image_path)
        config = f"--oem 3 --psm {self.tesseract_psm}"
        raw_text = pytesseract.image_to_string(processed, config=config)
        confidence = self._mean_tesseract_confidence(processed, config)
        cleaned_text = self._clean_text(raw_text)

        return OCRResult(
            method="tesseract",
            image_path=str(image_path),
            text=cleaned_text,
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

    def run_trocr(self, image_path: str | Path) -> OCRResult:
        """Deep-learning OCR path using Microsoft's TrOCR model."""
        if TrOCRProcessor is None or VisionEncoderDecoderModel is None:
            raise ImportError(
                "TrOCR dependencies are not available. Install transformers and torch "
                "in a supported Python environment such as Python 3.11 or 3.12 to use TrOCR."
            )

        self._load_trocr_model()
        processed = self.preprocess_receipt(image_path)
        rgb_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(rgb_image)

        pixel_values = self._trocr_processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values
        generated_ids = self._trocr_model.generate(pixel_values)
        raw_text = self._trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        cleaned_text = self._clean_text(raw_text)

        return OCRResult(
            method="trocr",
            image_path=str(image_path),
            text=cleaned_text,
            raw_text=raw_text,
            confidence=None,
            metadata={
                "model_name": self.TROCR_MODEL_NAME,
                "preprocessing": [
                    "grayscale",
                    "gaussian_blur",
                    "adaptive_threshold",
                    "deskew",
                ],
            },
        )

    def run_paddleocr(self, image_path: str | Path) -> OCRResult:
        """Deep-learning OCR path using PaddleOCR."""
        if PaddleOCR is None:
            raise ImportError(
                "PaddleOCR dependencies are not available. Install paddlepaddle and paddleocr."
            )

        self._load_paddle_ocr()
        processed = self.preprocess_receipt(image_path)
        rgb_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(rgb_image)

        results = self._paddle_ocr.ocr(np.array(pil_image), cls=True)
        raw_text = ""
        confidence = 0.0
        if results and results[0]:
            texts = [line[1][0] for line in results[0]]
            confidences = [line[1][1] for line in results[0]]
            raw_text = "\n".join(texts)
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
        cleaned_text = self._clean_text(raw_text)

        return OCRResult(
            method="paddleocr",
            image_path=str(image_path),
            text=cleaned_text,
            raw_text=raw_text,
            confidence=round(confidence, 2) if confidence else None,
            metadata={
                "preprocessing": [
                    "grayscale",
                    "gaussian_blur",
                    "adaptive_threshold",
                    "deskew",
                ],
            },
        )

    def run_dataset_labels(self, image_path: str | Path) -> OCRResult:
        """Reference-text path using the provided SROIE words/bboxes labels."""
        image_path = Path(image_path)
        receipt_key = image_path.stem
        row = self._get_sroie_row(receipt_key)
        raw_text = self._reconstruct_text_from_labels(row["words"], row["bboxes"])
        cleaned_text = self._clean_text(raw_text)

        return OCRResult(
            method="labels",
            image_path=str(image_path),
            text=cleaned_text,
            raw_text=raw_text,
            confidence=1.0,
            metadata={
                "source": "SROIE provided words+bboxes",
                "receipt_key": receipt_key,
            },
        )

    def extract_text(
        self, image_path: str | Path, method: str = "tesseract"
    ) -> OCRResult:
        """Public entry point for the rest of the pipeline."""
        method = method.lower()
        if method == "tesseract":
            return self.run_tesseract_ocr(image_path)
        if method == "trocr":
            return self.run_trocr(image_path)
        if method == "paddleocr":
            return self.run_paddleocr(image_path)
        if method == "labels":
            return self.run_dataset_labels(image_path)
        raise ValueError(
            "method must be 'tesseract', 'trocr', 'paddleocr', or 'labels'"
        )

    def _load_trocr_model(self) -> None:
        if self._trocr_model is not None and self._trocr_processor is not None:
            return

        self._trocr_processor = TrOCRProcessor.from_pretrained(self.TROCR_MODEL_NAME)
        self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
            self.TROCR_MODEL_NAME
        )

    def _load_paddle_ocr(self) -> None:
        if self._paddle_ocr is not None:
            return

        self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def _get_sroie_row(self, receipt_key: str) -> dict[str, Any]:
        if self._sroie_index is None:
            local_labels_path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "labels"
                / "sroie_labels.json"
            )
            if local_labels_path.exists():
                self._sroie_index = json.loads(local_labels_path.read_text())
            else:
                if load_dataset is None:
                    raise ImportError(
                        "datasets is not installed. Install the datasets package to use the labels method."
                    )

                dataset = load_dataset("jsdnrs/ICDAR2019-SROIE")
                index: dict[str, dict[str, Any]] = {}
                for split in dataset.values():
                    for row in split:
                        index[str(row["key"])] = {
                            "words": row["words"],
                            "bboxes": row["bboxes"],
                        }
                self._sroie_index = index

        if receipt_key not in self._sroie_index:
            raise KeyError(f"No provided labels found for receipt key: {receipt_key}")
        return self._sroie_index[receipt_key]

    def _reconstruct_text_from_labels(
        self,
        words: list[str],
        bboxes: list[list[int]],
    ) -> str:
        if not words or not bboxes or len(words) != len(bboxes):
            return ""

        entries = []
        heights = []
        for word, bbox in zip(words, bboxes):
            x1, y1, x2, y2 = bbox
            heights.append(max(y2 - y1, 1))
            entries.append(
                {
                    "word": word.strip(),
                    "x1": x1,
                    "y_center": (y1 + y2) / 2,
                }
            )

        line_threshold = max(8.0, median(heights) * 0.7)
        entries.sort(key=lambda item: (item["y_center"], item["x1"]))

        lines: list[list[dict[str, Any]]] = []
        for entry in entries:
            if not entry["word"]:
                continue

            if not lines:
                lines.append([entry])
                continue

            current_line = lines[-1]
            current_y = sum(item["y_center"] for item in current_line) / len(
                current_line
            )
            if abs(entry["y_center"] - current_y) <= line_threshold:
                current_line.append(entry)
            else:
                lines.append([entry])

        text_lines = []
        for line in lines:
            ordered_words = [
                item["word"] for item in sorted(line, key=lambda item: item["x1"])
            ]
            text_lines.append(" ".join(ordered_words))

        return "\n".join(text_lines)

    @staticmethod
    def _clean_text(text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        filtered = [line for line in lines if line]
        return "\n".join(filtered)

    @staticmethod
    def _mean_tesseract_confidence(image: np.ndarray, config: str) -> float | None:
        data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        confidences = []
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
