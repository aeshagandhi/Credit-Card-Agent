# OCR Models Comparison Guide

This guide shows how to run and compare the three OCR models: Tesseract, TrOCR, and PaddleOCR.

## Models Overview

| Model | Type | Speed | Accuracy | Dependencies |
|-------|------|-------|----------|--------------|
| **Tesseract** | Classical/Traditional | Fast | Good for printed text | `pytesseract` |
| **TrOCR** | Deep Learning (Transformer) | Slow | High accuracy on documents | `transformers`, `torch` |
| **PaddleOCR** | Deep Learning (Paddle) | Medium | High accuracy, multi-lang support | `paddlepaddle`, `paddleocr` |

## Setup

First, install all dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
# Core dependencies
pip install opencv-python Pillow pytesseract numpy pandas

# TrOCR dependencies
pip install transformers torch==2.2.2

# PaddleOCR dependencies
pip install paddlepaddle paddleocr
```

## Usage

### 1. Run a Single OCR Method

Test Tesseract:
```bash
python src/main.py --image path/to/receipt.jpg --ocr-method tesseract
```

Test TrOCR:
```bash
python src/main.py --image path/to/receipt.jpg --ocr-method trocr
```

Test PaddleOCR:
```bash
python src/main.py --image path/to/receipt.jpg --ocr-method paddleocr
```

### 2. Compare All Three Models

Run all three OCR methods and see their outputs side-by-side:

```bash
python src/main.py --image path/to/receipt.jpg --compare-ocr
```

**Output Example:**
```
=== OCR Comparison ===

--- TESSERACT ---
Confidence: 85.50
Text:
[extracted text]

--- TROCR ---
Confidence: None
Text:
[extracted text]

--- PADDLEOCR ---
Confidence: 92.30
Text:
[extracted text]
```

### 3. Use with Pipeline Presets

```bash
# Pipeline v1 (Tesseract + Planning v1)
python src/main.py --image path/to/receipt.jpg --pipeline-version v1

# Pipeline v2 (TrOCR + Planning v2)
python src/main.py --image path/to/receipt.jpg --pipeline-version v2

# Override individual components
python src/main.py --image path/to/receipt.jpg --pipeline-version v1 --ocr-method paddleocr
```

### 4. Save Results to JSON

Compare all models and save outputs:

```bash
python src/main.py --image path/to/receipt.jpg --compare-ocr --save-json results/comparison.json
```

Or run full pipeline with specific method:

```bash
python src/main.py --image path/to/receipt.jpg --ocr-method paddleocr --save-json results/paddle_output.json
```

## Comparison Script (Batch Processing)

Create `compare_receipts.py` to test multiple receipts:

```python
import json
from pathlib import Path
from src.perception import ReceiptPerception

def compare_all_receipts(image_dir: str, output_dir: str = "outputs/comparisons"):
    """Compare all OCR methods on all receipts in a directory."""
    perception = ReceiptPerception()
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = ["tesseract", "trocr", "paddleocr"]
    
    for image_file in image_dir.glob("*.jpg"):
        print(f"\nProcessing {image_file.name}...")
        comparison = {
            "image": image_file.name,
            "results": {}
        }
        
        for method in methods:
            try:
                result = perception.extract_text(image_file, method=method)
                comparison["results"][method] = {
                    "confidence": result.confidence,
                    "text": result.text,
                    "raw_text": result.raw_text,
                }
                print(f"  ✓ {method}")
            except Exception as e:
                print(f"  ✗ {method}: {e}")
                comparison["results"][method] = {"error": str(e)}
        
        # Save individual comparison
        output_file = output_dir / f"{image_file.stem}_comparison.json"
        output_file.write_text(json.dumps(comparison, indent=2))
        print(f"  Saved to {output_file}")

if __name__ == "__main__":
    compare_all_receipts("data/receipts")
```

Usage:
```bash
python compare_receipts.py
```

## Model Characteristics

### Tesseract
- **Pros**: Lightweight, fast, no GPU needed
- **Cons**: Lower accuracy on skewed/rotated text
- **Best for**: Simple, clean, printed receipts
- **Confidence**: Available (0-100)

### TrOCR  
- **Pros**: High accuracy, good with handwritten text, transformer-based
- **Cons**: Slower, requires GPU for speed, larger memory footprint
- **Best for**: Complex layouts, mixed printed/handwritten
- **Confidence**: Not provided

### PaddleOCR
- **Pros**: Good balance of speed/accuracy, multi-language support, lightweight
- **Cons**: Slightly less accuracy than TrOCR on some layouts
- **Best for**: Fast processing with good accuracy, multi-language documents
- **Confidence**: Available (0-1 scale, converted to 0-100)

## Performance Benchmarking

Time each method on your receipts:

```python
import time
from pathlib import Path
from src.perception import ReceiptPerception

perception = ReceiptPerception()
image_path = "path/to/receipt.jpg"
methods = ["tesseract", "trocr", "paddleocr"]

for method in methods:
    start = time.time()
    result = perception.extract_text(image_path, method=method)
    elapsed = time.time() - start
    print(f"{method:12} | Time: {elapsed:.2f}s | Confidence: {result.confidence}")
```

## Troubleshooting

### PaddleOCR not found
```bash
pip install paddlepaddle paddleocr --upgrade
```

### TrOCR memory issues
- Use on GPU: `torch.cuda.is_available()`
- Or reduce batch size / use smaller model

### Tesseract not found
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Next Steps

1. **Evaluate accuracy**: Use `notebooks/evaluation.ipynb` to benchmark models
2. **Fine-tune preprocessing**: Adjust `preprocess_receipt()` parameters
3. **Integrate winner**: Use best-performing model as default in pipeline v3
