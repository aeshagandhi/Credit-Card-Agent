from __future__ import annotations

from pathlib import Path
import json

from datasets import load_dataset


def main() -> None:
    project_root = Path(__file__).resolve().parent
    receipts_dir = project_root / "data" / "receipts_nano"
    labels_dir = project_root / "data" / "labels"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("34data/nano-receipts")
    rows = dataset["train"]
    index: dict[str, dict[str, str]] = {}

    for row in rows:
        image = row["image"]
        filename = row["filename"]
        receipt_id = row["receipt_id"]
        image.save(receipts_dir / filename)
        index[filename] = {"receipt_id": receipt_id}

    index_path = labels_dir / "nano_receipts_index.json"
    index_path.write_text(json.dumps(index, indent=2))

    print(
        {
            "count": len(index),
            "receipts_dir": str(receipts_dir),
            "index_path": str(index_path),
        }
    )


if __name__ == "__main__":
    main()
