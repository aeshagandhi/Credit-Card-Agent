# Credit Card Agent

This project builds a receipt-based credit card recommendation agent with three stages:

1. `Perception`: turn a receipt image into text
2. `Planning`: turn receipt text into spending categories
3. `Control`: use an LLM agent with web tools to recommend a card

The project is implemented in two versions:

- `Version 1` uses non-deep-learning perception and planning
- `Version 2` uses deep-learning perception and planning

Both versions share the same LLM-based control phase.

## Version Overview

### Version 1: Non-DL

- `Perception`: Tesseract OCR
- `Planning`: classical merchant lookup + keyword-based categorization
- `Control`: shared LLM agent with web search

### Version 2: DL

- `Perception`: TrOCR
- `Planning`: transformer-based semantic transaction classifier
- `Control`: shared LLM agent with web search

### Reference Path

For comparison, the notebook also supports:

- `labels`: SROIE-provided words and bounding boxes reconstructed into text

This is useful for checking whether errors are coming from OCR or from the downstream pipeline.

## Project Structure

```text
credit-card-agent/
├── README.md
├── requirements.txt
├── .env
├── tool_registry.py
├── data/
│   ├── receipts/
│   ├── labels/
│   └── sample_profiles/
├── notebooks/
│   ├── execution_notebook.ipynb
│   ├── perception_experiments.ipynb
│   ├── planning_experiments.ipynb
│   └── evaluation.ipynb
├── src/
│   ├── main.py
│   ├── run_sample_receipts.py
│   ├── perception.py
│   ├── planning.py
│   ├── control.py
│   ├── evaluate.py
│   └── utils.py
├── models/
│   └── planning_v1.pkl
└── outputs/
    ├── ocr_results/
    └── recommendations/
```

## End-to-End Pipeline

```text
receipt image
  -> perception
  -> extracted receipt text
  -> planning
  -> spending profile
  -> control
  -> credit card recommendation
```

The same pipeline shape is used for both versions. Only perception and planning change between V1 and V2.

## Stage 1: Perception

Code: [src/perception.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/src/perception.py:1)

The perception module converts a receipt image into text. It supports three modes.

### `tesseract`

This is the non-DL perception path.

Processing steps:

1. Load image with OpenCV
2. Convert to grayscale
3. Apply Gaussian blur
4. Apply adaptive thresholding
5. Estimate skew and deskew the image
6. Run Tesseract OCR
7. Clean and return text

### `trocr`

This is the DL perception path.

Processing steps:

1. Run the same preprocessing as above
2. Convert the processed image to RGB
3. Pass the image into TrOCR
4. Decode generated text tokens
5. Clean and return text

Current note:

- TrOCR is currently applied to full receipt images
- It usually works better on cropped lines or regions
- So if it underperforms, that is mostly a pipeline setup issue, not proof that DL OCR is worse

### `labels`

This is the reference comparison path.

Processing steps:

1. Load cached SROIE `words` and `bboxes`
2. Group words into lines by vertical position
3. Sort words left-to-right
4. Reconstruct cleaned receipt text

This is not the main perception system. It is there so we can compare OCR-based input against the dataset’s provided text.

## Stage 2: Planning

Code: [src/planning.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/src/planning.py:1)

The planning module turns receipt text into a `SpendingProfile`.

Target categories:

- `groceries`
- `dining`
- `travel`
- `gas`
- `entertainment`
- `shopping`
- `healthcare`
- `other`

### Shared Planning Preprocessing

Both planning versions first do the same basic parsing:

1. Split OCR text into lines
2. Detect the merchant from the top of the receipt
3. Extract monetary amounts from lines
4. Remove summary rows like `TOTAL`, `SUBTOTAL`, `TAX`, and payment rows
5. Build item descriptions from remaining purchase-like lines

After that, V1 and V2 differ in how category decisions are made.

### Planning V1: Classical

This is the non-DL planning version.

How decisions are made:

1. Check whether the merchant strongly implies a category
2. Score the line using hand-written category keywords
3. Fall back to the merchant default when possible
4. Otherwise assign `other`

This is fast, interpretable, and simple, but it struggles with messy OCR abbreviations and semantic ambiguity.

### Planning V2: DL

This is the deep-learning planning version.

How decisions are made:

1. Use the merchant plus line description as the classification text
2. Run a transformer-based semantic classifier
3. Pick the best category label from the shared category set
4. If confidence is weak, fall back to merchant-based defaults or `other`

Implementation detail:

- The code uses a transformer zero-shot classification pipeline by default
- If a fine-tuned model is later placed in `models/planning_v2/`, the planner can point to that instead

This gives V2 a much more proposal-aligned planning path than simple keyword logic.

### Planning Output

Both V1 and V2 return the same `SpendingProfile` structure:

- `merchant`
- `planner_version`
- `planner_metadata`
- `category_totals`
- `total_amount`
- `line_items`
- `uncategorized_lines`

That shared output contract is what allows the control phase to stay identical across both versions.

## Stage 3: Control

Code: [src/control.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/src/control.py:1)

The control phase is shared across both V1 and V2.

It is an LLM-based agent that uses your local [tool_registry.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/tool_registry.py:1) tools to research current credit card information before recommending a card.

### Control Inputs

- the `SpendingProfile` from planning
- the current date
- tool access for live research

### Control Tools

The control agent can use:

- `web_search`
- `fetch_webpage`
- `calculator`
- `save_research_note`

### How The Control Agent Works

1. Receive the spending profile
2. Identify the top spend categories
3. Use `web_search` to find currently relevant credit cards
4. Use `fetch_webpage` to inspect promising sources and card details
5. Use `calculator` when useful for estimated annual value
6. Ask the LLM to synthesize a structured recommendation
7. Return JSON output

The control output includes:

- primary recommendation
- runner-up recommendation
- explanation
- caveats
- card rankings
- sources used

This means the control phase is no longer a fixed card lookup table. It is now a shared LLM research agent for both versions of the pipeline.

## Why The Architecture Matters

The project is intentionally structured so that:

- V1 and V2 differ only in perception and planning
- control is shared
- outputs remain compatible across versions

That makes the comparison cleaner:

- if V2 does better, we can attribute more of that gain to the DL perception/planning steps
- if `labels` does much better than OCR methods, the main bottleneck is perception
- if both planning versions behave similarly on `labels`, the planning difference may be small compared with OCR noise

## Streamlit UI

There is now a front-facing Streamlit app for presentation and demo use:

[app.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/app.py:1)

Run it with:

```bash
streamlit run app.py
```

What the UI supports:

- upload one or more receipt images
- choose a pipeline preset:
  - `Version 1: Classical`
  - `Version 2: Deep Learning`
  - `Labels Reference`
- combine multiple receipts into one spending profile
- generate one final shared control-phase recommendation
- inspect parsed text, category totals, and line items for each receipt
- ask follow-up questions in a small chat panel grounded in the current run

The Streamlit app is the best option for a final presentation because it feels like a user-facing product rather than a research notebook.

## Notebook Workflow

Use [execution_notebook.ipynb](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/notebooks/execution_notebook.ipynb:1) as the main interface.

The notebook now compares pipeline variants, not just OCR methods.

Default configurations:

- `version_1_non_dl`: `tesseract` + `planning v1`
- `version_2_dl`: `trocr` + `planning v2`
- `labels_reference`: `labels` + `planning v2`

The notebook shows:

- the receipt image
- parsed text for each pipeline
- spending profile comparison
- final recommendation comparison
- a batch table with inline thumbnails and parsed text blocks

## CLI Usage

### Single Receipt

Code: [src/main.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/src/main.py:1)

Example V1:

```bash
python src/main.py --image data/receipts/X00016469619.jpg --pipeline-version v1
```

Example V2:

```bash
python src/main.py --image data/receipts/X00016469619.jpg --pipeline-version v2
```

Example reference run:

```bash
python src/main.py --image data/receipts/X00016469619.jpg --ocr-method labels --planning-version v2
```

### Batch Sample Run

Code: [src/run_sample_receipts.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/src/run_sample_receipts.py:1)

```bash
python src/run_sample_receipts.py --pipeline-version v1 --limit 3
python src/run_sample_receipts.py --pipeline-version v2 --limit 3
```

## Environment Setup

Add your OpenAI key to `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Important notes:

- `torch` and `transformers` are required for TrOCR and planning V2
- the Tesseract binary must be installed separately on your machine
- the notebook should use the project `.venv`
- the control phase uses live web access through `tool_registry`, so internet access is required at runtime

## Current Limitations

1. TrOCR is still being run on whole receipts instead of segmented lines.
2. Planning V2 is transformer-based but not yet truly fine-tuned on a custom receipt transaction dataset.
3. Control recommendations depend on the quality of both live web results and the upstream spending profile.
4. OCR errors can still dominate the final recommendation quality.

## Main Files

- [src/perception.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/src/perception.py:1)
- [src/planning.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/src/planning.py:1)
- [src/control.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/src/control.py:1)
- [tool_registry.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/tool_registry.py:1)
- [src/main.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/src/main.py:1)
- [notebooks/execution_notebook.ipynb](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Credit-Card-Agent/notebooks/execution_notebook.ipynb:1)
