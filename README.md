# Credit Card Agent

This project is a receipt-based credit card recommendation system built as a school project prototype. The pipeline takes a receipt image, extracts text, converts that text into spending categories, and then uses an LLM research agent to recommend a credit card based on the spending profile.

The project is organized around three stages:

1. `Perception`: convert a receipt image into text
2. `Planning`: convert receipt text into structured spending categories
3. `Control`: use an LLM plus web research tools to recommend a card

The architecture is intentionally set up so that both project versions share the same control phase and the same output format. Only perception and planning change between versions.

## Current Versions

### Version 1: Classical

- `Perception`: Tesseract OCR
- `Planning`: rule-based merchant and keyword categorization
- `Control`: shared LLM agent with web research

### Version 2: Deep Learning

- `Perception`: PaddleOCR
- `Planning`: transformer-based semantic classification
- `Control`: shared LLM agent with web research

### Comparison Paths

The repo also keeps two comparison modes:

- `Experimental: TrOCR`: a DL OCR comparison path that is still useful for side-by-side evaluation
- `Curated Receipt Text`: dataset-provided text reconstructed from annotations so we can compare downstream behavior without OCR noise

In the current codebase, `Version 2` defaults to `paddleocr + planning v2`.

## End-to-End Pipeline

```text
receipt image
  -> perception
  -> extracted text
  -> planning
  -> spending profile
  -> control
  -> card recommendation
```

The shared output contract is:

- OCR text and metadata
- a `SpendingProfile`
- a final recommendation with explanation, rankings, caveats, and sources

That design is what makes it possible to compare V1, V2, TrOCR, and labels-reference runs inside the same notebook and UI.

## Agent Framing

Because this project is for an agents class, it helps to think about the pipeline in agent terms:

- `Perception` is the observation layer. It turns raw visual input into a text representation the rest of the system can use.
- `Planning` is the state-abstraction layer. It converts messy OCR output into a structured spending profile that is easier to reason over.
- `Control` is the decision-making layer. It uses that structured state, interacts with external tools, gathers additional evidence from the web, and produces a final recommendation.

In other words, the project is not just "OCR plus a card suggester." It is a staged agent pipeline in which each phase reduces uncertainty before handing the problem to the next phase.

## Project Structure

```text
credit-card-agent/
├── app.py
├── README.md
├── requirements.txt
├── .env
├── .streamlit/
│   └── config.toml
├── tool_registry.py
├── download_nano_receipts.py
├── data/
│   ├── receipt_dataset/
│   ├── receipts_nano/
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
├── tests/
│   └── test_pipeline.py
├── models/
│   └── planning_v1.pkl
└── outputs/
```

## Data Sources

The repo currently supports multiple receipt sources.

### Primary Local Dataset

- `data/receipt_dataset/ds0/img`: receipt images
- `data/receipt_dataset/ds0/ann`: matching annotation JSON files

This is the main dataset currently used by the helper utilities and demo flows. When possible, the project uses these images as the default sample receipts.

### Additional Datasets

- `data/receipts_nano`: Nano receipt images downloaded through the Hugging Face loader workflow
- `data/receipts`: older receipt image set used earlier in the project
- `data/labels/sroie_labels.json`: cached SROIE words and bounding boxes for label-based comparison
- `data/labels/nano_receipts_index.json`: Nano dataset index metadata

### Dataset Preference Order

The helper functions in [src/utils.py](src/utils.py) choose sample data in this order:

1. `data/receipt_dataset/ds0/img`
2. `data/receipts_nano`
3. `data/receipts`

For label-based comparison, the code first looks for a local annotation JSON in `data/receipt_dataset/ds0/ann` and then falls back to cached SROIE labels if needed.

## Stage 1: Perception

Code: [src/perception.py](src/perception.py)

The perception module returns a shared `OCRResult` object no matter which OCR method is used. That means the downstream planning and control phases do not need to change when the perception method changes.

### Perception Input and Output

Input:

- a receipt image path

Output:

- `method`: which OCR path was used
- `image_path`
- `text`: cleaned OCR text
- `raw_text`: raw extracted text before cleaning
- `confidence`: when available
- `metadata`: preprocessing and method-specific details

The important design choice is that all perception methods return the same object shape. That makes perception swappable without changing planning or control.

### `tesseract`

This is the non-DL OCR path used by `Version 1`.

Steps:

1. Load the image with OpenCV
2. Convert to grayscale
3. Apply Gaussian blur
4. Apply adaptive thresholding
5. Estimate skew and deskew
6. Run Tesseract
7. Clean the extracted text

Logic details:

- skew is estimated from the foreground pixel coordinates using OpenCV's `minAreaRect`
- the angle is corrected so upright receipts do not accidentally rotate by about 90 degrees
- OCR confidence is computed separately from `pytesseract.image_to_data`
- the cleaned text removes empty lines but otherwise preserves receipt order

This path is simple and explainable, which makes it a good non-DL baseline.

### `paddleocr`

This is the main DL OCR path used by `Version 2`.

Steps:

1. Load the original receipt image
2. Run PaddleOCR document orientation handling
3. Run PaddleOCR text detection
4. Run PaddleOCR text recognition
5. Collect recognized lines and confidence scores
6. Clean the extracted text

This works better than the current TrOCR setup in this repo because PaddleOCR is designed for full document OCR, while TrOCR is currently being applied more directly to whole receipt images.

Logic details:

- PaddleOCR is run on the original image instead of the heavily thresholded one because the model expects document-level structure
- the code enables document orientation classification and textline orientation handling before recognition
- recognized strings are collected from `rec_texts`, while numeric confidence values are collected from `rec_scores`
- the final confidence reported by the project is the average of the recognized line scores

For this project, PaddleOCR is the most practical DL perception method because it already includes the detection-plus-recognition workflow needed for receipts.

### `trocr`

This is an experimental comparison OCR path.

Steps:

1. Apply the same classical preprocessing used for Tesseract
2. Convert the processed image to RGB
3. Send it into `microsoft/trocr-base-printed`
4. Decode generated tokens into text
5. Clean the extracted text

TrOCR is still valuable for comparison, but it is not the main production path in this project right now.

Logic details:

- TrOCR is currently being used as a direct image-to-text recognizer on the full receipt image
- that is not its ideal setup, because it often works better on smaller cropped text regions or lines
- so when TrOCR performs badly here, that should be interpreted as a pipeline limitation, not necessarily a bad model overall

### `labels`

This is the reference-text comparison path.

It reconstructs receipt text from:

- local annotation JSON in `data/receipt_dataset/ds0/ann`, or
- cached SROIE words and bounding boxes in `data/labels/sroie_labels.json`

This path is useful for checking whether errors are caused mostly by OCR or by later planning/control logic.

Logic details:

- words are reconstructed into lines by grouping boxes with similar vertical centers
- the grouping threshold is based on the median text-box height
- words inside each line are sorted left-to-right before rejoining the text

This is important for evaluation because it gives us a nearly OCR-free text baseline while keeping the rest of the pipeline unchanged.

### Why Perception Matters in This Project

Perception is the main source of upstream uncertainty. If item names or totals are corrupted here, planning cannot fully recover them later. That is why the repo keeps several perception modes:

- one classical baseline
- one stronger DL path
- one experimental DL comparison path
- one reference-text path

## Stage 2: Planning

Code: [src/planning.py](src/planning.py)

The planner converts receipt text into a `SpendingProfile`. Both versions output the same structure:

- `merchant`
- `planner_version`
- `planner_metadata`
- `category_totals`
- `total_amount`
- `line_items`
- `uncategorized_lines`

Target categories are:

- `groceries`
- `dining`
- `travel`
- `gas`
- `entertainment`
- `shopping`
- `healthcare`
- `other`

### Planning Input and Output

Input:

- OCR text from perception

Output:

- detected merchant
- categorized line items
- category totals
- planner metadata about how decisions were made

The planner's job is to transform noisy receipt text into a compact "state" that the control phase can reason over.

### Shared Preprocessing

Both planning versions first do the same basic parsing:

1. Normalize the OCR text into lines
2. Detect the merchant from the top of the receipt
3. Extract monetary amounts
4. Drop summary rows such as `TOTAL`, `SUBTOTAL`, `TAX`, and payment lines
5. Build purchase-like line items

Logic details:

- merchant detection looks at the first few lines of the receipt
- if one of those lines matches a known merchant pattern, that merchant is used
- otherwise the first line is treated as the merchant fallback
- candidate purchase lines are selected only if they contain a parseable amount
- price text is stripped out before the description is categorized

This shared preprocessing is important because it keeps the V1 and V2 comparison fair. Both versions start from the same parsed candidate lines.

### Planning V1

This is the classical non-DL planning path.

It mainly uses:

- merchant lookup defaults
- keyword matching by category
- simple fallback rules

This version is easier to interpret and explain, but it is also more brittle when OCR is noisy or item names are unusual.

Detailed decision logic:

1. Combine merchant text and item description into one text string
2. Check whether the merchant directly implies a category such as `Costco -> groceries` or `Chipotle -> dining`
3. If not, count keyword hits for each category
4. Choose the category with the strongest keyword score
5. If no keywords match, fall back to the merchant default when available
6. If nothing is informative, assign `other`

This version behaves like a hand-built symbolic classifier. Its advantage is interpretability. Its weakness is that it cannot reason semantically beyond its merchant list and keyword list.

### Planning V2

This is the DL planning path.

It uses:

- an optional LLM-assisted receipt structure parser to separate purchase lines from summary lines
- merchant plus line-item description as the classification text
- a transformer zero-shot classifier
- shared category labels
- fallback logic when confidence is weak

The default model is a zero-shot NLI classifier rather than a custom fine-tuned receipt classifier. That is appropriate for this project scope, but it is still a limitation to mention in the final presentation.

Detailed decision logic:

1. Parse the receipt into candidate purchase lines
2. If an OpenAI API key is available, V2 first tries an LLM receipt parser that separates:
   - true purchased items
   - subtotal, tax, tip, total, and payment lines
   - metadata such as date, cashier, table number, or receipt ID
3. If the LLM parser is unavailable or fails, fall back to the heuristic candidate-line parser
4. Estimate a default merchant category
5. If the merchant is known, use the lookup-based merchant category
6. If the merchant is not known, run a zero-shot classifier on the merchant name itself to infer a semantic default category
7. For each candidate purchase line, create a classification text using the merchant plus item description
8. Run zero-shot classification over the project category labels except `other`
9. Take the top-scoring category and its score
10. If the score is below the threshold, fall back to the merchant default or `other`
11. Store both the chosen category and the model score in the line item
12. If summary lines such as tax, tip, fees, or discounts are detected, allocate the spend-relevant ones back into category totals while still excluding duplicate lines such as subtotal, total, payment, and change

Important implementation details:

- the default V2 model is `typeform/distilbert-base-uncased-mnli`
- if a local model exists in `models/planning_v2/`, the code will prefer that instead
- the current score threshold is `0.35`
- if an API key is configured, the default receipt-structure parser model is `gpt-4o-mini`
- the fallback heuristic parser can pair amount-only lines with the preceding item description, which helps avoid counting `Subtotal`, `Tax`, and `Total` as purchased items
- spend-relevant summary lines like `Tax`, `Tip`, `Service Charge`, `Fees`, and `Discount` can now be handled separately from item lines so the parsed spend total is closer to the actual charged amount

This is more proposal-aligned than pure keyword rules because it can make semantic judgments about item names, even when the wording is not an exact keyword match.

### Why Planning Matters in This Project

Planning is the bridge between raw OCR text and the final recommendation. The control phase does not read the receipt directly. It reads the `SpendingProfile`. That means planning is effectively the world-model abstraction step for the agent.

If planning gets the categories wrong, even a strong control agent may recommend the wrong card for the wrong reasons.

## Stage 3: Control

Code: [src/control.py](src/control.py)

The control phase is shared across all versions. It is no longer rule-based. It is an LLM agent that researches current card options and returns a structured recommendation.

The control agent uses [tool_registry.py](tool_registry.py) and currently has access to:

- `web_search`
- `fetch_webpage`
- `calculator`
- `save_research_note`

### Control Inputs

- the `SpendingProfile`
- the current date
- live web-search and webpage-fetch tools

### Control as the Agentic Phase

This is the most explicitly agent-like part of the system.

Instead of mapping categories to a fixed card table, the control module:

1. reads the current spending profile
2. decides what outside information it still needs
3. uses tools to gather that information
4. checks whether it has enough evidence
5. produces a structured decision with sources

That loop is why this phase is genuinely agentic rather than just another static classifier.

### Control Output

The final output includes:

- primary recommendation
- runner-up recommendation
- explanation
- caveats
- ranked card list
- sources

This is what makes the project proposal-aligned: both V1 and V2 use the same LLM-based control phase.

### Detailed Control Logic

The control module runs an iterative tool-using LLM loop:

1. Build a system prompt that explains the task, the current date, and the required JSON schema
2. Build a user prompt containing:
   - the spending profile
   - the top spending categories
   - suggested search queries such as `best credit cards for dining rewards`
   - the required response structure
3. Call the OpenAI model with tool use enabled
4. If the model requests tools, execute them locally and append their results back into the conversation
5. Repeat until the model returns a final JSON answer

The code also enforces several agent constraints:

- the agent must use `web_search` and `fetch_webpage` before a final answer is accepted
- tool arguments are parsed as JSON and executed through a controlled wrapper
- the final answer must be valid JSON
- the final JSON is normalized into a `CardRecommendation` object
- the loop is capped at `max_tool_rounds = 10` so the agent cannot run forever

### Why This Is Better Than a Rule-Based Control Step

A rule-based control stage would have to rely on a hardcoded and quickly outdated mapping from spend categories to cards. The LLM agent can instead:

- research current cards
- verify card details from live pages
- compare multiple options
- adapt when card offerings change over time
- provide sources and caveats instead of only returning a label

That is especially important for credit cards, because reward structures, annual fees, and promotional positioning change over time.

### Example Control Flow

At a high level, a typical successful control run looks like this:

1. identify the top categories from the spending profile
2. search for current reward cards for those categories
3. open a few promising pages
4. estimate annual value using the parsed spend totals
5. return a best card, a runner-up, short caveats, and the sources used

## Streamlit Demo UI

Code: [app.py](app.py)

The repo includes a front-facing Streamlit app for presentation and demo use.

Run it with:

```bash
streamlit run app.py
```

The UI supports:

- uploading one or more receipt images
- choosing a pipeline preset
- combining multiple receipts into one aggregate spending profile
- running the shared recommendation phase
- viewing parsed text, category totals, and line items
- chatting about the results after a run

Current presets in the UI:

- `Version 1: Classical`
- `Version 2: Deep Learning`
- `Experimental: TrOCR`
- `Curated Receipt Text`

The UI also now includes a light-theme configuration in [.streamlit/config.toml](.streamlit/config.toml) so it remains readable even if the computer itself is using dark mode.

## Execution Notebook

Notebook: [notebooks/execution_notebook.ipynb](notebooks/execution_notebook.ipynb)

This is the main comparison notebook for the project. It is meant to show the pipeline side by side across methods, not just OCR outputs.

The notebook currently compares:

- `version_1_non_dl`: `tesseract` + `planning v1`
- `version_2_dl`: `paddleocr` + `planning v2`
- `experimental_trocr`: `trocr` + `planning v2`
- `labels_reference`: `labels` + `planning v2`

The notebook shows:

- the selected receipt image
- parsed text for each method
- planning outputs
- recommendation outputs
- comparison tables with larger images and extracted text for visual inspection

## Environment Setup

This project is easiest to run with Python `3.11` or `3.12`.

### 1. Create and activate a virtual environment

Example:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 4. External dependency note

Tesseract OCR requires the Tesseract binary to be installed separately on your machine. The Python package alone is not enough.

## Important Runtime Notes

- `Version 2` depends on `paddleocr` and `paddlepaddle`
- `Experimental: TrOCR` and planning V2 depend on `transformers` and `torch`
- the first run of PaddleOCR or TrOCR may download model weights
- the control phase requires internet access because it uses live research tools
- the notebook and Streamlit app should be run from the project environment so the OCR and ML dependencies are available

## Recommended Run Commands

### Streamlit

```bash
streamlit run app.py
```

### Main notebook

```bash
jupyter lab notebooks/execution_notebook.ipynb
```

### Single-receipt CLI

Recommended module-style invocation from the repo root:

```bash
python -m src.main --image data/receipt_dataset/ds0/img/1007-receipt.jpg --pipeline-version v1
python -m src.main --image data/receipt_dataset/ds0/img/1007-receipt.jpg --pipeline-version v2
python -m src.main --image data/receipt_dataset/ds0/img/1007-receipt.jpg --ocr-method labels --planning-version v2
```

### OCR comparison

```bash
python -m src.main --image data/receipt_dataset/ds0/img/1007-receipt.jpg --compare-ocr
```

### Batch sample run

```bash
python -m src.run_sample_receipts --pipeline-version v1 --limit 3
python -m src.run_sample_receipts --pipeline-version v2 --limit 3
```

## Why The Architecture Matters

The repo is structured so that:

- V1 and V2 differ only in perception and planning
- the control phase stays shared
- outputs stay compatible across methods

That makes evaluation much cleaner:

- if `labels` performs much better than OCR methods, the bottleneck is mostly perception
- if V2 outperforms V1 on the same receipt, the gain likely comes from the DL perception/planning path
- if planning outputs look similar across methods when fed clean labels, then OCR quality is probably the main source of downstream error

## Current Limitations

1. TrOCR is still being run on whole receipt images instead of segmented lines or regions.
2. Planning V2 is transformer-based, but it is not a custom fine-tuned receipt classifier.
3. OCR quality still has a large effect on downstream planning quality.
4. The control agent depends on live web results and LLM behavior, so recommendations can vary slightly over time.
5. This is a prototype-style school project, so the goal is a clear working pipeline and comparison framework rather than a fully production-hardened system.

## Main Files

- [app.py](app.py)
- [src/perception.py](src/perception.py)
- [src/planning.py](src/planning.py)
- [src/control.py](src/control.py)
- [src/utils.py](src/utils.py)
- [src/main.py](src/main.py)
- [src/run_sample_receipts.py](src/run_sample_receipts.py)
- [tool_registry.py](tool_registry.py)
- [notebooks/execution_notebook.ipynb](notebooks/execution_notebook.ipynb)
