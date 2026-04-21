from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from PIL import Image
import streamlit as st

from src.control import CreditCardRecommender
from src.perception import ReceiptPerception
from src.planning import CATEGORIES, ReceiptPlanner
from src.utils import (
    has_reference_labels,
    list_receipt_images,
    merge_spending_profiles,
    ocr_result_as_dict,
    project_root,
    preferred_labeled_receipts_dir,
    preferred_receipts_dir,
    preview_text,
    resolve_reference_labeled_image,
    run_receipt_pipeline,
    save_uploaded_bytes,
)


PROJECT_ROOT = project_root()
load_dotenv(PROJECT_ROOT / ".env", override=False)

PIPELINE_PRESETS = {
    "Version 1: Classical": {
        "pipeline_version": "v1",
        "ocr_method": "tesseract",
        "planning_version": "v1",
        "description": "Classical OCR plus rule-based planning.",
    },
    "Version 2: Deep Learning": {
        "pipeline_version": "v2",
        "ocr_method": "paddleocr",
        "planning_version": "v2",
        "description": "PaddleOCR plus transformer-based planning.",
    },
    "Experimental: TrOCR": {
        "pipeline_version": "v2",
        "ocr_method": "trocr",
        "planning_version": "v2",
        "description": "TrOCR plus transformer-based planning for side-by-side OCR comparison.",
    },
    "Labels Reference": {
        "pipeline_version": None,
        "ocr_method": "labels",
        "planning_version": "v2",
        "description": "Dataset-provided text with planning v2 for comparison.",
    },
}


st.set_page_config(
    page_title="Receipt Card Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_perception() -> ReceiptPerception:
    return ReceiptPerception()


@st.cache_resource(show_spinner=False)
def get_planner() -> ReceiptPlanner:
    return ReceiptPlanner()


@st.cache_resource(show_spinner=False)
def get_recommender() -> CreditCardRecommender:
    return CreditCardRecommender()


@st.cache_resource(show_spinner=False)
def get_chat_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --paper: #f6f1e7;
            --card: rgba(255, 252, 245, 0.84);
            --ink: #12263a;
            --muted: #5b6b7a;
            --accent: #c56b2f;
            --accent-deep: #6f3b18;
            --leaf: #3b6658;
            --border: rgba(18, 38, 58, 0.14);
        }

        html, body, [class*="stApp"] {
            color-scheme: light;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(197, 107, 47, 0.14), transparent 32%),
                radial-gradient(circle at top right, rgba(59, 102, 88, 0.14), transparent 28%),
                linear-gradient(180deg, #f8f5ee 0%, #f1eadf 100%);
            color: var(--ink) !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: transparent !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 250, 242, 0.98), rgba(243, 235, 220, 0.98)) !important;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink) !important;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: Georgia, "Times New Roman", serif;
            color: var(--ink) !important;
            letter-spacing: -0.02em;
        }

        p, li, label, span, div, small {
            color: var(--ink);
        }

        a {
            color: var(--accent-deep) !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] * {
            color: var(--ink) !important;
        }

        .stRadio label,
        .stCheckbox label,
        .stMultiSelect label,
        .stFileUploader label,
        .stChatInput label {
            color: var(--ink) !important;
            font-weight: 600;
        }

        [data-baseweb="radio"] *,
        [data-baseweb="checkbox"] *,
        [data-baseweb="select"] *,
        [data-baseweb="tag"] *,
        [data-baseweb="popover"] * {
            color: var(--ink) !important;
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        .stFileUploader > div > div {
            background: rgba(255, 251, 244, 0.95) !important;
            border: 1px solid rgba(18, 38, 58, 0.16) !important;
            border-radius: 16px !important;
            box-shadow: none !important;
        }

        [data-baseweb="select"] input,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        .stChatInput input {
            color: var(--ink) !important;
            caret-color: var(--accent-deep) !important;
        }

        .stButton button,
        .stDownloadButton button,
        .stFileUploader button {
            background: linear-gradient(135deg, #c56b2f, #a5521c) !important;
            color: #fff9f2 !important;
            border: none !important;
            border-radius: 999px !important;
            box-shadow: 0 10px 24px rgba(165, 82, 28, 0.22);
        }

        .stButton button:hover,
        .stDownloadButton button:hover,
        .stFileUploader button:hover {
            filter: brightness(1.03);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 251, 244, 0.82) !important;
            border: 1px solid rgba(18, 38, 58, 0.12) !important;
            border-radius: 14px 14px 0 0 !important;
            color: var(--muted) !important;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(255, 255, 255, 0.96) !important;
            color: var(--ink) !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: rgba(255, 255, 255, 0.82) !important;
            border: 1px solid rgba(18, 38, 58, 0.12) !important;
            border-radius: 18px !important;
            padding: 0.3rem !important;
        }

        [data-testid="stDataFrame"] * ,
        [data-testid="stTable"] * {
            color: var(--ink) !important;
        }

        [data-testid="stAlert"] {
            background: rgba(255, 250, 242, 0.92) !important;
            border: 1px solid rgba(18, 38, 58, 0.12) !important;
            border-radius: 18px !important;
        }

        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, #c56b2f, #3b6658) !important;
        }

        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(18, 38, 58, 0.1);
            border-radius: 18px;
            padding: 0.3rem 0.75rem;
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(248,241,228,0.9));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.5rem 1.6rem 1.3rem 1.6rem;
            box-shadow: 0 18px 48px rgba(18, 38, 58, 0.08);
            margin-bottom: 1rem;
        }

        .hero-kicker {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--accent-deep);
            margin-bottom: 0.5rem;
            font-weight: 700;
        }

        .hero-copy {
            color: var(--muted);
            max-width: 52rem;
            font-size: 1rem;
            line-height: 1.55;
        }

        .metric-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            min-height: 140px;
            box-shadow: 0 10px 30px rgba(18, 38, 58, 0.05);
        }

        .metric-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            color: var(--muted);
            margin-bottom: 0.45rem;
            font-weight: 700;
        }

        .metric-value {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.75rem;
            color: var(--ink);
            line-height: 1.1;
        }

        .metric-sub {
            color: var(--muted);
            margin-top: 0.45rem;
            font-size: 0.92rem;
            line-height: 1.35;
        }

        .recommendation-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(245,236,221,0.88));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            box-shadow: 0 14px 38px rgba(18, 38, 58, 0.08);
            height: 100%;
        }

        .rec-eyebrow {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--accent-deep);
            margin-bottom: 0.45rem;
            font-weight: 700;
        }

        .rec-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.7rem;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }

        .rec-meta {
            color: var(--muted);
            margin-bottom: 0.7rem;
        }

        .source-pill {
            display: inline-block;
            border: 1px solid rgba(18, 38, 58, 0.14);
            border-radius: 999px;
            padding: 0.28rem 0.65rem;
            margin: 0.2rem 0.35rem 0 0;
            background: rgba(59, 102, 88, 0.08);
            color: var(--ink);
            font-size: 0.84rem;
        }

        .receipt-caption {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .receipt-panel {
            background: rgba(255,255,255,0.7);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def pick_top_category(category_totals: dict[str, float]) -> tuple[str, float]:
    non_zero = [(category, amount) for category, amount in category_totals.items() if amount > 0]
    if not non_zero:
        return "none", 0.0
    return max(non_zero, key=lambda item: item[1])


def render_metric_card(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_card(title: str, issuer: str, value: float, eyebrow: str) -> None:
    st.markdown(
        f"""
        <div class="recommendation-card">
            <div class="rec-eyebrow">{eyebrow}</div>
            <div class="rec-title">{title}</div>
            <div class="rec-meta">{issuer}<br>{format_currency(value)} estimated annual value</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_input_assets(
    uploaded_files: list[Any] | None,
    selected_sample_names: list[str],
    sample_dir: Path,
) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []

    if uploaded_files:
        for uploaded in uploaded_files:
            assets.append(
                {
                    "name": uploaded.name,
                    "kind": "upload",
                    "bytes": uploaded.getvalue(),
                }
            )

    for sample_name in selected_sample_names:
        sample_path = sample_dir / sample_name
        if sample_path.exists():
            assets.append(
                {
                    "name": sample_name,
                    "kind": "sample",
                    "path": sample_path,
                }
            )

    return assets


def analyze_receipts(
    assets: list[dict[str, Any]],
    preset: dict[str, Any],
    run_control: bool,
) -> dict[str, Any]:
    perception = get_perception()
    planner = get_planner()
    receipt_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    temporary_paths: list[Path] = []

    progress = st.progress(0, text="Preparing receipt analysis...")

    try:
        for index, asset in enumerate(assets):
            progress.progress(
                index / max(len(assets), 1),
                text=f"Reading {asset['name']}...",
            )

            image_path: Path
            image_payload: bytes | None = None
            pipeline_image_path: Path
            if asset["kind"] == "upload":
                image_payload = asset["bytes"]
                image_path = save_uploaded_bytes(asset["name"], image_payload)
                temporary_paths.append(image_path)
            else:
                image_path = asset["path"]
                image_payload = image_path.read_bytes()

            try:
                pipeline_image_path = image_path
                if preset["ocr_method"] == "labels":
                    reference_image_path = resolve_reference_labeled_image(
                        image_path=image_path if asset["kind"] == "sample" else None,
                        file_name=asset["name"],
                        file_bytes=image_payload,
                    )
                    if reference_image_path is None:
                        raise FileNotFoundError(
                            "No matching labeled reference receipt was found in the local dataset "
                            f"for uploaded file '{asset['name']}'."
                        )
                    pipeline_image_path = reference_image_path

                result = run_receipt_pipeline(
                    image_path=pipeline_image_path,
                    perception=perception,
                    planner=planner,
                    pipeline_version=preset["pipeline_version"],
                    ocr_method=preset["ocr_method"],
                    planning_version=preset["planning_version"],
                    run_control=False,
                )
                result["display_name"] = asset["name"]
                result["image_bytes"] = image_payload
                receipt_results.append(result)
            except Exception as exc:
                errors.append({"receipt": asset["name"], "error": str(exc)})

        progress.progress(1.0, text="Building combined spending profile...")

        if not receipt_results:
            raise RuntimeError("No receipts were successfully processed.")

        aggregate_profile = merge_spending_profiles(
            [result["spending_profile"] for result in receipt_results],
            source_names=[result["display_name"] for result in receipt_results],
        )

        aggregate_recommendation = None
        if run_control:
            progress.progress(1.0, text="Researching card options and generating recommendation...")
            recommender = get_recommender()
            aggregate_recommendation = recommender.recommend_card(aggregate_profile)

        return {
            "preset": preset,
            "receipt_results": receipt_results,
            "aggregate_profile": aggregate_profile,
            "aggregate_recommendation": aggregate_recommendation,
            "errors": errors,
        }
    finally:
        progress.empty()
        for temp_path in temporary_paths:
            temp_path.unlink(missing_ok=True)


def category_chart_frame(profile) -> pd.DataFrame:
    rows = [
        {"category": category.title(), "amount": amount}
        for category, amount in profile.category_totals.items()
        if amount > 0
    ]
    if not rows:
        rows = [{"category": "No parsed spend", "amount": 0.0}]
    return pd.DataFrame(rows).set_index("category")


def line_items_frame(line_items: list[Any]) -> pd.DataFrame:
    rows = [
        {
            "description": item.description,
            "amount": item.amount,
            "category": item.category,
            "score": item.score,
        }
        for item in line_items
    ]
    return pd.DataFrame(rows)


def sources_markup(sources: list[dict[str, str]]) -> str:
    pills = []
    for source in sources:
        title = source.get("title", "Source")
        url = source.get("url", "")
        pills.append(
            f'<a class="source-pill" href="{url}" target="_blank">{title}</a>'
        )
    return "".join(pills)


def build_chat_context(analysis: dict[str, Any]) -> str:
    receipt_context = []
    for result in analysis["receipt_results"]:
        receipt_context.append(
            {
                "receipt_name": result["display_name"],
                "pipeline": result["pipeline"],
                "ocr": {
                    "method": result["ocr_result"].method,
                    "confidence": result["ocr_result"].confidence,
                    "text_preview": preview_text(result["ocr_result"].text, limit=900),
                },
                "planning": result["spending_profile"].as_dict(),
            }
        )

    payload = {
        "pipeline_preset": analysis["preset"],
        "aggregate_spending_profile": analysis["aggregate_profile"].as_dict(),
        "aggregate_recommendation": (
            analysis["aggregate_recommendation"].as_dict()
            if analysis["aggregate_recommendation"]
            else None
        ),
        "receipts": receipt_context,
        "errors": analysis["errors"],
    }
    return json.dumps(payload, indent=2)


def answer_pipeline_question(question: str, analysis: dict[str, Any]) -> str:
    client = get_chat_client()
    if client is None:
        return "OpenAI API access is not configured, so chat follow-up is unavailable."

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the front-end assistant for a receipt-to-credit-card recommendation demo. "
                    "Answer using only the analysis context provided to you. "
                    "If the answer is not supported by the current results, say that directly. "
                    "Keep answers concise, useful, and presentation-friendly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Analysis context:\n{build_chat_context(analysis)}\n\n"
                    f"User question: {question}"
                ),
            },
        ],
    )
    return response.choices[0].message.content or "I couldn't generate an answer."


def render_sidebar() -> tuple[str, bool, list[str]]:
    st.sidebar.markdown("## Demo Controls")
    preset_name = st.sidebar.radio(
        "Pipeline preset",
        list(PIPELINE_PRESETS.keys()),
        help="Choose which pipeline version powers the demo.",
    )
    run_control = st.sidebar.checkbox(
        "Run card recommendation",
        value=True,
        help="Turn this off if you want to inspect perception and planning only.",
    )

    sample_dir = preferred_labeled_receipts_dir() if preset_name == "Labels Reference" else preferred_receipts_dir()
    sample_choices = [path.name for path in list_receipt_images(sample_dir)[:12]]
    selected_samples = st.sidebar.multiselect(
        "Or load sample receipts",
        sample_choices,
        help="Helpful for a quick in-class demo without uploading files.",
    )

    st.sidebar.markdown("## About")
    st.sidebar.caption(
        "The control phase uses an LLM plus live web research tools. "
        "Multiple receipts are merged into one combined spending profile before the final recommendation."
    )
    return preset_name, run_control, selected_samples, sample_dir


def render_receipt_preview(assets: list[dict[str, Any]]) -> None:
    if not assets:
        return

    st.markdown("### Selected Receipts")
    columns = st.columns(min(3, len(assets)))
    for index, asset in enumerate(assets):
        column = columns[index % len(columns)]
        with column:
            if asset["kind"] == "upload":
                image = Image.open(BytesIO(st.session_state["uploaded_file_map"][asset["name"]]))
            else:
                image = Image.open(asset["path"])

            st.image(image, use_container_width=True)
            st.markdown(f"<div class='receipt-caption'>{asset['name']}</div>", unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    preset_name, run_control, selected_samples, sample_dir = render_sidebar()
    preset = PIPELINE_PRESETS[preset_name]

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Receipt-to-Card Presentation UI</div>
            <h1>Receipt Card Advisor</h1>
            <div class="hero-copy">
                Upload one or more receipts, choose a pipeline version, and turn messy transaction text into a
                credit card recommendation. This UI sits on top of the same perception, planning, and shared
                LLM control pipeline used in the project notebook.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="receipt-panel">
            <strong>{preset_name}</strong><br>
            {preset['description']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload receipt images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="You can upload one receipt or a small batch. The UI will merge all parsed spending into one final recommendation.",
    )

    st.session_state["uploaded_file_map"] = {
        uploaded.name: uploaded.getvalue()
        for uploaded in (uploaded_files or [])
    }

    assets = build_input_assets(uploaded_files, selected_samples, sample_dir=sample_dir)
    render_receipt_preview(assets)

    analyze_clicked = st.button("Analyze Receipts", type="primary", use_container_width=True)
    if analyze_clicked:
        if not assets:
            st.warning("Upload at least one receipt image or choose a sample receipt first.")
        else:
            try:
                with st.spinner("Running the pipeline..."):
                    st.session_state["analysis"] = analyze_receipts(
                        assets=assets,
                        preset=preset,
                        run_control=run_control,
                    )
                st.session_state["chat_messages"] = [
                    {
                        "role": "assistant",
                        "content": (
                            "The receipts are processed. Ask me about the spending profile, "
                            "why a card was recommended, or where the pipeline struggled."
                        ),
                    }
                ]
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")

    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Run the analysis to see the recommendation dashboard.")
        return

    aggregate_profile = analysis["aggregate_profile"]
    aggregate_recommendation = analysis["aggregate_recommendation"]
    top_category, top_amount = pick_top_category(aggregate_profile.category_totals)

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card(
            "Receipts processed",
            str(len(analysis["receipt_results"])),
            "Successful uploads included in the final profile.",
        )
    with metric_columns[1]:
        render_metric_card(
            "Total parsed spend",
            format_currency(aggregate_profile.total_amount),
            "Combined total across all categorized line items.",
        )
    with metric_columns[2]:
        render_metric_card(
            "Top category",
            top_category.title(),
            f"{format_currency(top_amount)} in the strongest spending bucket.",
        )
    with metric_columns[3]:
        render_metric_card(
            "Pipeline",
            preset_name,
            f"{preset['ocr_method']} perception with planning {preset['planning_version']}.",
        )

    dashboard_tabs = st.tabs(["Recommendation", "Receipts", "Chat"])

    with dashboard_tabs[0]:
        chart_col, rec_col = st.columns([1.05, 1.25], gap="large")

        with chart_col:
            st.markdown("### Spending Snapshot")
            st.bar_chart(category_chart_frame(aggregate_profile), height=360)
            st.markdown("### Aggregate Category Totals")
            category_rows = [
                {"category": category.title(), "amount": amount}
                for category, amount in aggregate_profile.category_totals.items()
                if amount > 0
            ]
            if category_rows:
                st.dataframe(pd.DataFrame(category_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No category totals were extracted from the uploaded receipts.")

        with rec_col:
            st.markdown("### Final Recommendation")
            if aggregate_recommendation is None:
                st.info("Recommendation generation was skipped for this run.")
            else:
                primary_col, runner_col = st.columns(2)
                with primary_col:
                    render_recommendation_card(
                        aggregate_recommendation.primary_card,
                        aggregate_recommendation.primary_issuer,
                        aggregate_recommendation.primary_estimated_value,
                        "Primary recommendation",
                    )
                with runner_col:
                    render_recommendation_card(
                        aggregate_recommendation.runner_up_card,
                        aggregate_recommendation.runner_up_issuer,
                        aggregate_recommendation.runner_up_estimated_value,
                        "Runner-up",
                    )

                st.markdown("#### Why this recommendation")
                st.write(aggregate_recommendation.explanation)

                if aggregate_recommendation.caveats:
                    st.markdown("#### Caveats")
                    for caveat in aggregate_recommendation.caveats:
                        st.write(f"- {caveat}")

                if aggregate_recommendation.card_rankings:
                    st.markdown("#### Ranked Options")
                    st.dataframe(
                        pd.DataFrame(aggregate_recommendation.card_rankings),
                        use_container_width=True,
                        hide_index=True,
                    )

                if aggregate_recommendation.sources:
                    st.markdown("#### Research Sources")
                    st.markdown(
                        sources_markup(aggregate_recommendation.sources),
                        unsafe_allow_html=True,
                    )
                    for source in aggregate_recommendation.sources:
                        with st.expander(source["title"]):
                            st.write(source["why_it_matters"])
                            st.markdown(f"[Open source]({source['url']})")

        if analysis["errors"]:
            st.warning("Some receipts could not be processed.")
            st.dataframe(pd.DataFrame(analysis["errors"]), use_container_width=True, hide_index=True)

    with dashboard_tabs[1]:
        st.markdown("### Receipt-by-Receipt Breakdown")
        for index, result in enumerate(analysis["receipt_results"]):
            title = (
                f"{result['display_name']} — "
                f"{result['spending_profile'].merchant or 'Unknown merchant'}"
            )
            with st.expander(title, expanded=index == 0):
                left_col, right_col = st.columns([0.9, 1.1], gap="large")
                with left_col:
                    st.image(result["image_bytes"], use_container_width=True)
                    st.caption(
                        f"Perception: {result['pipeline']['perception_method']} | "
                        f"Planning: {result['pipeline']['planning_version']}"
                    )

                with right_col:
                    st.markdown("#### Parsed Text")
                    st.code(preview_text(result["ocr_result"].text, limit=3500), language=None)
                    if has_reference_labels(result["ocr_result"].image_path):
                        st.caption("Matching reference labels are available for this receipt.")

                st.markdown("#### Category Totals")
                receipt_totals = [
                    {"category": category.title(), "amount": amount}
                    for category, amount in result["spending_profile"].category_totals.items()
                    if amount > 0
                ]
                if receipt_totals:
                    st.dataframe(pd.DataFrame(receipt_totals), use_container_width=True, hide_index=True)
                else:
                    st.info("No category totals were extracted for this receipt.")

                st.markdown("#### Parsed Line Items")
                items_df = line_items_frame(result["spending_profile"].line_items)
                if items_df.empty:
                    st.info("No line items were parsed from this receipt.")
                else:
                    st.dataframe(items_df, use_container_width=True, hide_index=True)

                with st.expander("Raw structured output"):
                    st.json(
                        {
                            "pipeline": result["pipeline"],
                            "ocr": ocr_result_as_dict(result["ocr_result"]),
                            "planning": result["spending_profile"].as_dict(),
                        }
                    )

    with dashboard_tabs[2]:
        st.markdown("### Ask About This Analysis")
        st.caption(
            "This chat is grounded in the current run. It can explain the spend profile, "
            "recommendation logic, and visible OCR issues."
        )

        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "Run an analysis first, then ask follow-up questions here.",
                }
            ]

        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask about the recommendation, the parsed spend, or OCR quality")
        if prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = answer_pipeline_question(prompt, analysis)
                    except Exception as exc:
                        answer = f"Chat failed: {exc}"
                    st.markdown(answer)
            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
