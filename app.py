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
from src.planning import ReceiptPlanner
from src.utils import (
    has_reference_labels,
    list_receipt_images,
    merge_spending_profiles,
    ocr_result_as_dict,
    project_root,
    preferred_receipts_dir,
    preview_text,
    resolve_reference_labeled_image,
    run_receipt_pipeline,
    save_uploaded_bytes,
)


PROJECT_ROOT = project_root()
load_dotenv(PROJECT_ROOT / ".env", override=False)

PIPELINE_PRESETS = {
    "Classic Pipeline (V1)": {
        "pipeline_version": "v1",
        "ocr_method": "tesseract",
        "planning_version": "v1",
        "description": "Tesseract OCR with the classical planning pipeline.",
    },
    "PaddleOCR Pipeline (V2)": {
        "pipeline_version": "v2",
        "ocr_method": "paddleocr",
        "planning_version": "v2",
        "description": "PaddleOCR with the deep-learning planning pipeline.",
    },
    "Structured Reference Text": {
        "pipeline_version": None,
        "ocr_method": "labels",
        "planning_version": "v2",
        "description": "Receipt text reconstructed from dataset annotations for controlled comparison.",
    },
}

METHOD_LABELS = {
    "tesseract": "Tesseract OCR",
    "paddleocr": "PaddleOCR",
    "labels": "Structured Reference Text",
}

PLANNING_LABELS = {
    "v1": "Classical Planning",
    "v2": "Deep Planning",
}


st.set_page_config(
    page_title="Receipt Rewards Advisor",
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
            --paper: #f3f7fb;
            --card: rgba(255, 255, 255, 0.88);
            --ink: #11253f;
            --muted: #60748a;
            --accent: #1d5d8c;
            --accent-deep: #123c61;
            --leaf: #0f7a6c;
            --border: rgba(17, 37, 63, 0.12);
        }

        html, body, [class*="stApp"] {
            color-scheme: light;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29, 93, 140, 0.12), transparent 34%),
                radial-gradient(circle at top right, rgba(15, 122, 108, 0.12), transparent 30%),
                linear-gradient(180deg, #f7fbff 0%, #eef4f8 100%);
            color: var(--ink) !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: transparent !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(251, 253, 255, 0.98), rgba(239, 245, 249, 0.98)) !important;
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
            background: rgba(248, 251, 254, 0.96) !important;
            border: 1px solid rgba(17, 37, 63, 0.14) !important;
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
            background: linear-gradient(135deg, #1d5d8c, #123c61) !important;
            color: #f8fbff !important;
            border: none !important;
            border-radius: 999px !important;
            box-shadow: 0 12px 24px rgba(18, 60, 97, 0.18);
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
            background: rgba(245, 249, 252, 0.88) !important;
            border: 1px solid rgba(17, 37, 63, 0.12) !important;
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
            border: 1px solid rgba(17, 37, 63, 0.12) !important;
            border-radius: 18px !important;
            padding: 0.3rem !important;
        }

        [data-testid="stDataFrame"] * ,
        [data-testid="stTable"] * {
            color: var(--ink) !important;
        }

        [data-testid="stAlert"] {
            background: rgba(248, 251, 254, 0.94) !important;
            border: 1px solid rgba(17, 37, 63, 0.12) !important;
            border-radius: 18px !important;
        }

        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, #1d5d8c, #0f7a6c) !important;
        }

        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(17, 37, 63, 0.1);
            border-radius: 18px;
            padding: 0.3rem 0.75rem;
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(240,247,252,0.94));
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 1.65rem 1.75rem 1.5rem 1.75rem;
            box-shadow: 0 20px 56px rgba(17, 37, 63, 0.08);
            margin-bottom: 1rem;
        }

        .hero-kicker {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--accent-deep);
            margin-bottom: 0.45rem;
            font-weight: 700;
        }

        .hero-copy {
            color: var(--muted);
            max-width: 54rem;
            font-size: 1rem;
            line-height: 1.55;
        }

        .workflow-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.96), rgba(243,248,252,0.92));
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(17, 37, 63, 0.05);
            margin-bottom: 1rem;
        }

        .workflow-label {
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.72rem;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .workflow-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.2rem;
            margin-bottom: 0.35rem;
        }

        .workflow-meta {
            color: var(--muted);
            font-size: 0.93rem;
            line-height: 1.45;
        }

        .workflow-pills {
            margin-top: 0.8rem;
        }

        .workflow-pill {
            display: inline-block;
            margin: 0 0.4rem 0.35rem 0;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: rgba(17, 37, 63, 0.06);
            border: 1px solid rgba(17, 37, 63, 0.1);
            font-size: 0.82rem;
            color: var(--ink);
        }

        .metric-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            min-height: 140px;
            box-shadow: 0 10px 30px rgba(17, 37, 63, 0.05);
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
            background: linear-gradient(145deg, rgba(255,255,255,0.96), rgba(242,247,252,0.9));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            box-shadow: 0 14px 38px rgba(17, 37, 63, 0.08);
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
            border: 1px solid rgba(17, 37, 63, 0.14);
            border-radius: 999px;
            padding: 0.28rem 0.65rem;
            margin: 0.2rem 0.35rem 0 0;
            background: rgba(29, 93, 140, 0.08);
            color: var(--ink);
            font-size: 0.84rem;
        }

        .receipt-caption {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.35rem;
        }

        .receipt-panel {
            background: rgba(255,255,255,0.78);
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


def display_ocr_method(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def display_planning_version(version: str | None) -> str:
    if version is None:
        return "Not set"
    return PLANNING_LABELS.get(version, version.upper())


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


def render_workflow_card(title: str, description: str, ocr_method: str, planning_version: str | None) -> None:
    st.markdown(
        f"""
        <div class="workflow-card">
            <div class="workflow-label">Active Workflow</div>
            <div class="workflow-title">{title}</div>
            <div class="workflow-meta">{description}</div>
            <div class="workflow-pills">
                <span class="workflow-pill">{display_ocr_method(ocr_method)}</span>
                <span class="workflow-pill">{display_planning_version(planning_version)}</span>
            </div>
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
    control_status = "skipped"
    control_message: str | None = None

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
                            "No matching structured reference receipt was found in the local dataset "
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
            try:
                recommender = get_recommender()
                aggregate_recommendation = recommender.recommend_card(aggregate_profile)
                control_status = "succeeded"
            except Exception as exc:
                control_status = "failed"
                control_message = str(exc)
        else:
            control_status = "skipped"

        return {
            "preset": preset,
            "receipt_results": receipt_results,
            "aggregate_profile": aggregate_profile,
            "aggregate_recommendation": aggregate_recommendation,
            "control_status": control_status,
            "control_message": control_message,
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
        rows = [{"category": "No detected spend", "amount": 0.0}]
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
        "control_status": analysis.get("control_status", "unknown"),
        "control_message": analysis.get("control_message"),
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


def render_sidebar() -> tuple[str, bool, list[str], Path]:
    st.sidebar.markdown("## Analysis Settings")
    preset_name = st.sidebar.radio(
        "Workflow",
        list(PIPELINE_PRESETS.keys()),
        help="Choose the pipeline you want to run on the uploaded receipts.",
    )
    run_control = st.sidebar.checkbox(
        "Generate card recommendation",
        value=True,
        help="Turn this off if you only want OCR and spend classification results.",
    )

    sample_dir = preferred_receipts_dir()
    sample_choices = [path.name for path in list_receipt_images(sample_dir)[:12]]
    selected_samples = st.sidebar.multiselect(
        "Use sample receipts",
        sample_choices,
        help="Quick way to demo the app without uploading your own files.",
    )

    st.sidebar.markdown("## How It Works")
    st.sidebar.caption(
        "Receipts are parsed into spend categories, then combined into one spending profile. "
        "The control phase uses an LLM plus live web research to recommend a current credit card."
    )
    return preset_name, run_control, selected_samples, sample_dir


def render_receipt_preview(assets: list[dict[str, Any]]) -> None:
    if not assets:
        return

    st.markdown("### Receipt Queue")
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
            <div class="hero-kicker">Spend Intelligence For Card Rewards</div>
            <h1>Receipt Rewards Advisor</h1>
            <div class="hero-copy">
                Upload one or more receipts, compare how each workflow interprets the spend, and generate a
                credit card recommendation backed by the same perception, planning, and LLM-driven control
                pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_workflow_card(
        title=preset_name,
        description=preset["description"],
        ocr_method=preset["ocr_method"],
        planning_version=preset["planning_version"],
    )

    uploaded_files = st.file_uploader(
        "Upload receipts",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload one receipt or a small batch. All parsed spend will be merged into a single recommendation profile.",
    )

    st.session_state["uploaded_file_map"] = {
        uploaded.name: uploaded.getvalue()
        for uploaded in (uploaded_files or [])
    }

    assets = build_input_assets(uploaded_files, selected_samples, sample_dir=sample_dir)
    render_receipt_preview(assets)

    analyze_clicked = st.button("Run Analysis", type="primary", use_container_width=True)
    if analyze_clicked:
        if not assets:
            st.warning("Add at least one receipt image or choose a sample receipt to begin.")
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
                            "Your receipts are analyzed. Ask about the spending profile, "
                            "recommendation logic, or where OCR quality may have affected the result."
                        ),
                    }
                ]
            except Exception as exc:
                st.error(f"The analysis could not be completed: {exc}")

    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Run an analysis to unlock the recommendation dashboard.")
        return

    aggregate_profile = analysis["aggregate_profile"]
    aggregate_recommendation = analysis["aggregate_recommendation"]
    control_status = analysis.get("control_status", "unknown")
    control_message = analysis.get("control_message")
    top_category, top_amount = pick_top_category(aggregate_profile.category_totals)

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card(
            "Receipts analyzed",
            str(len(analysis["receipt_results"])),
            "Receipts successfully included in the final profile.",
        )
    with metric_columns[1]:
        render_metric_card(
            "Detected spend",
            format_currency(aggregate_profile.total_amount),
            "Combined spend total parsed from all receipt line items.",
        )
    with metric_columns[2]:
        render_metric_card(
            "Top spend category",
            top_category.title(),
            f"{format_currency(top_amount)} in the largest parsed category.",
        )
    with metric_columns[3]:
        render_metric_card(
            "Workflow",
            preset_name,
            f"{display_ocr_method(preset['ocr_method'])} with {display_planning_version(preset['planning_version'])}.",
        )

    dashboard_tabs = st.tabs(["Recommendation", "Receipt Review", "Assistant"])

    with dashboard_tabs[0]:
        chart_col, rec_col = st.columns([1.05, 1.25], gap="large")

        with chart_col:
            st.markdown("### Spending Snapshot")
            st.bar_chart(category_chart_frame(aggregate_profile), height=360)
            st.markdown("### Category Totals")
            category_rows = [
                {"category": category.title(), "amount": amount}
                for category, amount in aggregate_profile.category_totals.items()
                if amount > 0
            ]
            if category_rows:
                st.dataframe(pd.DataFrame(category_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No spend categories were extracted from this run.")

        with rec_col:
            st.markdown("### Final Recommendation")
            if control_status == "failed":
                st.warning(
                    "Live card research did not finish in time for this run. "
                    "Your OCR and spend-classification results are still available below."
                )
                if control_message:
                    st.caption(control_message)
            if aggregate_recommendation is None:
                if control_status == "failed":
                    st.info(
                        "A final card recommendation could not be completed this time, "
                        "but the parsed receipt analysis is still shown."
                    )
                else:
                    st.info("Card recommendation was turned off for this run.")
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
                    st.markdown("#### Ranked Card Options")
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
        st.markdown("### Receipt-by-Receipt Review")
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
                        f"OCR: {display_ocr_method(result['pipeline']['perception_method'])} | "
                        f"Planning: {display_planning_version(result['pipeline']['planning_version'])}"
                    )

                with right_col:
                    st.markdown("#### Parsed Receipt Text")
                    st.code(preview_text(result["ocr_result"].text, limit=3500), language=None)
                    if has_reference_labels(result["ocr_result"].image_path):
                        st.caption("Matching structured annotations are available for this receipt.")

                st.markdown("#### Category Totals")
                receipt_totals = [
                    {"category": category.title(), "amount": amount}
                    for category, amount in result["spending_profile"].category_totals.items()
                    if amount > 0
                ]
                if receipt_totals:
                    st.dataframe(pd.DataFrame(receipt_totals), use_container_width=True, hide_index=True)
                else:
                    st.info("No spend categories were extracted for this receipt.")

                st.markdown("#### Parsed Line Items")
                items_df = line_items_frame(result["spending_profile"].line_items)
                if items_df.empty:
                    st.info("No line items were parsed from this receipt.")
                else:
                    st.dataframe(items_df, use_container_width=True, hide_index=True)

                with st.expander("Technical output"):
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
            "This assistant is grounded in the current run. It can explain the spend profile, "
            "recommendation logic, and visible OCR limitations."
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

        prompt = st.chat_input("Ask about the recommendation, spend profile, or OCR quality")
        if prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = answer_pipeline_question(prompt, analysis)
                    except Exception as exc:
                        answer = f"The assistant could not answer this question: {exc}"
                    st.markdown(answer)
            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
