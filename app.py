from __future__ import annotations

import html
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
import streamlit.components.v1 as components

from src.control import CreditCardRecommender
from src.perception import ReceiptPerception
from src.planning import ReceiptPlanner
from src.utils import (
    has_reference_labels,
    list_receipt_images,
    merge_spending_profiles,
    ocr_result_as_dict,
    profile_categorized_total,
    profile_display_total,
    profile_reported_total,
    profile_total_delta,
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

TEAM_MEMBERS = [
    "Aesha Gandhi",
    "Gaurav Law",
    "Pranshul Bhatnagar",
]

EXAMPLE_RECEIPT_INPUT = """Receipt image: grocery purchase
WHOLE FOODS MARKET
Organic Apples      5.49
Salmon Fillet      14.99
Sparkling Water     4.29
Tax                 2.04
Total              26.81"""

EXAMPLE_PERCEPTION_OUTPUT = """Classical OCR (V1):
WHOLE F00DS MARKET
Organic Apples 5.49
Salmon Fillet 14.99
Sparkling Water 4.29
Tax 2.04
Total 26.81

DL OCR (V2):
WHOLE FOODS MARKET
Organic Apples 5.49
Salmon Fillet 14.99
Sparkling Water 4.29
Tax 2.04
Total 26.81"""

EXAMPLE_PLANNING_OUTPUT = """spending_profile = {
  merchant: "Whole Foods Market",
  reported_total: 26.81,
  category_totals: {
    groceries: 24.77,
    other: 2.04
  },
  top_category: "groceries",
  parsed_line_items: 4
}"""

EXAMPLE_CONTROL_OUTPUT = """LLM recommendation:
Primary card: Amex Gold
Runner-up: Blue Cash Preferred

Reasoning:
- groceries dominate the spend
- supermarket rewards are the best fit
- card comparison is backed by live web research"""

RESOURCE_CACHE_VERSION = "2026-04-21-profile-compat-v1"


st.set_page_config(
    page_title="Receipt Rewards Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_perception(cache_version: str = RESOURCE_CACHE_VERSION) -> ReceiptPerception:
    _ = cache_version
    return ReceiptPerception()


@st.cache_resource(show_spinner=False)
def get_planner(cache_version: str = RESOURCE_CACHE_VERSION) -> ReceiptPlanner:
    _ = cache_version
    return ReceiptPlanner()


@st.cache_resource(show_spinner=False)
def get_recommender(cache_version: str = RESOURCE_CACHE_VERSION) -> CreditCardRecommender:
    _ = cache_version
    return CreditCardRecommender()


@st.cache_resource(show_spinner=False)
def get_chat_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


@st.cache_data(show_spinner=False)
def load_flowchart_html() -> str | None:
    flowchart_path = PROJECT_ROOT / "receipt-rewards-technical-flowchart.html"
    if not flowchart_path.exists():
        return None
    return flowchart_path.read_text(encoding="utf-8")


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

        .intro-shell {
            background: linear-gradient(135deg, rgba(255,255,255,0.97), rgba(239,247,252,0.94));
            border: 1px solid var(--border);
            border-radius: 30px;
            padding: 1.85rem 1.9rem 1.75rem 1.9rem;
            box-shadow: 0 22px 58px rgba(17, 37, 63, 0.08);
            margin-bottom: 1rem;
        }

        .intro-kicker {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.76rem;
            color: var(--accent-deep);
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .intro-copy {
            color: var(--muted);
            max-width: 56rem;
            font-size: 1.02rem;
            line-height: 1.6;
        }

        .team-pills {
            margin-top: 0.95rem;
        }

        .team-pill {
            display: inline-block;
            margin: 0 0.45rem 0.4rem 0;
            padding: 0.34rem 0.8rem;
            border-radius: 999px;
            background: rgba(17, 37, 63, 0.06);
            border: 1px solid rgba(17, 37, 63, 0.1);
            font-size: 0.88rem;
        }

        .intro-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.8fr);
            gap: 0.9rem;
            margin-top: 1rem;
        }

        .intro-panel {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(17, 37, 63, 0.1);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            height: 100%;
        }

        .intro-panel-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.22rem;
            margin-bottom: 0.28rem;
        }

        .intro-panel-copy {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.5;
        }

        .intro-mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.95rem;
        }

        .intro-mini-card {
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(17, 37, 63, 0.1);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            height: 100%;
        }

        .intro-mini-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.69rem;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 0.22rem;
        }

        .intro-mini-title {
            font-weight: 700;
            margin-bottom: 0.24rem;
        }

        .intro-mini-copy {
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.45;
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

        .architecture-shell {
            background: linear-gradient(145deg, rgba(255,255,255,0.96), rgba(241,247,252,0.94));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.25rem 1.3rem;
            box-shadow: 0 14px 34px rgba(17, 37, 63, 0.06);
            margin-bottom: 1rem;
        }

        .architecture-kicker {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.73rem;
            color: var(--accent-deep);
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .architecture-copy {
            color: var(--muted);
            max-width: 58rem;
            line-height: 1.55;
            margin-bottom: 0.9rem;
        }

        .architecture-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.9rem;
        }

        .architecture-phase {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(17, 37, 63, 0.1);
            border-radius: 20px;
            padding: 1rem 1.05rem;
        }

        .architecture-phase-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.15rem;
            margin-bottom: 0.3rem;
        }

        .architecture-phase-text {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .architecture-track {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(17, 37, 63, 0.1);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            height: 100%;
        }

        .architecture-track-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.2rem;
            margin-bottom: 0.25rem;
        }

        .architecture-track-subtitle {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
            margin-bottom: 0.8rem;
        }

        .architecture-step {
            background: rgba(17, 37, 63, 0.04);
            border: 1px solid rgba(17, 37, 63, 0.08);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.7rem;
        }

        .architecture-step-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.7rem;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .architecture-step-title {
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .architecture-step-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.4;
        }

        .architecture-shared {
            background: linear-gradient(135deg, rgba(29,93,140,0.1), rgba(15,122,108,0.1));
            border: 1px solid rgba(17, 37, 63, 0.1);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            margin-top: 1rem;
        }

        .architecture-pills {
            margin-top: 0.7rem;
        }

        .architecture-pill {
            display: inline-block;
            margin: 0 0.45rem 0.35rem 0;
            padding: 0.28rem 0.68rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(17, 37, 63, 0.1);
            font-size: 0.82rem;
        }

        .example-shell {
            margin-top: 1rem;
        }

        .example-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.8rem;
        }

        .example-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(17, 37, 63, 0.1);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            height: 100%;
            box-shadow: 0 10px 28px rgba(17, 37, 63, 0.05);
        }

        .example-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.69rem;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 0.24rem;
        }

        .example-title {
            font-weight: 700;
            margin-bottom: 0.28rem;
        }

        .example-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
            margin-bottom: 0.75rem;
        }

        .example-pre {
            white-space: pre-wrap;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.8rem;
            line-height: 1.45;
            background: rgba(17, 37, 63, 0.04);
            border: 1px solid rgba(17, 37, 63, 0.08);
            border-radius: 14px;
            padding: 0.8rem 0.85rem;
            color: var(--ink);
        }

        .example-note {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.65rem;
        }

        @media (max-width: 1100px) {
            .intro-grid,
            .intro-mini-grid,
            .example-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 900px) {
            .architecture-grid {
                grid-template-columns: 1fr;
            }
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


def render_intro_tab() -> None:
    team_markup = "".join(f'<span class="team-pill">{member}</span>' for member in TEAM_MEMBERS)
    st.markdown(
        f"""
        <div class="intro-shell">
            <div class="intro-kicker">Project Introduction</div>
            <h1>Receipt-to-Card Recommendation Agent</h1>
            <div class="intro-copy">
                This project tackles a practical but messy problem: people make purchases every day, but they usually
                do not know which credit card best matches their real spending behavior. We use receipt images as the
                starting point, transform them into a structured spending profile, and then use an agentic control
                phase to recommend a card backed by live research.
            </div>
            <div class="team-pills">{team_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-grid">
            <div class="intro-panel">
                <div class="intro-panel-title">What Problem Are We Solving?</div>
                <div class="intro-panel-copy">
                    Receipts are difficult inputs for an agent because they are visual, noisy, and inconsistent. They
                    contain small text, summary sections, taxes, tips, totals, and merchant-specific layouts. At the
                    same time, they capture real spending behavior much more directly than a user simply guessing their
                    monthly habits.
                </div>
            </div>
            <div class="intro-panel">
                <div class="intro-panel-title">What Does Our Agent Do?</div>
                <div class="intro-panel-copy">
                    The agent reads receipt images, extracts text, builds a spending profile, and recommends a credit
                    card. The system is organized around the three required phases from class: perception, planning,
                    and control.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project Framing")
    st.markdown(
        """
        <div class="intro-mini-grid">
            <div class="intro-mini-card">
                <div class="intro-mini-label">Input</div>
                <div class="intro-mini-title">Receipt Images</div>
                <div class="intro-mini-copy">
                    The agent starts from uploaded receipt photos, so the perception stage is genuinely visual rather
                    than text-only.
                </div>
            </div>
            <div class="intro-mini-card">
                <div class="intro-mini-label">Core Goal</div>
                <div class="intro-mini-title">Turn Messy Purchases Into Structured Spend</div>
                <div class="intro-mini-copy">
                    We convert OCR text into merchant, line items, categories, receipt totals, and a spending profile
                    that the recommendation agent can reason over.
                </div>
            </div>
            <div class="intro-mini-card">
                <div class="intro-mini-label">Output</div>
                <div class="intro-mini-title">Actionable Card Recommendation</div>
                <div class="intro-mini-copy">
                    The control phase compares current cards and returns a primary recommendation, runner-up, and
                    explanation using the parsed spending profile.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Why Two Agent Versions?")
    st.markdown(
        """
        <div class="architecture-track">
            <div class="architecture-track-subtitle">
                The project includes both a non-DL and a DL version of the agent. This lets us compare how better
                perception and planning change the quality of the final recommendation while keeping the task and the
                control phase consistent.
            </div>
            <div class="architecture-pills">
                <span class="architecture-pill">Version 1: Classical OCR + Classical Planning</span>
                <span class="architecture-pill">Version 2: PaddleOCR + Deep Planning</span>
                <span class="architecture-pill">Shared LLM Control Phase</span>
                <span class="architecture-pill">Same Task, Fair Comparison</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results_tab() -> None:
    st.markdown(
        """
        <div class="intro-shell">
            <div class="intro-kicker">Project Wrap-Up</div>
            <h1>Results, Lessons, and Next Steps</h1>
            <div class="intro-copy">
                This final page summarizes what we observed across the non-DL and DL agents, what we learned from
                evaluation, and what we would improve next if we continued the project.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Overall Results")
    result_cols = st.columns(3, gap="large")
    with result_cols[0]:
        st.markdown(
            """
            <div class="architecture-step">
                <div class="architecture-step-label">Result 1</div>
                <div class="architecture-step-title">The Non-DL Agent Was a Useful Baseline</div>
                <div class="architecture-step-copy">
                    The classical pipeline gave an interpretable baseline and worked reasonably well on cleaner receipts,
                    but it was more brittle when merchant names or line-item wording varied.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_cols[1]:
        st.markdown(
            """
            <div class="architecture-step">
                <div class="architecture-step-label">Result 2</div>
                <div class="architecture-step-title">The DL Agent Was More Robust</div>
                <div class="architecture-step-copy">
                    PaddleOCR plus deep planning handled noisy layouts and restaurant-style receipts more reliably,
                    especially when receipt structure was irregular or OCR text was incomplete.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_cols[2]:
        st.markdown(
            """
            <div class="architecture-step">
                <div class="architecture-step-label">Result 3</div>
                <div class="architecture-step-title">Perception Quality Drove Downstream Quality</div>
                <div class="architecture-step-copy">
                    The structured reference-text workflow showed that when the text input is cleaner, planning and
                    recommendation quality improve noticeably. OCR remained the main bottleneck in the full pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Lessons Learned")
    lesson_cols = st.columns(2, gap="large")
    with lesson_cols[0]:
        st.markdown(
            """
            <div class="architecture-track">
                <div class="architecture-track-title">What Was Hard</div>
                <div class="architecture-track-subtitle">
                    Receipt understanding is difficult because totals, taxes, tips, and payment lines often look similar
                    to real purchased items.
                </div>
                <div class="architecture-step">
                    <div class="architecture-step-label">Lesson</div>
                    <div class="architecture-step-title">OCR Errors Compound</div>
                    <div class="architecture-step-copy">
                        Small perception errors can distort category totals, spend profiles, and final recommendations.
                    </div>
                </div>
                <div class="architecture-step">
                    <div class="architecture-step-label">Lesson</div>
                    <div class="architecture-step-title">Receipt Structure Matters</div>
                    <div class="architecture-step-copy">
                        The agent needs to separate line items from summary lines before categorization; otherwise totals
                        and payment lines can get counted incorrectly.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with lesson_cols[1]:
        st.markdown(
            """
            <div class="architecture-track">
                <div class="architecture-track-title">What Helped</div>
                <div class="architecture-track-subtitle">
                    Keeping the interfaces between perception, planning, and control stable made the project much easier
                    to debug, compare, and present.
                </div>
                <div class="architecture-step">
                    <div class="architecture-step-label">Lesson</div>
                    <div class="architecture-step-title">Shared Control Made Comparison Fair</div>
                    <div class="architecture-step-copy">
                        Using the same LLM-based control phase for both agents made it easier to isolate the effect of
                        perception and planning choices.
                    </div>
                </div>
                <div class="architecture-step">
                    <div class="architecture-step-label">Lesson</div>
                    <div class="architecture-step-title">Transparent UI Was Important</div>
                    <div class="architecture-step-copy">
                        Showing receipt image, OCR text, spending profile, and final recommendation in one interface made
                        failures easier to diagnose and explain.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Future Steps")
    future_cols = st.columns(3, gap="large")
    with future_cols[0]:
        st.markdown(
            """
            <div class="architecture-step">
                <div class="architecture-step-label">Next Step</div>
                <div class="architecture-step-title">Improve Receipt Understanding</div>
                <div class="architecture-step-copy">
                    Strengthen item extraction and summary-line separation so tax, tip, discounts, and totals are
                    handled more consistently across merchants.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with future_cols[1]:
        st.markdown(
            """
            <div class="architecture-step">
                <div class="architecture-step-label">Next Step</div>
                <div class="architecture-step-title">Expand Evaluation</div>
                <div class="architecture-step-copy">
                    Add more receipts, more merchant types, and clearer quantitative metrics for OCR quality, category
                    accuracy, and recommendation usefulness.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with future_cols[2]:
        st.markdown(
            """
            <div class="architecture-step">
                <div class="architecture-step-label">Next Step</div>
                <div class="architecture-step-title">Broaden Card Research</div>
                <div class="architecture-step-copy">
                    Expand the recommendation space with a richer set of current cards, benefits, and issuer rules to
                    make the control phase more comprehensive.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Closing")
    st.markdown(
        """
        <div class="architecture-track">
            <div class="architecture-track-title">Final Takeaway</div>
            <div class="architecture-track-subtitle">
                This project uses an image-based agent that includes perception,
                planning, and control, and it compares a non-DL and DL version on the same task.
            </div>
            <div class="architecture-pills">
                <span class="architecture-pill">Visual receipt input</span>
                <span class="architecture-pill">Two agent versions</span>
                <span class="architecture-pill">Shared LLM control phase</span>
                <span class="architecture-pill">Evaluation-oriented design</span>
                <span class="architecture-pill">Transparent end-to-end UI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_stage_card(label: str, title: str, copy: str, content: str) -> None:
    st.markdown(
        f"""
        <div class="example-card">
            <div class="example-label">{html.escape(label)}</div>
            <div class="example-title">{html.escape(title)}</div>
            <div class="example-copy">{html.escape(copy)}</div>
            <div class="example-pre">{html.escape(content)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flowchart_section() -> None:
    st.markdown("### Technical Flow Diagram")
    st.caption(
        "This diagram gives a visual overview of how receipt inputs move through perception, planning, and control."
    )

    flowchart_html = load_flowchart_html()
    if flowchart_html is None:
        st.info("The technical flowchart file was not found in the project directory.")
        return

    expanded_flowchart_html = flowchart_html.replace(
        "max-width: 1100px;",
        "width: 100%; max-width: none;",
    ).replace(
        "padding: 32px 32px 64px;",
        "padding: 10px 0 32px;",
    ).replace(
        "margin: 0 auto;",
        "margin: 0;",
    )

    components.html(expanded_flowchart_html, width=1800, height=1120, scrolling=False)


def render_architecture_example_section() -> None:
    st.markdown("### Example Stage-by-Stage Walkthrough")
    st.markdown(
        """
        <div class="example-shell">
            <div class="example-note">
                This is an illustrative example that shows the kind of data each stage produces. The exact text,
                totals, and recommendation will vary by receipt and workflow.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    example_columns = st.columns(4, gap="large")
    with example_columns[0]:
        render_example_stage_card(
            label="Step 1",
            title="Input Receipt",
            copy="The agent begins with a receipt image containing line items, tax, and a final total.",
            content=EXAMPLE_RECEIPT_INPUT,
        )
    with example_columns[1]:
        render_example_stage_card(
            label="Step 2",
            title="Perception Output",
            copy="OCR converts the receipt image into text. The classical and DL pipelines may differ in how cleanly they read the same receipt.",
            content=EXAMPLE_PERCEPTION_OUTPUT,
        )
    with example_columns[2]:
        render_example_stage_card(
            label="Step 3",
            title="Planning Output",
            copy="Planning turns raw text into a structured spending profile with merchant, totals, and category-level spend.",
            content=EXAMPLE_PLANNING_OUTPUT,
        )
    with example_columns[3]:
        render_example_stage_card(
            label="Step 4",
            title="Control Output",
            copy="The LLM-based control phase uses the spending profile plus live card research to produce the final recommendation.",
            content=EXAMPLE_CONTROL_OUTPUT,
        )


def render_architecture_tab(active_preset_name: str, active_preset: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="architecture-shell">
            <div class="architecture-kicker">Architecture Overview</div>
            <h3>Perception, Planning, and Control</h3>
            <div class="architecture-copy">
                The system is built as a three-stage agent pipeline. Perception converts receipt images into text,
                planning turns that text into structured spending behavior, and control recommends a credit card using
                an LLM with live web research. The two required agent versions differ in perception and planning, while
                sharing the same control phase for a fair comparison.
            </div>
            <div class="architecture-grid">
                <div class="architecture-phase">
                    <div class="architecture-phase-title">Perception</div>
                    <div class="architecture-phase-text">Image-based OCR extracts readable receipt text from uploaded receipt photos.</div>
                </div>
                <div class="architecture-phase">
                    <div class="architecture-phase-title">Planning</div>
                    <div class="architecture-phase-text">Receipt text is transformed into merchant, line items, category totals, and a spending profile.</div>
                </div>
                <div class="architecture-phase">
                    <div class="architecture-phase-title">Control</div>
                    <div class="architecture-phase-text">An LLM plus web tools evaluates current cards and recommends the best fit for the parsed spend profile.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_col, compare_col = st.columns([1.05, 1.25], gap="large")

    with current_col:
        render_workflow_card(
            title=f"Active Workflow: {active_preset_name}",
            description=active_preset["description"],
            ocr_method=active_preset["ocr_method"],
            planning_version=active_preset["planning_version"],
        )

        st.markdown(
            """
            <div class="architecture-track">
                <div class="architecture-track-title">Evaluation Support</div>
                <div class="architecture-track-subtitle">
                    Structured Reference Text uses the dataset annotation JSON to reconstruct receipt text.
                    It is helpful for showing whether downstream errors come from OCR quality or from later planning logic.
                </div>
                <div class="architecture-step">
                    <div class="architecture-step-label">Reference Input</div>
                    <div class="architecture-step-title">Structured Reference Text</div>
                    <div class="architecture-step-copy">
                        This mode is useful in demos and evaluation because it approximates a cleaner upper bound for
                        the rest of the pipeline.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with compare_col:
        comparison_markup = f"""
        <div class="architecture-track">
            <div class="architecture-track-title">Two Required Agent Versions</div>
            <div class="architecture-track-subtitle">
                Both agents perform the same task, but they differ in how they handle perception and planning.
            </div>
            <div class="architecture-step">
                <div class="architecture-step-label">Version 1</div>
                <div class="architecture-step-title">Non-DL Agent</div>
                <div class="architecture-step-copy">
                    <strong>Perception:</strong> Tesseract OCR with classical image preprocessing such as grayscale conversion,
                    thresholding, and deskewing. In this project, Tesseract represents the non-DL approach because it
                    relies on a traditional OCR pipeline and pattern-based character recognition rather than modern deep
                    neural text detectors and recognizers.<br>
                    <strong>Planning:</strong> Classical parsing and non-DL categorization logic that converts OCR text into spending totals.
                </div>
            </div>
            <div class="architecture-step">
                <div class="architecture-step-label">Version 2</div>
                <div class="architecture-step-title">DL Agent</div>
                <div class="architecture-step-copy">
                    <strong>Perception:</strong> PaddleOCR for stronger receipt text extraction. It represents the DL
                    approach because it uses deep-learning models to detect text regions and recognize the text content,
                    which makes it more robust on noisy, irregular receipt layouts.<br>
                    <strong>Planning:</strong> Deep planning with transformer-based semantic classification and more flexible receipt understanding.
                </div>
            </div>
            <div class="architecture-shared">
                <div class="architecture-step-label">Shared Control Phase</div>
                <div class="architecture-step-title">LLM Recommendation Agent</div>
                <div class="architecture-step-copy">
                    Both versions use the same control phase: an OpenAI model with web search and webpage tools. This
                    keeps the comparison fair and makes it easier to see how perception and planning quality affect the
                    final recommendation.
                </div>
                <div class="architecture-pills">
                    <span class="architecture-pill">Live card research</span>
                    <span class="architecture-pill">LLM reasoning</span>
                    <span class="architecture-pill">Structured JSON output</span>
                    <span class="architecture-pill">Shared across both agents</span>
                </div>
            </div>
        </div>
        """
        st.markdown(comparison_markup, unsafe_allow_html=True)

    st.markdown("### Why This Design")
    reason_cols = st.columns(3, gap="large")
    with reason_cols[0]:
        st.markdown(
            """
            <div class="architecture-step">
                <div class="architecture-step-label">Design Choice</div>
                <div class="architecture-step-title">Image-Based Input</div>
                <div class="architecture-step-copy">
                    The project starts from receipt images rather than text so the perception phase is genuinely visual
                    and satisfies the course requirement.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with reason_cols[1]:
        st.markdown(
            """
            <div class="architecture-step">
                <div class="architecture-step-label">Design Choice</div>
                <div class="architecture-step-title">Two Comparable Agents</div>
                <div class="architecture-step-copy">
                    The non-DL and DL versions share the same task and control phase, making it easier to compare them
                    in evaluation and explain the effect of the upstream modeling choices.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with reason_cols[2]:
        st.markdown(
            """
            <div class="architecture-step">
                <div class="architecture-step-label">Design Choice</div>
                <div class="architecture-step-title">Stable Interface</div>
                <div class="architecture-step-copy">
                    Every perception method outputs receipt text, and every planning method outputs the same spending
                    profile structure. That stability keeps the rest of the pipeline reusable.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_architecture_example_section()
    render_flowchart_section()


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


def profile_summary_frame(profile) -> pd.DataFrame:
    rows = [
        {"field": "Merchant", "value": profile.merchant or "Unknown"},
        {"field": "Displayed receipt total", "value": format_currency(profile_display_total(profile))},
        {
            "field": "Reported total from receipt",
            "value": format_currency(profile_reported_total(profile))
            if profile_reported_total(profile) is not None
            else "Not found",
        },
        {"field": "Categorized spend total", "value": format_currency(profile_categorized_total(profile))},
        {
            "field": "Difference",
            "value": format_currency(profile_total_delta(profile))
            if profile_total_delta(profile) is not None
            else "N/A",
        },
        {"field": "Planning version", "value": display_planning_version(profile.planner_version)},
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
    st.sidebar.markdown("## Demo Settings")
    st.sidebar.caption("Presentation flow: Introduction -> Architecture -> Demo")
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
    top_tabs = st.tabs(["Introduction", "Architecture", "Demo", "Results & Closing"])

    with top_tabs[0]:
        render_intro_tab()

    with top_tabs[1]:
        render_architecture_tab(active_preset_name=preset_name, active_preset=preset)

    with top_tabs[2]:
        st.markdown(
            f"""
            <div class="hero-shell">
                <div class="hero-kicker">Interactive Demo</div>
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
        if analysis:
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
                    "Receipt total",
                    format_currency(aggregate_profile.total_amount),
                    "Uses detected total or balance lines when available, with categorized spend as fallback.",
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

        demo_tabs = st.tabs(["Recommendation", "Receipt Review", "Assistant"])

        with demo_tabs[0]:
            if not analysis:
                st.info("Run an analysis to view recommendation results for the selected workflow.")
            else:
                chart_col, rec_col = st.columns([1.05, 1.25], gap="large")

                with chart_col:
                    st.markdown("### Spending Snapshot")
                    st.bar_chart(category_chart_frame(aggregate_profile), height=360)
                    st.caption(
                        "Category totals are the parsed spend breakdown used for recommendation. "
                        "The receipt total above is shown separately so OCR item over-counting does not distort the headline total."
                    )
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

        with demo_tabs[1]:
            if not analysis:
                st.info("Run an analysis to inspect receipt text, parsed items, and category totals.")
            else:
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

                        st.markdown("#### Spending Profile Summary")
                        st.dataframe(
                            profile_summary_frame(result["spending_profile"]),
                            use_container_width=True,
                            hide_index=True,
                        )

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

        with demo_tabs[2]:
            if not analysis:
                st.markdown("### Ask About This Analysis")
                st.caption(
                    "Run an analysis first, then use this assistant to ask follow-up questions about OCR quality, spend categories, or the recommendation."
                )
            else:
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

    with top_tabs[3]:
        render_results_tab()


if __name__ == "__main__":
    main()
