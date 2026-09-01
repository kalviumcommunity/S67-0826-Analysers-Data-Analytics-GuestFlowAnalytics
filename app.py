from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.analytics import (
    calculate_kpis,
    infer_room_inventory,
    monthly_seasonality,
    monthly_trends,
    segment_analysis,
    top_values,
)
from modules.data_processing import MONTH_ORDER, load_and_clean_data
from modules.database import initialize_database
from modules.forecasting import forecast_direction, forecast_occupancy
from modules.recommendations import build_recommendations


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "guestflow_cleaned_data.csv"
DB_PATH = BASE_DIR / "database" / "hotel_data.db"

BLUE = "#2563EB"
GREEN = "#10B981"
RED = "#EF4444"
TEXT = "#1E293B"
BG = "#F8FAFC"


st.set_page_config(
    page_title="GuestFlow Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_styles() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: {BG};
                color: {TEXT};
            }}
            [data-testid="stHeader"] {{
                background: rgba(248, 250, 252, 0.92);
            }}
            .block-container {{
                max-width: 1180px;
                padding-top: 2.2rem;
                padding-bottom: 4rem;
            }}
            h1, h2, h3, p, div {{
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            h1 {{
                color: {TEXT};
                font-size: 2.65rem;
                letter-spacing: 0;
                margin-bottom: 0.25rem;
            }}
            h2 {{
                color: {TEXT};
                border-top: 1px solid #E2E8F0;
                padding-top: 2rem;
                margin-top: 2rem;
            }}
            .subhead {{
                color: #475569;
                font-size: 1.1rem;
                margin-bottom: 0.3rem;
            }}
            .description {{
                color: #64748B;
                font-size: 1rem;
                margin-bottom: 1.5rem;
            }}
            .card {{
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
                padding: 1.1rem;
                height: 100%;
            }}
            .metric-label {{
                color: #64748B;
                font-size: 0.85rem;
                font-weight: 650;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}
            .metric-value {{
                color: {TEXT};
                font-size: 1.8rem;
                font-weight: 750;
                margin-top: 0.25rem;
            }}
            .insight {{
                background: #FFFFFF;
                border-left: 4px solid {BLUE};
                border-radius: 8px;
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
                padding: 1rem 1.1rem;
                color: #334155;
                margin-top: 1rem;
            }}
            .priority {{
                display: inline-block;
                border-radius: 999px;
                padding: 0.15rem 0.55rem;
                color: #FFFFFF;
                font-size: 0.78rem;
                font-weight: 700;
            }}
            .priority-high {{ background: {RED}; }}
            .priority-medium {{ background: {BLUE}; }}
            .priority-low {{ background: {GREEN}; }}
            div[data-testid="stDataFrame"] {{
                background: #FFFFFF;
                border-radius: 8px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Cleaning hotel booking dataset...")
def prepare_data() -> pd.DataFrame:
    return load_and_clean_data(DATA_PATH)


@st.cache_resource(show_spinner="Building SQLite analytics database...")
def prepare_database(df: pd.DataFrame) -> str:
    initialize_database(df, DB_PATH)
    return str(DB_PATH)


def format_money(value: float) -> str:
    return f"₹{value:,.0f}"


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_filter(years: list[int], months: list[str], cities: list[str]) -> str:
    conditions = []
    if years:
        year_list = ", ".join(str(year) for year in years)
        conditions.append(f"CAST(strftime('%Y', arrival_date) AS INTEGER) IN ({year_list})")
    if months:
        month_numbers = [MONTH_ORDER.index(month) + 1 for month in months]
        month_list = ", ".join(f"'{month:02d}'" for month in month_numbers)
        conditions.append(f"strftime('%m', arrival_date) IN ({month_list})")
    if cities:
        city_list = ", ".join(sql_text(city) for city in cities)
        conditions.append(f"city IN ({city_list})")
    clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    return clause


def filter_frame(
    df: pd.DataFrame, years: list[int], months: list[str], cities: list[str]
) -> pd.DataFrame:
    filtered = df.copy()
    if years:
        filtered = filtered[filtered["arrival_date"].dt.year.isin(years)]
    if months:
        filtered = filtered[filtered["month_name"].isin(months)]
    if cities:
        filtered = filtered[filtered["city"].isin(cities)]
    return filtered


def style_segment_table(df: pd.DataFrame):
    high_cancel_idx = df["Cancellation Rate"].idxmax() if not df.empty else None
    display = df.copy()
    display["Cancellation Rate"] = display["Cancellation Rate"].map(lambda value: f"{value:.1f}%")
    display["Average ADR"] = display["Average ADR"].fillna(0).map(lambda value: f"₹{value:,.2f}")
    display["Revenue Contribution"] = display["Revenue Contribution"].map(format_money)

    def highlight(row):
        return [
            "background-color: #FEF2F2; color: #991B1B; font-weight: 700;"
            if row.name == high_cancel_idx
            else ""
            for _ in row
        ]

    return display.style.apply(highlight, axis=1)
