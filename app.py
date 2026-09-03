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

def main() -> None:
    apply_styles()

    st.markdown("# GuestFlow Analytics")
    st.markdown(
        '<div class="subhead">Hotel Occupancy & Revenue Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="description">Turn guest booking behavior into smarter occupancy and revenue decisions.</div>',
        unsafe_allow_html=True,
    )

    try:
        df = prepare_data()
        db_path = prepare_database(df)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    available_years = sorted(df["arrival_date"].dt.year.unique().tolist())
    available_cities = sorted(df["city"].dropna().unique().tolist())
    filter_col_1, filter_col_2, filter_col_3 = st.columns([1, 2, 1.4])
    with filter_col_1:
        selected_years = st.multiselect(
            "Year",
            options=available_years,
        )
    with filter_col_2:
        selected_months = st.multiselect(
            "Month",
            options=MONTH_ORDER,
        )
    with filter_col_3:
        selected_cities = st.multiselect(
            "City",
            options=available_cities,
        )

    filtered_df = filter_frame(df, selected_years, selected_months, selected_cities)
    date_filter = sql_filter(selected_years, selected_months, selected_cities)

    if filtered_df.empty:
        st.warning("No bookings match the selected year, month, and city filters.")
        st.stop()

    inventory = infer_room_inventory(df)
    kpis = calculate_kpis(filtered_df, inventory)
    segment_df = segment_analysis(db_path, date_filter)
    trend = monthly_trends(db_path, inventory, date_filter)
    season = monthly_seasonality(trend)
    forecast_df = forecast_occupancy(trend)
    tops = top_values(segment_df, trend)

    st.markdown("## Key Performance Indicators")
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        metric_card("Occupancy Rate", f"{kpis['occupancy_rate']:.1f}%")
    with kpi_cols[1]:
        metric_card("Total Revenue", format_money(kpis["total_revenue"]))
    with kpi_cols[2]:
        metric_card("Cancellation Rate", f"{kpis['cancellation_rate']:.1f}%")
    with kpi_cols[3]:
        metric_card("Average Daily Rate", f"₹{kpis['average_adr']:,.2f}")

    st.markdown("## Customer Segment Analysis")
    chart_col, table_col = st.columns([0.9, 1.25])
    with chart_col:
        donut = px.pie(
            segment_df,
            names="Segment",
            values="Bookings",
            hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        donut.update_traces(textposition="inside", textinfo="percent+label")
        donut.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(donut, use_container_width=True)
    with table_col:
        st.dataframe(
            style_segment_table(
                segment_df[
                    [
                        "Segment",
                        "Bookings",
                        "Cancellation Rate",
                        "Average ADR",
                        "Revenue Contribution",
                    ]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    highest_cancel = segment_df.loc[segment_df["Cancellation Rate"].idxmax()]
    insight(
        f"{highest_cancel['Segment']} customers have the highest cancellation rate at "
        f"{highest_cancel['Cancellation Rate']:.1f}% and may contribute significantly to occupancy uncertainty."
    )

    st.markdown("## Occupancy Trends")
    line = px.line(
        trend,
        x="arrival_month",
        y="occupancy_rate",
        markers=True,
        labels={"arrival_month": "Month", "occupancy_rate": "Occupancy Rate (%)"},
    )
    line.update_traces(line_color=BLUE, marker_color=BLUE)
    line.update_layout(
        yaxis_ticksuffix="%",
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(line, use_container_width=True)
    if len(trend) > 1:
        peak = trend.loc[trend["occupancy_rate"].idxmax()]
        low = trend.loc[trend["occupancy_rate"].idxmin()]
        insight(
            f"Occupancy peaked in {peak['arrival_month'].strftime('%B %Y')} at "
            f"{peak['occupancy_rate']:.1f}% and fell lowest in "
            f"{low['arrival_month'].strftime('%B %Y')} at {low['occupancy_rate']:.1f}%."
        )

    st.markdown("## Cancellation Overview")
    cancel_bar = px.bar(
        segment_df.sort_values("Cancellation Rate", ascending=False),
        x="Segment",
        y="Cancellation Rate",
        color="Cancellation Rate",
        color_continuous_scale=[[0, GREEN], [0.5, BLUE], [1, RED]],
        labels={"Cancellation Rate": "Cancellation Rate (%)"},
    )
    cancel_bar.update_layout(
        yaxis_ticksuffix="%",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(cancel_bar, use_container_width=True)
    insight(
        f"<strong>Highest Cancellation Risk</strong><br>{highest_cancel['Segment']} has the highest "
        f"cancellation rate at {highest_cancel['Cancellation Rate']:.1f}%."
    )

    st.markdown("## Seasonal Occupancy")
    if not season.empty:
        peak_month = season.loc[season["occupancy_rate"].idxmax()]
        low_month = season.loc[season["occupancy_rate"].idxmin()]
        volatile_month = season.loc[season["volatility"].idxmax()]

        season_chart = px.bar(
            season,
            x="month",
            y="occupancy_rate",
            color="occupancy_rate",
            color_continuous_scale=[[0, "#DBEAFE"], [1, BLUE]],
            category_orders={"month": MONTH_ORDER},
            labels={"month": "Month", "occupancy_rate": "Average Occupancy (%)"},
        )
        season_chart.update_layout(
            yaxis_ticksuffix="%",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(season_chart, use_container_width=True)

        seasonal_cols = st.columns(3)
        with seasonal_cols[0]:
            metric_card("Peak Month", f"{peak_month['month']} - {peak_month['occupancy_rate']:.1f}%")
        with seasonal_cols[1]:
            metric_card("Lowest Month", f"{low_month['month']} - {low_month['occupancy_rate']:.1f}%")
        with seasonal_cols[2]:
            metric_card(
                "Highest Volatility",
                f"{volatile_month['month']} - {volatile_month['volatility']:.1f} pts",
            )

    st.markdown("## Occupancy Forecast")
    if not forecast_df.empty:
        forecast_chart = px.line(
            forecast_df,
            x="arrival_month",
            y="occupancy_rate",
            color="type",
            markers=True,
            color_discrete_map={"Historical": BLUE, "Predicted": GREEN},
            labels={"arrival_month": "Month", "occupancy_rate": "Occupancy Rate (%)", "type": ""},
        )
        forecast_chart.update_layout(
            yaxis_ticksuffix="%",
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(forecast_chart, use_container_width=True)
        insight(forecast_direction(forecast_df))

    st.markdown("## Revenue Recommendations")
    recs = build_recommendations(segment_df, trend, forecast_df)
    rec_cols = st.columns(2)
    for index, rec in enumerate(recs):
        priority_class = rec["priority"].lower()
        with rec_cols[index % 2]:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">{rec['title']}</div>
                    <p>{rec['explanation']}</p>
                    <span class="priority priority-{priority_class}">{rec['priority']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("## Executive Summary")
    recommended_action = recs[0]["title"] if recs else "Review demand patterns"
    summary_cols = st.columns(4)
    with summary_cols[0]:
        metric_card("Top Revenue Segment", tops.get("top_revenue_segment", "N/A"))
    with summary_cols[1]:
        metric_card("Highest Cancellation Segment", tops.get("highest_cancellation_segment", "N/A"))
    with summary_cols[2]:
        metric_card("Peak Occupancy Period", tops.get("peak_occupancy_period", "N/A"))
    with summary_cols[3]:
        metric_card("Recommended Action", recommended_action)

    st.caption(
        "Occupancy is calculated from completed occupied room-nights against the total room inventory in the selected hotels and cities."
    )


if __name__ == "__main__":
    main()
