from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from modules.database import query


def infer_room_inventory(df: pd.DataFrame) -> dict[str, int]:
    if "room_inventory" in df.columns:
        return (
            df.groupby("hotel")["room_inventory"]
            .max()
            .clip(lower=1)
            .astype(int)
            .to_dict()
        )

    monthly = (
        df[df["is_canceled"] == 0]
        .groupby(["hotel", "year_month"], as_index=False)["occupied_room_nights"]
        .sum()
    )
    inventory: dict[str, int] = {}
    for hotel, hotel_df in monthly.groupby("hotel"):
        per_day = hotel_df["occupied_room_nights"] / 30
        rooms = int(np.ceil(max(per_day.quantile(0.95), per_day.max() * 0.9, 1)))
        inventory[hotel] = max(rooms, 1)
    return inventory


def available_room_nights(df: pd.DataFrame, inventory: dict[str, int]) -> int:
    if df.empty:
        return 0
    month_hotel = df[["hotel", "year_month"]].drop_duplicates()
    total = 0
    for _, row in month_hotel.iterrows():
        year, month = [int(part) for part in row["year_month"].split("-")]
        days = calendar.monthrange(year, month)[1]
        total += inventory.get(row["hotel"], 1) * days
    return int(total)


def calculate_kpis(df: pd.DataFrame, inventory: dict[str, int]) -> dict[str, float]:
    bookings = len(df)
    occupied = float(df["occupied_room_nights"].sum())
    available = available_room_nights(df, inventory)
    uncanceled = df[df["is_canceled"] == 0]
    return {
        "occupancy_rate": (occupied / available * 100) if available else 0,
        "total_revenue": float(df["revenue"].sum()),
        "cancellation_rate": (df["is_canceled"].mean() * 100) if bookings else 0,
        "average_adr": float(uncanceled["adr"].mean()) if not uncanceled.empty else 0,
    }


def segment_analysis(db_path: str, date_filter: str = "") -> pd.DataFrame:
    sql = f"""
        SELECT
            market_segment AS Segment,
            COUNT(*) AS Bookings,
            AVG(is_canceled) * 100 AS "Cancellation Rate",
            AVG(CASE WHEN is_canceled = 0 THEN adr END) AS "Average ADR",
            SUM(revenue) AS "Revenue Contribution",
            SUM(occupied_room_nights) AS OccupiedRoomNights
        FROM bookings
        {date_filter}
        GROUP BY market_segment
        ORDER BY Bookings DESC
    """
    return query(db_path, sql)


def monthly_trends(db_path: str, inventory: dict[str, int], date_filter: str = "") -> pd.DataFrame:
    sql = f"""
        SELECT
            year_month,
            hotel,
            MAX(room_inventory) AS room_inventory,
            COUNT(*) AS bookings,
            SUM(occupied_room_nights) AS occupied_room_nights,
            SUM(revenue) AS revenue,
            AVG(is_canceled) * 100 AS cancellation_rate
        FROM bookings
        {date_filter}
        GROUP BY year_month, hotel
        ORDER BY year_month
    """
    monthly = query(db_path, sql)
    if monthly.empty:
        return monthly

    monthly["days"] = monthly["year_month"].apply(
        lambda value: calendar.monthrange(*[int(part) for part in value.split("-")])[1]
    )
    monthly["available_room_nights"] = monthly.apply(
        lambda row: max(row.get("room_inventory", 0), inventory.get(row["hotel"], 1))
        * row["days"],
        axis=1,
    )
    trend = (
        monthly.groupby("year_month", as_index=False)
        .agg(
            bookings=("bookings", "sum"),
            occupied_room_nights=("occupied_room_nights", "sum"),
            available_room_nights=("available_room_nights", "sum"),
            revenue=("revenue", "sum"),
            cancellation_rate=("cancellation_rate", "mean"),
        )
        .sort_values("year_month")
    )
    trend["occupancy_rate"] = (
        trend["occupied_room_nights"] / trend["available_room_nights"] * 100
    ).clip(0, 100)
    trend["arrival_month"] = pd.to_datetime(trend["year_month"] + "-01")
    return trend


def monthly_seasonality(trend: pd.DataFrame) -> pd.DataFrame:
    if trend.empty:
        return trend
    season = trend.copy()
    season["month"] = season["arrival_month"].dt.month_name()
    return (
        season.groupby("month", as_index=False)
        .agg(
            occupancy_rate=("occupancy_rate", "mean"),
            volatility=("occupancy_rate", "std"),
        )
        .fillna({"volatility": 0})
    )


def top_values(segment_df: pd.DataFrame, trend: pd.DataFrame) -> dict[str, str]:
    if segment_df.empty:
        return {}
    top_revenue = segment_df.loc[segment_df["Revenue Contribution"].idxmax()]
    high_cancel = segment_df.loc[segment_df["Cancellation Rate"].idxmax()]
    peak = trend.loc[trend["occupancy_rate"].idxmax()] if not trend.empty else None
    return {
        "top_revenue_segment": str(top_revenue["Segment"]),
        "highest_cancellation_segment": str(high_cancel["Segment"]),
        "highest_cancellation_rate": float(high_cancel["Cancellation Rate"]),
        "peak_occupancy_period": peak["arrival_month"].strftime("%B %Y") if peak is not None else "N/A",
    }
