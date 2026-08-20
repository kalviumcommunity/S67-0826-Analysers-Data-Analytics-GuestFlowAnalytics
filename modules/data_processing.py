from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "segment",
    "hotel_name",
    "city",
    "total_rooms",
    "check_in_date",
    "lead_time_days",
    "length_of_stay",
    "num_guests",
    "num_rooms",
    "adr",
    "total_price",
    "booking_status",
    "is_cancelled",
]

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def load_and_clean_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place guestflow_cleaned_data.csv in the data folder."
        )

    df = pd.read_csv(csv_path, na_values=["NULL", "NA", ""])
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing required columns: {', '.join(missing)}")

    df = df.drop_duplicates().copy()

    numeric_defaults = {
        "is_cancelled": 0,
        "is_no_show": 0,
        "is_completed": 0,
        "lead_time_days": 0,
        "length_of_stay": 1,
        "num_guests": 1,
        "num_rooms": 1,
        "total_rooms": 1,
        "adr": 0,
        "total_price": 0,
        "revenue_per_night": 0,
    }
    for column, default in numeric_defaults.items():
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)

    text_defaults = {
        "hotel_name": "Unknown Hotel",
        "city": "Unknown City",
        "segment": "Unknown",
        "booking_status": "Unknown",
        "season_tag": "Unknown",
    }
    for column, default in text_defaults.items():
        if column in df.columns:
            df[column] = df[column].fillna(default).astype(str).str.strip()

    df["arrival_date"] = pd.to_datetime(df["check_in_date"], errors="coerce")
    df = df.dropna(subset=["arrival_date"]).copy()

    df["hotel"] = df["hotel_name"]
    df["market_segment"] = df["segment"].str.replace("_", " ", regex=False)
    df["customer_type"] = df["market_segment"]
    df["country"] = "India"
    df["lead_time"] = df["lead_time_days"].clip(lower=0)
    df["total_nights"] = df["length_of_stay"].clip(lower=1)
    df["guest_count"] = df["num_guests"].clip(lower=1)
    df["rooms_booked"] = df["num_rooms"].clip(lower=1)
    df["room_inventory"] = df["total_rooms"].clip(lower=1)
    df["is_canceled"] = df["is_cancelled"].clip(lower=0, upper=1)
    if "is_completed" not in df.columns:
        df["is_completed"] = (df["booking_status"].str.lower() == "completed").astype(int)
    df["is_realized"] = np.where((df["is_canceled"] == 0) & (df["is_completed"] == 1), 1, 0)
    df["adr"] = df["adr"].clip(lower=0)
    fallback_revenue = df["adr"] * df["rooms_booked"] * df["total_nights"]
    df["revenue"] = np.where(
        df["is_realized"] == 1,
        df["total_price"].where(df["total_price"] > 0, fallback_revenue),
        0,
    )
    df["occupied_room_nights"] = np.where(
        df["is_realized"] == 1,
        df["rooms_booked"] * df["total_nights"],
        0,
    )
    df["year_month"] = df["arrival_date"].dt.to_period("M").astype(str)
    df["month_name"] = df["arrival_date"].dt.month_name()

    columns = [
        "hotel",
        "arrival_date",
        "year_month",
        "month_name",
        "city",
        "market_segment",
        "customer_type",
        "is_canceled",
        "is_realized",
        "lead_time",
        "adr",
        "revenue",
        "occupied_room_nights",
        "total_nights",
        "rooms_booked",
        "room_inventory",
        "guest_count",
        "country",
    ]
    return df[columns].sort_values("arrival_date").reset_index(drop=True)
