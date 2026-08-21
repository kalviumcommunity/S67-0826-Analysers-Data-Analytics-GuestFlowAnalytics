from pathlib import Path
import sqlite3

import pandas as pd


def initialize_database(df: pd.DataFrame, db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        db_df = df.copy()
        db_df["arrival_date"] = db_df["arrival_date"].dt.strftime("%Y-%m-%d")
        db_df.to_sql("bookings", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bookings_date ON bookings(arrival_date)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bookings_segment ON bookings(market_segment)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bookings_city ON bookings(city)")


def query(db_path: str | Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)
