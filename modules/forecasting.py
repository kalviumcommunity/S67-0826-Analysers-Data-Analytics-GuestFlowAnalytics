import numpy as np
import pandas as pd


def forecast_occupancy(monthly_trend: pd.DataFrame, periods: int = 4) -> pd.DataFrame:
    if monthly_trend.empty:
        return pd.DataFrame()

    history = monthly_trend[["arrival_month", "occupancy_rate"]].dropna().copy()
    history["type"] = "Historical"

    if len(history) < 2:
        return history

    x = np.arange(len(history))
    y = history["occupancy_rate"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    seasonal_adjustment = y[-min(6, len(y)) :].mean() - y.mean()

    future_dates = pd.date_range(
        history["arrival_month"].max() + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )
    future_x = np.arange(len(history), len(history) + periods)
    predicted = np.clip(intercept + slope * future_x + seasonal_adjustment, 0, 100)

    forecast = pd.DataFrame(
        {
            "arrival_month": future_dates,
            "occupancy_rate": predicted,
            "type": "Predicted",
        }
    )
    return pd.concat([history, forecast], ignore_index=True)


def forecast_direction(forecast_df: pd.DataFrame) -> str:
    predicted = forecast_df[forecast_df["type"] == "Predicted"]
    history = forecast_df[forecast_df["type"] == "Historical"]
    if predicted.empty or history.empty:
        return "Occupancy forecast is limited because there is not enough historical data."

    delta = predicted["occupancy_rate"].iloc[-1] - history["occupancy_rate"].iloc[-1]
    direction = "increase" if delta >= 0 else "decrease"
    return f"Occupancy is expected to {direction} by {abs(delta):.1f} percentage points over the upcoming period."
