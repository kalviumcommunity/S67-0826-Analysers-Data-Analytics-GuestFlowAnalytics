import pandas as pd


def build_recommendations(
    segment_df: pd.DataFrame, trend: pd.DataFrame, forecast_df: pd.DataFrame
) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    if segment_df.empty:
        return recommendations

    high_cancel = segment_df.loc[segment_df["Cancellation Rate"].idxmax()]
    stable = segment_df.loc[segment_df["Cancellation Rate"].idxmin()]

    if high_cancel["Cancellation Rate"] >= 25:
        recommendations.append(
            {
                "title": "High Cancellation Risk",
                "priority": "High",
                "explanation": (
                    f"{high_cancel['Segment']} bookings cancel at "
                    f"{high_cancel['Cancellation Rate']:.1f}%. Consider tighter policies, "
                    "deposit rules, or direct-booking incentives for this segment."
                ),
            }
        )

    predicted = forecast_df[forecast_df["type"] == "Predicted"]
    if not predicted.empty:
        avg_predicted = predicted["occupancy_rate"].mean()
        if avg_predicted < 55:
            recommendations.append(
                {
                    "title": "Low Occupancy Outlook",
                    "priority": "High",
                    "explanation": (
                        f"Forecast occupancy averages {avg_predicted:.1f}%. Use targeted offers "
                        "and promotional pricing in upcoming low-demand months."
                    ),
                }
            )
        elif avg_predicted > 75:
            recommendations.append(
                {
                    "title": "High Demand Pricing",
                    "priority": "Medium",
                    "explanation": (
                        f"Forecast occupancy averages {avg_predicted:.1f}%. Review ADR and "
                        "raise rates during high-demand periods."
                    ),
                }
            )

    if not trend.empty:
        low_month = trend.loc[trend["occupancy_rate"].idxmin()]
        recommendations.append(
            {
                "title": "Seasonal Demand Gap",
                "priority": "Medium",
                "explanation": (
                    f"{low_month['arrival_month'].strftime('%B %Y')} has the weakest occupancy "
                    f"at {low_month['occupancy_rate']:.1f}%. Build offers around this period."
                ),
            }
        )

    recommendations.append(
        {
            "title": "Reliable Segment Growth",
            "priority": "Low",
            "explanation": (
                f"{stable['Segment']} has the lowest cancellation rate at "
                f"{stable['Cancellation Rate']:.1f}%. Increase campaigns toward this dependable segment."
            ),
        }
    )

    return recommendations[:4]
