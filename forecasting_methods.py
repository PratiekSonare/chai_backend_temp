"""
Simple inventory forecasting methods.
Alternatives to Prophet for stable, low-volume, discrete integer inventory data.
All methods enforce a floor of 0 (inventory cannot go negative).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def naive_forecast(series: pd.Series, periods: int) -> pd.DataFrame:
    """
    Last Value Carried Forward.
    The most realistic forecast for stable inventory — next week = this week.
    """
    last_val = float(series.iloc[-1])
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=periods, freq="W")

    predicted = [max(0, round(last_val))] * periods
    return pd.DataFrame({
        "ds": future_dates,
        "predicted": predicted,
        "lower": predicted,
        "upper": predicted,
    })


def moving_average_forecast(series: pd.Series, periods: int, window: int = 3) -> pd.DataFrame:
    """
    Simple Moving Average forecast.
    Uses the mean of the last `window` observations as the forecast level.
    Confidence bands widen slightly over time.
    """
    if len(series) < window:
        window = len(series)
    avg = float(series.tail(window).mean())
    std = float(series.tail(window).std()) if len(series) > 1 else 0
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=periods, freq="W")

    predicted = []
    lower = []
    upper = []
    for i in range(periods):
        val = max(0, round(avg))
        predicted.append(val)
        lower.append(max(0, round(val - 1.96 * std * np.sqrt(1 + i / len(series)))))
        upper.append(round(val + 1.96 * std * np.sqrt(1 + i / len(series))))

    return pd.DataFrame({"ds": future_dates, "predicted": predicted, "lower": lower, "upper": upper})


def exponential_smoothing_forecast(series: pd.Series, periods: int, alpha: float = 0.3) -> pd.DataFrame:
    """
    Simple Exponential Smoothing (level only, no trend).
    Smooths out noise and projects a flat level forward.
    Best for stable inventory with minor fluctuations.
    """
    values = series.values.astype(float)
    # Initialize with first value
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])

    level = smoothed[-1]
    # Estimate residual variance for confidence bands
    residuals = values - np.array(smoothed[:-1] + [level])
    residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0

    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=periods, freq="W")

    predicted = []
    lower = []
    upper = []
    for i in range(periods):
        val = max(0, round(level))
        predicted.append(val)
        lower.append(max(0, round(val - 1.96 * residual_std * np.sqrt(i + 1))))
        upper.append(round(val + 1.96 * residual_std * np.sqrt(i + 1)))

    return pd.DataFrame({"ds": future_dates, "predicted": predicted, "lower": lower, "upper": upper})


def holt_linear_forecast(series: pd.Series, periods: int, alpha: float = 0.3, beta: float = 0.1) -> pd.DataFrame:
    """
    Holt's Linear Trend (Double Exponential Smoothing).
    Captures level + linear trend. Good when inventory has a consistent upward/downward drift.
    """
    values = series.values.astype(float)
    if len(values) < 2:
        return naive_forecast(series, periods)

    level = values[0]
    trend = values[1] - values[0]

    for v in values[1:]:
        new_level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level

    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=periods, freq="W")

    predicted = []
    lower = []
    upper = []
    for i in range(1, periods + 1):
        val = max(0, round(level + trend * i))
        predicted.append(val)
        # Wider bands for trend model since uncertainty grows
        spread = abs(trend) * i * 0.5 + 2
        lower.append(max(0, round(val - spread)))
        upper.append(round(val + spread))

    return pd.DataFrame({"ds": future_dates, "predicted": predicted, "lower": lower, "upper": upper})


def croston_forecast(series: pd.Series, periods: int, alpha: float = 0.1) -> pd.DataFrame:
    """
    Croston's Method — industry standard for intermittent/low-demand inventory.
    Separately smooths the inter-demand intervals and demand sizes,
    then combines them for a stable forecast.
    """
    values = series.values.astype(float)
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=periods, freq="W")

    # Identify demand events (non-zero stock changes or current levels)
    # For inventory, treat the stock level itself as the "demand signal"
    demand_indices = np.where(values > 0)[0]
    if len(demand_indices) == 0:
        # All zeros — forecast zeros
        return pd.DataFrame({
            "ds": future_dates,
            "predicted": [0] * periods,
            "lower": [0] * periods,
            "upper": [0] * periods,
        })

    # Separate demand sizes and inter-demand intervals
    demand_sizes = []
    intervals = []
    prev_idx = 0
    for idx in demand_indices:
        demand_sizes.append(values[idx])
        if idx > prev_idx:
            intervals.append(idx - prev_idx)
        prev_idx = idx

    if not demand_sizes:
        return naive_forecast(series, periods)

    # Smoothed demand size
    q_smooth = demand_sizes[0]
    for q in demand_sizes[1:]:
        q_smooth = alpha * q + (1 - alpha) * q_smooth

    # Smoothed interval (if we have intervals)
    if intervals:
        p_smooth = intervals[0]
        for p in intervals[1:]:
            p_smooth = alpha * p + (1 - alpha) * p_smooth
    else:
        p_smooth = 1.0

    # Croston forecast = smoothed demand / smoothed interval
    forecast_val = q_smooth / p_smooth if p_smooth > 0 else q_smooth

    std = float(np.std(demand_sizes)) if len(demand_sizes) > 1 else 0

    predicted = []
    lower = []
    upper = []
    for i in range(periods):
        val = max(0, round(forecast_val))
        predicted.append(val)
        lower.append(max(0, round(val - 1.96 * std)))
        upper.append(round(val + 1.96 * std))

    return pd.DataFrame({"ds": future_dates, "predicted": predicted, "lower": lower, "upper": upper})


def run_forecast_comparison(
    prophet_df: pd.DataFrame,
    methods: List[str],
    forecast_months: int,
) -> Dict[str, Dict]:
    """
    Run multiple forecasting methods on the same time series.

    Args:
        prophet_df: DataFrame with 'ds' and 'y' columns (from prepare_prophet_dataframe)
        methods: list of method names to run
        forecast_months: how many months to forecast

    Returns:
        Dict mapping method name -> {forecast: [...], method_info: {...}}
    """
    series = prophet_df.set_index("ds")["y"]
    periods = forecast_months * 4  # weekly

    method_map = {
        "naive": ("Naive (Last Value)", "Carries forward the last observed stock level. Best for stable inventory.", naive_forecast),
        "moving_avg": ("Moving Average", "Averages the last 3 weeks. Smooths short-term noise.", lambda s, p: moving_average_forecast(s, p, window=3)),
        "exp_smoothing": ("Exp. Smoothing", "Weighted average giving more importance to recent observations.", lambda s, p: exponential_smoothing_forecast(s, p, alpha=0.3)),
        "holt": ("Holt Linear", "Captures level + linear trend. Good for consistent drift.", lambda s, p: holt_linear_forecast(s, p, alpha=0.3, beta=0.1)),
        "croston": ("Croston's Method", "Industry standard for intermittent/low-demand inventory.", lambda s, p: croston_forecast(s, p, alpha=0.1)),
    }

    results = {}
    for method_key in methods:
        if method_key not in method_map:
            continue
        name, description, func = method_map[method_key]
        try:
            fc = func(series, periods)
            results[method_key] = {
                "method": name,
                "description": description,
                "forecast": [
                    {
                        "date": row["ds"].strftime("%Y-%m-%d") if hasattr(row["ds"], "strftime") else str(row["ds"]),
                        "predicted": int(row["predicted"]),
                        "lower": int(row["lower"]),
                        "upper": int(row["upper"]),
                    }
                    for _, row in fc.iterrows()
                ],
            }
        except Exception as e:
            results[method_key] = {
                "method": name,
                "description": description,
                "error": str(e),
                "forecast": [],
            }

    return results
