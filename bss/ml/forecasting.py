import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bss.config import Settings
from bss.data_loader import DataLoader
from bss.utils import apply_plot_style, save_figure, save_table


def run(cfg: Settings, dl: DataLoader) -> dict:
    apply_plot_style(cfg)
    results = {}
    fc = cfg.ml.forecasting

    daily = dl.daily_sales().copy()
    daily = daily[["date", "Sales"]].rename(columns={"date": "ds", "Sales": "y"})
    daily = daily.sort_values("ds").reset_index(drop=True)

    # Temporal train/test split
    cutoff = daily["ds"].max() - pd.Timedelta(days=fc.test_days)
    train = daily[daily["ds"] <= cutoff].copy()
    test = daily[daily["ds"] > cutoff].copy()

    results["data_points"] = len(daily)
    results["train_size"] = len(train)
    results["test_size"] = len(test)

    # --- Prophet ---
    try:
        from prophet import Prophet

        # Log transform to handle skew
        train_prophet = train.copy()
        train_prophet["y"] = np.log1p(train_prophet["y"])

        model = Prophet(
            changepoint_prior_scale=fc.prophet_changepoint_prior,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        model.fit(train_prophet)

        future = model.make_future_dataframe(periods=fc.test_days + fc.forecast_days)
        forecast_raw = model.predict(future)

        # Inverse log transform
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast_raw[col] = np.expm1(forecast_raw[col])

        forecast = forecast_raw[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        results["prophet_forecast"] = forecast
        save_table(forecast, "prophet_forecast", cfg)

        # Evaluate on test set
        test_pred = forecast[forecast["ds"].isin(test["ds"])].copy()
        if not test_pred.empty:
            merged = test.merge(test_pred, on="ds")
            mae = np.mean(np.abs(merged["y"] - merged["yhat"]))
            rmse = np.sqrt(np.mean((merged["y"] - merged["yhat"]) ** 2))
            results["prophet_mae"] = mae
            results["prophet_rmse"] = rmse

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(daily["ds"], daily["y"], label="Actual", alpha=0.7, linewidth=1)
        ax.plot(forecast["ds"], forecast["yhat"], label="Prophet Forecast", color="red", linewidth=1.5)
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                        alpha=0.15, color="red", label="Confidence Interval")
        ax.axvline(cutoff, color="green", linestyle="--", alpha=0.7, label="Train/Test Split")
        ax.set_title("Sales Forecast (Prophet)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Sales ($)")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, "prophet_forecast", cfg)

        # Components plot
        fig_comp = model.plot_components(forecast_raw)
        save_figure(fig_comp, "prophet_components", cfg)

    except ImportError:
        results["prophet_error"] = "prophet not installed"

    # --- ARIMA ---
    try:
        import pmdarima as pm

        arima_train = train.set_index("ds")["y"]
        model_arima = pm.auto_arima(
            arima_train,
            seasonal=fc.arima_seasonal,
            m=7,
            suppress_warnings=True,
            stepwise=True,
            error_action="ignore",
        )
        results["arima_order"] = str(model_arima.order)
        results["arima_seasonal_order"] = str(model_arima.seasonal_order)

        n_forecast = len(test) + fc.forecast_days
        arima_pred, arima_ci = model_arima.predict(n_periods=n_forecast, return_conf_int=True)

        future_dates = pd.date_range(start=train["ds"].max() + pd.Timedelta(days=1), periods=n_forecast)
        arima_df = pd.DataFrame({
            "ds": future_dates,
            "yhat": arima_pred,
            "yhat_lower": arima_ci[:, 0],
            "yhat_upper": arima_ci[:, 1],
        })
        results["arima_forecast"] = arima_df
        save_table(arima_df, "arima_forecast", cfg)

        # Evaluate on test set
        test_arima = arima_df[arima_df["ds"].isin(test["ds"])]
        if not test_arima.empty:
            merged_a = test.merge(test_arima, on="ds")
            results["arima_mae"] = np.mean(np.abs(merged_a["y"] - merged_a["yhat"]))
            results["arima_rmse"] = np.sqrt(np.mean((merged_a["y"] - merged_a["yhat"]) ** 2))

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(daily["ds"], daily["y"], label="Actual", alpha=0.7, linewidth=1)
        ax.plot(arima_df["ds"], arima_df["yhat"], label="ARIMA Forecast", color="orange", linewidth=1.5)
        ax.fill_between(arima_df["ds"], arima_df["yhat_lower"], arima_df["yhat_upper"],
                        alpha=0.15, color="orange")
        ax.axvline(cutoff, color="green", linestyle="--", alpha=0.7, label="Train/Test Split")
        ax.set_title(f"Sales Forecast (ARIMA {model_arima.order})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Sales ($)")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, "arima_forecast", cfg)

    except ImportError:
        results["arima_error"] = "pmdarima not installed"

    # --- Model comparison summary ---
    comparison = {}
    if "prophet_mae" in results:
        comparison["Prophet"] = {"MAE": results["prophet_mae"], "RMSE": results["prophet_rmse"]}
    if "arima_mae" in results:
        comparison["ARIMA"] = {"MAE": results["arima_mae"], "RMSE": results["arima_rmse"]}
    if comparison:
        comp_df = pd.DataFrame(comparison).T.reset_index()
        comp_df.columns = ["Model", "MAE", "RMSE"]
        results["model_comparison"] = comp_df
        save_table(comp_df, "forecast_model_comparison", cfg)

    return results
