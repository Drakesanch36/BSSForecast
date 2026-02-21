import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bss.config import Settings
from bss.data_loader import DataLoader
from bss.utils import apply_plot_style, save_figure, save_table


def run(cfg: Settings, dl: DataLoader) -> dict:
    apply_plot_style(cfg)
    results = {}

    li = dl.line_items()
    daily = dl.daily_sales()
    monthly = dl.monthly_sales()

    # --- Monthly sales trend ---
    results["monthly_sales"] = monthly
    save_table(monthly, "monthly_sales", cfg)

    fig, ax = plt.subplots(figsize=cfg.plot.figsize)
    ax.plot(monthly["Month"], monthly["Sales"], marker="o", linewidth=2)
    ax.fill_between(monthly["Month"], monthly["Sales"], alpha=0.15)
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, "monthly_sales_trend", cfg)

    # --- Weekly sales trend ---
    weekly = li.groupby("WeekOfYear", as_index=False)["Sales"].sum()
    results["weekly_sales"] = weekly
    save_table(weekly, "weekly_sales", cfg)

    fig, ax = plt.subplots(figsize=cfg.plot.figsize)
    ax.bar(weekly["WeekOfYear"], weekly["Sales"], color=sns.color_palette(cfg.plot.palette)[0])
    ax.set_title("Weekly Sales")
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Revenue ($)")
    fig.tight_layout()
    save_figure(fig, "weekly_sales", cfg)

    # --- Hourly order distribution ---
    hourly = li.groupby("Hour").agg(
        orders=("Name", "nunique"),
        units=("Lineitem quantity", "sum"),
    ).reset_index()
    results["hourly_distribution"] = hourly
    save_table(hourly, "hourly_distribution", cfg)

    fig, ax = plt.subplots(figsize=cfg.plot.figsize)
    ax.bar(hourly["Hour"], hourly["orders"], color=sns.color_palette(cfg.plot.palette)[1])
    ax.set_title("Orders by Hour of Day")
    ax.set_xlabel("Hour (24h)")
    ax.set_ylabel("Number of Orders")
    ax.set_xticks(range(0, 24))
    fig.tight_layout()
    save_figure(fig, "hourly_orders", cfg)

    # --- Day of week patterns ---
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = li.groupby("DayOfWeek").agg(
        sales=("Sales", "sum"),
        orders=("Name", "nunique"),
    ).reset_index()
    dow["day_name"] = dow["DayOfWeek"].map(dict(enumerate(dow_names)))
    results["day_of_week"] = dow
    save_table(dow, "day_of_week_sales", cfg)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(dow["day_name"], dow["sales"], color=sns.color_palette(cfg.plot.palette)[2])
    ax.set_title("Sales by Day of Week")
    ax.set_ylabel("Revenue ($)")
    fig.tight_layout()
    save_figure(fig, "day_of_week_sales", cfg)

    # --- Seasonal decomposition ---
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        ts = daily.set_index("date")["Sales"]
        if len(ts) >= 14:
            decomp = seasonal_decompose(ts, model="additive", period=7)
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            decomp.observed.plot(ax=axes[0], title="Observed")
            decomp.trend.plot(ax=axes[1], title="Trend")
            decomp.seasonal.plot(ax=axes[2], title="Seasonal (7-day)")
            decomp.resid.plot(ax=axes[3], title="Residual")
            fig.suptitle("Seasonal Decomposition of Daily Sales", fontsize=14, y=1.02)
            fig.tight_layout()
            save_figure(fig, "seasonal_decomposition", cfg)
            results["seasonal_decomposition"] = True
    except ImportError:
        results["seasonal_decomposition"] = False

    # --- Geographic distribution ---
    geo = li.dropna(subset=["Shipping Province"]) if "Shipping Province" in li.columns else pd.DataFrame()
    if not geo.empty:
        state_sales = geo.groupby("Shipping Province", as_index=False)["Sales"].sum()
        state_sales = state_sales.sort_values("Sales", ascending=False)
        results["geographic"] = state_sales
        save_table(state_sales, "geographic_sales", cfg)

        fig, ax = plt.subplots(figsize=(14, 6))
        top_states = state_sales.head(15)
        ax.barh(range(len(top_states)), top_states["Sales"].values[::-1],
                color=sns.color_palette(cfg.plot.palette, len(top_states)))
        ax.set_yticks(range(len(top_states)))
        ax.set_yticklabels(top_states["Shipping Province"].values[::-1])
        ax.set_xlabel("Revenue ($)")
        ax.set_title("Top 15 States by Revenue")
        fig.tight_layout()
        save_figure(fig, "geographic_sales", cfg)

    # --- Discount impact ---
    if "Discount Amount" in li.columns:
        disc = li.copy()
        disc["has_discount"] = disc["Discount Amount"].fillna(0) > 0
        disc_impact = disc.groupby("has_discount").agg(
            total_sales=("Sales", "sum"),
            avg_order=("Sales", "mean"),
            orders=("Name", "nunique"),
        ).reset_index()
        disc_impact["has_discount"] = disc_impact["has_discount"].map({True: "With Discount", False: "No Discount"})
        results["discount_impact"] = disc_impact
        save_table(disc_impact, "discount_impact", cfg)

    return results
