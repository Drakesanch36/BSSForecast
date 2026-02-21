import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bss.config import Settings
from bss.data_loader import DataLoader
from bss.utils import apply_plot_style, save_figure, save_table, fmt_currency


def run(cfg: Settings, dl: DataLoader) -> dict:
    apply_plot_style(cfg)
    results = {}
    n = cfg.analysis.top_n_products

    eo = dl.enriched_orders()

    # --- Top products by revenue ---
    prod_rev = eo.groupby("sku", as_index=False).agg(
        revenue=("Sales", "sum"),
        units=("Lineitem quantity", "sum"),
    ).sort_values("revenue", ascending=False)
    results["top_products"] = prod_rev.head(n)
    save_table(prod_rev, "top_products_by_revenue", cfg)

    fig, ax = plt.subplots(figsize=cfg.plot.figsize)
    top = prod_rev.head(n)
    ax.barh(range(n), top["revenue"].values[::-1])
    ax.set_yticks(range(n))
    ax.set_yticklabels(top["sku"].values[::-1], fontsize=8)
    ax.set_xlabel("Revenue ($)")
    ax.set_title(f"Top {n} Products by Revenue")
    fig.tight_layout()
    save_figure(fig, "top_products_revenue", cfg)

    # --- Top categories by profit ---
    cat_profit = eo.dropna(subset=["category_name", "profit_margin"]).copy()
    cat_summary = cat_profit.groupby("category_name", as_index=False).agg(
        revenue=("Sales", "sum"),
        profit=("Profit", "sum"),
        units=("Lineitem quantity", "sum"),
        profit_margin=("profit_margin", "mean"),
    ).sort_values("profit", ascending=False)
    results["category_profit"] = cat_summary
    save_table(cat_summary, "category_profitability", cfg)

    fig, ax = plt.subplots(figsize=cfg.plot.figsize)
    top_cat = cat_summary.head(n)
    colors = sns.color_palette(cfg.plot.palette, n)
    ax.barh(range(len(top_cat)), top_cat["profit"].values[::-1], color=colors)
    ax.set_yticks(range(len(top_cat)))
    ax.set_yticklabels(top_cat["category_name"].values[::-1], fontsize=8)
    ax.set_xlabel("Total Profit ($)")
    ax.set_title(f"Top {n} Categories by Profit")
    fig.tight_layout()
    save_figure(fig, "top_categories_profit", cfg)

    # --- Profitability matrix (BCG-style quadrants) ---
    median_rev = cat_summary["revenue"].median()
    median_margin = cat_summary["profit_margin"].median()

    def _quadrant(row):
        high_rev = row["revenue"] >= median_rev
        high_margin = row["profit_margin"] >= median_margin
        if high_rev and high_margin:
            return "Stars"
        if high_rev and not high_margin:
            return "Cash Cows"
        if not high_rev and high_margin:
            return "Question Marks"
        return "Dogs"

    cat_summary["quadrant"] = cat_summary.apply(_quadrant, axis=1)
    results["profitability_matrix"] = cat_summary
    save_table(cat_summary, "profitability_matrix", cfg)

    quad_colors = {"Stars": "#2ecc71", "Cash Cows": "#3498db", "Question Marks": "#f39c12", "Dogs": "#e74c3c"}
    fig, ax = plt.subplots(figsize=(10, 8))
    for quad, color in quad_colors.items():
        mask = cat_summary["quadrant"] == quad
        subset = cat_summary[mask]
        ax.scatter(subset["revenue"], subset["profit_margin"], label=quad,
                   color=color, s=subset["units"] / subset["units"].max() * 500 + 20, alpha=0.7)
    ax.axvline(median_rev, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(median_margin, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Revenue ($)")
    ax.set_ylabel("Profit Margin")
    ax.set_title("Category Profitability Matrix")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "profitability_matrix", cfg)

    # --- Category performance dashboard ---
    cat_summary["revenue_rank"] = cat_summary["revenue"].rank(ascending=False).astype(int)
    cat_summary["profit_rank"] = cat_summary["profit"].rank(ascending=False).astype(int)
    results["category_dashboard"] = cat_summary
    save_table(cat_summary, "category_performance_dashboard", cfg)

    return results
