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
    multiplier = cfg.ml.churn.recency_multiplier

    rfm = dl.customer_rfm().copy()

    # A customer is at risk if their recency exceeds their typical inter-order gap
    # by the configured multiplier
    rfm["expected_gap"] = rfm["avg_days_between_orders"] * multiplier
    rfm["at_risk"] = rfm["recency"] > rfm["expected_gap"]
    rfm["days_overdue"] = (rfm["recency"] - rfm["expected_gap"]).clip(lower=0)
    rfm["revenue_at_risk"] = rfm["monetary"]  # total historical spend

    at_risk = rfm[rfm["at_risk"]].sort_values("revenue_at_risk", ascending=False).reset_index(drop=True)

    results["total_customers"] = len(rfm)
    results["at_risk_count"] = len(at_risk)
    results["at_risk_pct"] = len(at_risk) / len(rfm) * 100 if len(rfm) > 0 else 0
    results["revenue_at_risk"] = at_risk["revenue_at_risk"].sum()
    results["at_risk_customers"] = at_risk
    results["all_customers"] = rfm

    save_table(at_risk[["Company", "recency", "frequency", "monetary", "avg_days_between_orders",
                         "expected_gap", "days_overdue", "revenue_at_risk"]],
               "at_risk_customers", cfg)

    # --- At-risk vs active ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    status_counts = rfm["at_risk"].value_counts()
    labels = ["Active", "At Risk"]
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].pie(status_counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Customer Status Distribution")

    if not at_risk.empty:
        top_risk = at_risk.head(10)
        axes[1].barh(range(len(top_risk)), top_risk["revenue_at_risk"].values[::-1], color="#e74c3c")
        axes[1].set_yticks(range(len(top_risk)))
        axes[1].set_yticklabels(top_risk["Company"].values[::-1], fontsize=8)
        axes[1].set_xlabel("Historical Revenue ($)")
        axes[1].set_title("Top 10 At-Risk Customers by Revenue")
    fig.suptitle("Churn Risk Analysis", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "churn_risk", cfg)

    # --- Recency vs expected gap scatter ---
    fig, ax = plt.subplots(figsize=(10, 8))
    active = rfm[~rfm["at_risk"]]
    ax.scatter(active["expected_gap"], active["recency"], alpha=0.4, label="Active", color="#2ecc71", s=20)
    if not at_risk.empty:
        ax.scatter(at_risk["expected_gap"], at_risk["recency"], alpha=0.6, label="At Risk", color="#e74c3c", s=30)
    max_val = max(rfm["recency"].max(), rfm["expected_gap"].max())
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Threshold line")
    ax.set_xlabel("Expected Re-order Gap (days)")
    ax.set_ylabel("Actual Recency (days)")
    ax.set_title("Churn Risk: Recency vs Expected Gap")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "churn_scatter", cfg)

    return results
