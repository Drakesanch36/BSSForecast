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

    li = dl.line_items()
    n = cfg.analysis.top_n_customers
    num_months = dl.date_range_months()

    # --- Top customers by revenue ---
    cust_sales = li.groupby("Company", as_index=False)["Sales"].sum()
    cust_sales = cust_sales.sort_values("Sales", ascending=False).reset_index(drop=True)
    total_sales = cust_sales["Sales"].sum()
    cust_sales["pct_total"] = cust_sales["Sales"] / total_sales * 100
    cust_sales["cumulative_pct"] = cust_sales["pct_total"].cumsum()
    results["top_customers"] = cust_sales.head(n)
    save_table(cust_sales, "top_customers_by_revenue", cfg)

    fig, ax = plt.subplots(figsize=cfg.plot.figsize)
    top = cust_sales.head(n)
    ax.barh(top["Company"][::-1], top["Sales"][::-1], color=sns.color_palette(cfg.plot.palette, n))
    ax.set_xlabel("Total Revenue ($)")
    ax.set_title(f"Top {n} Customers by Revenue")
    for i, (v, pct) in enumerate(zip(top["Sales"][::-1], top["pct_total"][::-1])):
        ax.text(v + total_sales * 0.005, i, f"{fmt_currency(v)} ({pct:.1f}%)", va="center", fontsize=8)
    save_figure(fig, "top_customers_revenue", cfg)

    # --- Customer type breakdown ---
    type_sales = li.groupby("Customer Type", as_index=False)["Sales"].sum()
    type_sales = type_sales.sort_values("Sales", ascending=False)
    type_sales["pct_total"] = type_sales["Sales"] / total_sales * 100
    results["customer_types"] = type_sales
    save_table(type_sales, "customer_type_breakdown", cfg)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(type_sales["Sales"], labels=type_sales["Customer Type"],
           autopct="%1.1f%%", colors=sns.color_palette(cfg.plot.palette, len(type_sales)))
    ax.set_title("Revenue by Customer Type")
    save_figure(fig, "customer_type_pie", cfg)

    # --- Top SKUs per customer ---
    sku_by_cust = li.groupby(["Company", "sku"], as_index=False)["Sales"].sum()
    sku_by_cust = sku_by_cust.sort_values(["Company", "Sales"], ascending=[True, False])
    top_skus = sku_by_cust.groupby("Company").head(5)
    results["top_skus_per_customer"] = top_skus
    save_table(top_skus, "top_skus_per_customer", cfg)

    # --- Transaction & unit rates (uses actual months, not hardcoded /12) ---
    monthly_txn = li.groupby(["Company", "sku", "Month"]).size().reset_index(name="count")
    txn_summary = monthly_txn.groupby(["Company", "sku"]).agg(
        transactions=("count", "sum"),
    ).reset_index()
    txn_summary["monthly_transaction_rate"] = txn_summary["transactions"] / num_months
    txn_summary = txn_summary.sort_values(["Company", "monthly_transaction_rate"], ascending=[True, False])
    results["transaction_rates"] = txn_summary
    save_table(txn_summary, "transaction_rates", cfg)

    unit_summary = li.groupby(["Company", "sku"], as_index=False)["Lineitem quantity"].sum()
    unit_summary.rename(columns={"Lineitem quantity": "total_units"}, inplace=True)
    unit_summary["monthly_unit_rate"] = unit_summary["total_units"] / num_months
    unit_summary = unit_summary.sort_values(["Company", "monthly_unit_rate"], ascending=[True, False])
    results["unit_rates"] = unit_summary
    save_table(unit_summary, "unit_rates", cfg)

    # --- RFM analysis ---
    rfm = dl.customer_rfm()
    for col in ["recency", "frequency", "monetary"]:
        ascending = col == "recency"
        rfm[f"{col}_score"] = pd.qcut(rfm[col].rank(method="first", ascending=ascending),
                                       q=cfg.analysis.rfm_segments, labels=False) + 1
    rfm["rfm_score"] = rfm["recency_score"] + rfm["frequency_score"] + rfm["monetary_score"]

    def _label(row):
        r, f, m = row["recency_score"], row["frequency_score"], row["monetary_score"]
        max_s = cfg.analysis.rfm_segments
        if r >= max_s and f >= max_s:
            return "Champions"
        if r >= max_s - 1 and f >= max_s - 1:
            return "Loyal"
        if r >= max_s - 1:
            return "Recent"
        if f >= max_s - 1:
            return "Frequent (At Risk)"
        if r <= 2 and f <= 2:
            return "Lost"
        return "Needs Attention"

    rfm["segment"] = rfm.apply(_label, axis=1)

    # Simple CLV estimate: avg_order_value * frequency * projected factor
    rfm["estimated_clv"] = rfm["avg_order_value"] * rfm["frequency"] * 1.5
    results["rfm"] = rfm
    save_table(rfm, "customer_rfm_segments", cfg)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, title in zip(axes, ["recency", "frequency", "monetary"],
                               ["Recency (days)", "Frequency (orders)", "Monetary ($)"]):
        ax.hist(rfm[col], bins=20, color=sns.color_palette(cfg.plot.palette)[0], edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(col.capitalize())
    fig.suptitle("RFM Distributions", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "rfm_distributions", cfg)

    seg_counts = rfm["segment"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    seg_counts.plot(kind="bar", ax=ax, color=sns.color_palette(cfg.plot.palette, len(seg_counts)))
    ax.set_title("Customer Segments")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    save_figure(fig, "customer_segments", cfg)

    return results
