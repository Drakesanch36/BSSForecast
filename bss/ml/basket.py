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
    bc = cfg.ml.basket

    eo = dl.enriched_orders()

    # Use categories (fewer items → less sparsity) or SKUs
    item_col = "category_name" if bc.use_categories else "sku"
    basket_data = eo.dropna(subset=[item_col, "Name"])

    # Build basket matrix: rows = orders, cols = items, values = 1/0
    basket = basket_data.groupby(["Name", item_col]).size().unstack(fill_value=0)
    basket = (basket > 0).astype(int)

    # Filter: need at least 2 items per order for association rules to matter
    multi_item = basket[basket.sum(axis=1) >= 2]
    results["basket_stats"] = {
        "total_orders": len(basket),
        "multi_item_orders": len(multi_item),
        "unique_items": len(basket.columns),
    }

    if len(multi_item) < 10:
        results["rules"] = pd.DataFrame()
        results["error"] = "Too few multi-item orders for association rules"
        return results

    try:
        from mlxtend.frequent_patterns import apriori, association_rules

        frequent = apriori(multi_item, min_support=bc.min_support, use_colnames=True)
        if frequent.empty:
            results["rules"] = pd.DataFrame()
            results["error"] = "No frequent itemsets found at configured min_support"
            return results

        rules = association_rules(frequent, metric="lift", min_threshold=bc.min_lift)
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

        # Convert frozensets to strings for readability
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

        results["rules"] = rules
        save_table(
            rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]],
            "association_rules", cfg
        )

        # --- Top rules visualization ---
        top_rules = rules.head(15)
        if not top_rules.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            labels = [f"{r['antecedents_str']} → {r['consequents_str']}" for _, r in top_rules.iterrows()]
            y_pos = range(len(labels))
            bars = ax.barh(y_pos, top_rules["lift"].values[::-1],
                          color=sns.color_palette(cfg.plot.palette, len(labels)))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels[::-1], fontsize=7)
            ax.set_xlabel("Lift")
            ax.set_title("Top Association Rules by Lift")
            ax.axvline(x=1, color="red", linestyle="--", alpha=0.5, label="Lift = 1 (random)")
            ax.legend()
            fig.tight_layout()
            save_figure(fig, "association_rules_lift", cfg)

        # --- Heatmap of support ---
        if len(rules) >= 5:
            pivot_items = list(set(
                [item for s in rules["antecedents"].head(20) for item in s] +
                [item for s in rules["consequents"].head(20) for item in s]
            ))[:15]
            co_occurrence = multi_item[pivot_items].T.dot(multi_item[pivot_items]) if all(
                i in multi_item.columns for i in pivot_items
            ) else pd.DataFrame()
            if not co_occurrence.empty:
                np.fill_diagonal(co_occurrence.values, 0)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(co_occurrence, cmap="YlOrRd", ax=ax, annot=True, fmt="d")
                ax.set_title("Category Co-occurrence Matrix")
                fig.tight_layout()
                save_figure(fig, "co_occurrence_heatmap", cfg)

    except ImportError:
        results["rules"] = pd.DataFrame()
        results["error"] = "mlxtend not installed"

    return results
