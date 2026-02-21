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
    n = cfg.analysis.top_n_searches

    search = dl.search_all_data()
    search = search.sort_values("Search count", ascending=False).reset_index(drop=True)

    # --- Top searched terms ---
    top_searches = search.head(n)
    results["top_searches"] = top_searches
    save_table(search, "all_searches_ranked", cfg)

    fig, ax = plt.subplots(figsize=cfg.plot.figsize)
    ax.barh(range(n), top_searches["Search count"].values[::-1],
            color=sns.color_palette(cfg.plot.palette, n))
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_searches["Search phrase"].values[::-1], fontsize=8)
    ax.set_xlabel("Search Count")
    ax.set_title(f"Top {n} Search Terms")
    fig.tight_layout()
    save_figure(fig, "top_searches", cfg)

    # --- Search-to-sales gap analysis (fuzzy matching) ---
    try:
        from rapidfuzz import fuzz, process

        products = dl.products_with_margins()
        product_names = products["product_name"].dropna().unique().tolist()
        category_names = products["category_name"].dropna().unique().tolist()
        match_targets = product_names + category_names

        threshold = cfg.analysis.fuzzy_match_threshold
        gap_rows = []
        for _, row in search.iterrows():
            term = str(row["Search phrase"])
            count = row["Search count"]
            match = process.extractOne(term, match_targets, scorer=fuzz.token_sort_ratio)
            if match:
                best_match, score, _ = match
                gap_rows.append({
                    "search_term": term,
                    "search_count": count,
                    "best_match": best_match,
                    "match_score": score,
                    "likely_found": score >= threshold,
                })

        gap_df = pd.DataFrame(gap_rows)
        unmet = gap_df[~gap_df["likely_found"]].sort_values("search_count", ascending=False)
        results["search_gap"] = gap_df
        results["unmet_demand"] = unmet
        save_table(gap_df, "search_to_sales_gap", cfg)
        save_table(unmet, "unmet_demand", cfg)

        if not unmet.empty:
            fig, ax = plt.subplots(figsize=cfg.plot.figsize)
            top_unmet = unmet.head(n)
            ax.barh(range(len(top_unmet)), top_unmet["search_count"].values[::-1],
                    color=sns.color_palette("Reds_r", len(top_unmet)))
            ax.set_yticks(range(len(top_unmet)))
            ax.set_yticklabels(top_unmet["search_term"].values[::-1], fontsize=8)
            ax.set_xlabel("Search Count")
            ax.set_title("Top Unmet Demand (Searches Without Good Product Match)")
            fig.tight_layout()
            save_figure(fig, "unmet_demand", cfg)

    except ImportError:
        results["search_gap"] = None
        results["unmet_demand"] = None

    return results
