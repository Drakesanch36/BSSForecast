import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from bss.config import Settings
from bss.data_loader import DataLoader
from bss.utils import apply_plot_style, save_figure, save_table


def run(cfg: Settings, dl: DataLoader) -> dict:
    apply_plot_style(cfg)
    results = {}
    sc = cfg.ml.segmentation

    rfm = dl.customer_rfm().copy()
    features = rfm[["recency", "frequency", "monetary"]].copy()

    # Log-transform monetary to reduce skew
    features["monetary"] = np.log1p(features["monetary"])
    features["frequency"] = np.log1p(features["frequency"])

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # --- Elbow method + silhouette ---
    k_range = range(2, sc.max_clusters + 1)
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=sc.random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    best_k = list(k_range)[np.argmax(silhouettes)]
    results["silhouette_scores"] = dict(zip(k_range, silhouettes))
    results["best_k"] = best_k

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(list(k_range), inertias, "o-")
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax2.plot(list(k_range), silhouettes, "o-")
    ax2.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.legend()
    fig.suptitle("Optimal Cluster Selection", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "cluster_selection", cfg)

    # --- Final clustering ---
    km_final = KMeans(n_clusters=best_k, random_state=sc.random_state, n_init=10)
    rfm["cluster"] = km_final.fit_predict(X)

    # Profile each cluster
    profiles = rfm.groupby("cluster").agg(
        count=("Company", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_order_value=("avg_order_value", "mean"),
    ).reset_index()

    # Assign business labels based on RFM characteristics
    label_map = {}
    sorted_by_monetary = profiles.sort_values("avg_monetary", ascending=False)
    labels = ["Champions", "Loyal Customers", "Promising", "At Risk", "Needs Attention", "Hibernating", "Lost", "New"]
    for i, (_, row) in enumerate(sorted_by_monetary.iterrows()):
        label_map[row["cluster"]] = labels[min(i, len(labels) - 1)]

    # Refine: low recency + low frequency → Lost; high recency + high frequency → Champions
    for _, row in profiles.iterrows():
        c = row["cluster"]
        if row["avg_recency"] > profiles["avg_recency"].quantile(0.75) and row["avg_frequency"] < profiles["avg_frequency"].quantile(0.25):
            label_map[c] = "Lost"
        elif row["avg_recency"] < profiles["avg_recency"].quantile(0.25) and row["avg_monetary"] > profiles["avg_monetary"].quantile(0.75):
            label_map[c] = "Champions"

    rfm["segment_label"] = rfm["cluster"].map(label_map)
    profiles["segment_label"] = profiles["cluster"].map(label_map)

    results["cluster_profiles"] = profiles
    results["customer_clusters"] = rfm
    save_table(profiles, "cluster_profiles", cfg)
    save_table(rfm[["Company", "cluster", "segment_label", "recency", "frequency", "monetary"]], "customer_clusters", cfg)

    # --- Scatter plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for cluster_id in sorted(rfm["cluster"].unique()):
        mask = rfm["cluster"] == cluster_id
        label = label_map.get(cluster_id, f"Cluster {cluster_id}")
        ax.scatter(rfm.loc[mask, "frequency"], rfm.loc[mask, "monetary"],
                   label=label, alpha=0.6, s=50)
    ax.set_xlabel("Frequency (orders)")
    ax.set_ylabel("Monetary ($)")
    ax.set_title("Customer Segments (K-Means on RFM)")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "customer_clusters_scatter", cfg)

    # --- Segment size bar chart ---
    seg_sizes = rfm["segment_label"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    seg_sizes.plot(kind="bar", ax=ax, color=sns.color_palette(cfg.plot.palette, len(seg_sizes)))
    ax.set_title("Customer Segment Distribution")
    ax.set_ylabel("Number of Customers")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    save_figure(fig, "segment_distribution", cfg)

    # --- Targeting recommendations ---
    recs = []
    for _, row in profiles.iterrows():
        label = row["segment_label"]
        if label == "Champions":
            recs.append({"segment": label, "recommendation": "Reward loyalty; offer exclusive deals and early access to new products."})
        elif label == "Loyal Customers":
            recs.append({"segment": label, "recommendation": "Upsell higher-value categories; request reviews and referrals."})
        elif label == "At Risk":
            recs.append({"segment": label, "recommendation": "Re-engagement campaign with special offers; ask for feedback."})
        elif label == "Lost":
            recs.append({"segment": label, "recommendation": "Win-back campaign with deep discounts; survey to understand churn reasons."})
        else:
            recs.append({"segment": label, "recommendation": "Nurture with targeted content and progressive offers."})
    results["recommendations"] = pd.DataFrame(recs)
    save_table(results["recommendations"], "segment_recommendations", cfg)

    return results
