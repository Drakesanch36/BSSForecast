import base64
from pathlib import Path
from datetime import datetime
import pandas as pd
from jinja2 import Template
from bss.config import Settings
from bss.data_loader import DataLoader
from bss.utils import fmt_currency, fmt_pct

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BSS Strategic Report</title>
<style>
:root { --primary: #1a5276; --accent: #2980b9; --bg: #f8f9fa; --card: #ffffff; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: #333; line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
header { background: var(--primary); color: white; padding: 40px 0; text-align: center; margin-bottom: 30px; }
header h1 { font-size: 2.2em; margin-bottom: 5px; }
header .subtitle { opacity: 0.85; font-size: 1.1em; }
.section { background: var(--card); border-radius: 8px; padding: 30px;
           margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.section h2 { color: var(--primary); border-bottom: 2px solid var(--accent);
              padding-bottom: 10px; margin-bottom: 20px; font-size: 1.5em; }
.section h3 { color: var(--accent); margin: 15px 0 10px; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin: 20px 0; }
.kpi { background: var(--bg); padding: 20px; border-radius: 6px; text-align: center;
       border-left: 4px solid var(--accent); }
.kpi .value { font-size: 1.8em; font-weight: bold; color: var(--primary); }
.kpi .label { font-size: 0.85em; color: #666; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }
th { background: var(--primary); color: white; padding: 10px 12px; text-align: left; }
td { padding: 8px 12px; border-bottom: 1px solid #eee; }
tr:hover { background: #f0f7ff; }
.figure { text-align: center; margin: 20px 0; }
.figure img { max-width: 100%; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
.figure .caption { font-size: 0.85em; color: #666; margin-top: 8px; }
.rec-box { background: #eaf2f8; border-left: 4px solid var(--accent); padding: 15px;
           margin: 10px 0; border-radius: 0 4px 4px 0; }
.rec-box strong { color: var(--primary); }
.alert { background: #fdedec; border-left: 4px solid #e74c3c; padding: 15px;
         margin: 10px 0; border-radius: 0 4px 4px 0; }
footer { text-align: center; padding: 20px; color: #999; font-size: 0.85em; }
</style>
</head>
<body>
<header>
<h1>Blue Summit Supplies</h1>
<div class="subtitle">Strategic Sales Analysis Report &mdash; {{ generated_date }}</div>
</header>
<div class="container">

<!-- Executive Summary -->
<div class="section">
<h2>Executive Summary</h2>
<div class="kpi-grid">
<div class="kpi"><div class="value">{{ total_revenue }}</div><div class="label">Total Revenue</div></div>
<div class="kpi"><div class="value">{{ total_orders }}</div><div class="label">Total Orders</div></div>
<div class="kpi"><div class="value">{{ total_customers }}</div><div class="label">Unique Customers</div></div>
<div class="kpi"><div class="value">{{ avg_order_value }}</div><div class="label">Avg Order Value</div></div>
</div>
<h3>Top Recommendations</h3>
{% for rec in executive_recs %}
<div class="rec-box"><strong>{{ loop.index }}.</strong> {{ rec }}</div>
{% endfor %}
</div>

<!-- Customer Analysis -->
<div class="section">
<h2>Customer Analysis</h2>
{{ embed_figure("top_customers_revenue") }}
{{ embed_figure("customer_type_pie") }}
<h3>RFM Segmentation</h3>
{{ embed_figure("rfm_distributions") }}
{{ embed_figure("customer_segments") }}
{% if customer_rfm_table %}
<h3>Customer Segments Overview</h3>
{{ customer_rfm_table }}
{% endif %}
</div>

<!-- Product Analysis -->
<div class="section">
<h2>Product Analysis</h2>
{{ embed_figure("top_products_revenue") }}
{{ embed_figure("top_categories_profit") }}
<h3>Profitability Matrix</h3>
{{ embed_figure("profitability_matrix") }}
</div>

<!-- Sales Trends -->
<div class="section">
<h2>Sales Trends</h2>
{{ embed_figure("monthly_sales_trend") }}
{{ embed_figure("weekly_sales") }}
{{ embed_figure("hourly_orders") }}
{{ embed_figure("day_of_week_sales") }}
{% if has_seasonal %}
<h3>Seasonal Decomposition</h3>
{{ embed_figure("seasonal_decomposition") }}
{% endif %}
{% if has_geographic %}
<h3>Geographic Distribution</h3>
{{ embed_figure("geographic_sales") }}
{% endif %}
</div>

<!-- Search / Demand Intelligence -->
<div class="section">
<h2>Demand Intelligence</h2>
{{ embed_figure("top_searches") }}
{% if has_unmet_demand %}
<h3>Unmet Demand</h3>
<p>These search terms have no close product match, representing potential inventory gaps:</p>
{{ embed_figure("unmet_demand") }}
{% endif %}
</div>

<!-- Customer Segmentation (ML) -->
<div class="section">
<h2>Customer Segmentation (K-Means)</h2>
{{ embed_figure("cluster_selection") }}
{{ embed_figure("customer_clusters_scatter") }}
{{ embed_figure("segment_distribution") }}
{% if segment_recs_table %}
<h3>Segment Targeting Recommendations</h3>
{{ segment_recs_table }}
{% endif %}
</div>

<!-- Market Basket -->
<div class="section">
<h2>Market Basket Analysis</h2>
{% if has_basket_rules %}
{{ embed_figure("association_rules_lift") }}
{% if has_heatmap %}
{{ embed_figure("co_occurrence_heatmap") }}
{% endif %}
{% if basket_rules_table %}
<h3>Top Association Rules</h3>
{{ basket_rules_table }}
{% endif %}
{% else %}
<p>{{ basket_error }}</p>
{% endif %}
</div>

<!-- Forecast -->
{% if has_forecast %}
<div class="section">
<h2>Sales Forecast</h2>
{{ embed_figure("prophet_forecast") }}
{{ embed_figure("prophet_components") }}
{% if has_arima %}
{{ embed_figure("arima_forecast") }}
{% endif %}
{% if forecast_comparison_table %}
<h3>Model Comparison</h3>
{{ forecast_comparison_table }}
{% endif %}
</div>
{% endif %}

<!-- Churn Risk -->
<div class="section">
<h2>Churn Risk Analysis</h2>
<div class="kpi-grid">
<div class="kpi"><div class="value">{{ at_risk_count }}</div><div class="label">At-Risk Customers</div></div>
<div class="kpi"><div class="value">{{ at_risk_pct }}</div><div class="label">% of Customer Base</div></div>
<div class="kpi"><div class="value">{{ revenue_at_risk }}</div><div class="label">Revenue at Risk</div></div>
</div>
{{ embed_figure("churn_risk") }}
{{ embed_figure("churn_scatter") }}
</div>

<!-- Strategic Recommendations -->
<div class="section">
<h2>Strategic Recommendations</h2>
{% for rec in strategic_recs %}
<div class="rec-box"><strong>{{ rec.title }}</strong><br>{{ rec.detail }}</div>
{% endfor %}
</div>

</div>
<footer>
Generated on {{ generated_date }} | BSS Strategic Analysis v1.0
</footer>
</body>
</html>"""


def _embed_figure(name: str, figures_dir: Path) -> str:
    path = figures_dir / f"{name}.png"
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<div class="figure"><img src="data:image/png;base64,{data}" alt="{name}"><div class="caption">{name.replace("_", " ").title()}</div></div>'


def _df_to_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return ""
    return df.head(max_rows).to_html(index=False, classes="", border=0, float_format="%.2f")


def generate(cfg: Settings, all_results: dict) -> Path:
    figures_dir = cfg.paths.figures_dir

    def embed_figure(name):
        return _embed_figure(name, figures_dir)

    # KPI calculations
    cust_res = all_results.get("customers", {})
    sales_res = all_results.get("sales", {})
    search_res = all_results.get("search", {})
    seg_res = all_results.get("segmentation", {})
    basket_res = all_results.get("basket", {})
    forecast_res = all_results.get("forecasting", {})
    churn_res = all_results.get("churn", {})

    monthly = sales_res.get("monthly_sales", pd.DataFrame())
    total_revenue = monthly["Sales"].sum() if not monthly.empty else 0
    total_orders = monthly["Orders"].sum() if "Orders" in monthly.columns else 0
    rfm = cust_res.get("rfm", pd.DataFrame())
    total_customers = len(rfm) if not rfm.empty else 0
    aov = total_revenue / total_orders if total_orders > 0 else 0

    # Executive recommendations
    exec_recs = []
    if churn_res.get("at_risk_count", 0) > 0:
        exec_recs.append(f"Launch re-engagement campaign for {churn_res['at_risk_count']} at-risk customers representing {fmt_currency(churn_res.get('revenue_at_risk', 0))} in historical revenue.")
    unmet = search_res.get("unmet_demand")
    if unmet is not None and not unmet.empty:
        top_terms = ", ".join(unmet["search_term"].head(3).tolist())
        exec_recs.append(f"Address unmet demand: top unmatched search terms include {top_terms}.")
    if not monthly.empty and len(monthly) >= 2:
        recent_trend = monthly["Sales"].iloc[-1] / monthly["Sales"].iloc[-2] - 1
        direction = "up" if recent_trend > 0 else "down"
        exec_recs.append(f"Sales trending {direction} {abs(recent_trend)*100:.1f}% month-over-month; {'maintain momentum' if recent_trend > 0 else 'investigate decline and adjust strategy'}.")
    rules = basket_res.get("rules", pd.DataFrame())
    if not rules.empty:
        top_rule = rules.iloc[0]
        exec_recs.append(f"Cross-sell opportunity: customers who buy {top_rule.get('antecedents_str', '?')} also buy {top_rule.get('consequents_str', '?')} (lift: {top_rule.get('lift', 0):.1f}x).")
    if not exec_recs:
        exec_recs.append("Comprehensive analysis complete. See sections below for detailed findings.")

    # Strategic recommendations
    strategic_recs = [
        {"title": "Focus on Top Customer Retention",
         "detail": "The top 10 customers drive a disproportionate share of revenue. Implement a key account program with dedicated support and volume discounts."},
        {"title": "Expand High-Margin Categories",
         "detail": "Categories in the 'Stars' quadrant (high revenue + high margin) should receive increased marketing investment and inventory priority."},
        {"title": "Optimize Order Timing",
         "detail": "Peak order hours and days should inform staffing, marketing campaign scheduling, and promotional timing."},
        {"title": "Close the Search-to-Sales Gap",
         "detail": "High-volume search terms with no matching products represent inventory expansion opportunities."},
        {"title": "Implement Automated Churn Prevention",
         "detail": "Set up automated triggers when customer recency exceeds their typical ordering pattern to send personalized re-engagement offers."},
    ]

    # Segment recommendations table
    seg_recs = seg_res.get("recommendations")
    seg_recs_table = _df_to_html(seg_recs) if seg_recs is not None else ""

    # Basket rules table
    basket_rules_table = ""
    if not rules.empty:
        basket_rules_table = _df_to_html(
            rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(10)
        )

    # Forecast comparison
    forecast_comp = forecast_res.get("model_comparison")
    forecast_comparison_table = _df_to_html(forecast_comp) if forecast_comp is not None else ""

    # Customer RFM summary table
    customer_rfm_table = ""
    if not rfm.empty and "segment" in rfm.columns:
        seg_summary = rfm.groupby("segment").agg(
            customers=("Company", "count"),
            avg_monetary=("monetary", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_recency=("recency", "mean"),
        ).reset_index()
        customer_rfm_table = _df_to_html(seg_summary)

    # Render template
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        generated_date=datetime.now().strftime("%B %d, %Y"),
        total_revenue=fmt_currency(total_revenue),
        total_orders=f"{total_orders:,}",
        total_customers=f"{total_customers:,}",
        avg_order_value=fmt_currency(aov),
        executive_recs=exec_recs,
        strategic_recs=strategic_recs,
        customer_rfm_table=customer_rfm_table,
        embed_figure=embed_figure,
        has_seasonal=sales_res.get("seasonal_decomposition", False),
        has_geographic="geographic" in sales_res,
        has_unmet_demand=unmet is not None and not unmet.empty,
        has_basket_rules=not rules.empty,
        has_heatmap=(figures_dir / "co_occurrence_heatmap.png").exists(),
        basket_rules_table=basket_rules_table,
        basket_error=basket_res.get("error", "No association rules generated."),
        segment_recs_table=seg_recs_table,
        has_forecast="prophet_forecast" in forecast_res or "arima_forecast" in forecast_res,
        has_arima="arima_forecast" in forecast_res,
        forecast_comparison_table=forecast_comparison_table,
        at_risk_count=churn_res.get("at_risk_count", 0),
        at_risk_pct=fmt_pct(churn_res.get("at_risk_pct", 0)),
        revenue_at_risk=fmt_currency(churn_res.get("revenue_at_risk", 0)),
    )

    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.report_path
    report_path.write_text(html)
    return report_path
