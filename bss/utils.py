from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("Agg")

_style_applied = False


def apply_plot_style(cfg):
    global _style_applied
    if _style_applied:
        return
    try:
        plt.style.use(cfg.plot.style)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": cfg.plot.figsize,
        "figure.dpi": cfg.plot.dpi,
        "savefig.dpi": cfg.plot.dpi,
        "savefig.bbox": "tight",
    })
    _style_applied = True


def save_figure(fig, name: str, cfg, close: bool = True):
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.figures_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def save_table(df: pd.DataFrame, name: str, cfg, formats: tuple = ("csv", "xlsx")):
    cfg.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if "csv" in formats:
        p = cfg.paths.tables_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    if "xlsx" in formats:
        p = cfg.paths.tables_dir / f"{name}.xlsx"
        df.to_excel(p, index=False)
        paths.append(p)
    return paths


def fmt_currency(val):
    return f"${val:,.2f}"


def fmt_pct(val):
    return f"{val:.1f}%"
