#!/usr/bin/env python3
"""BSS Strategic Analysis — single entry point."""

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Blue Summit Supplies Sales Analysis")
    parser.add_argument("--no-forecast", action="store_true",
                        help="Skip Prophet/ARIMA forecasting (faster)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to settings.yaml (default: config/settings.yaml)")
    args = parser.parse_args()

    from bss.config import load_settings
    from bss.data_loader import DataLoader
    from bss.utils import apply_plot_style

    cfg = load_settings(args.config)
    dl = DataLoader(cfg)
    apply_plot_style(cfg)

    # Ensure output dirs exist
    for d in [cfg.paths.output_dir, cfg.paths.figures_dir, cfg.paths.tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_results = {}
    t0 = time.time()

    # Phase 2: Descriptive Analytics
    print("[1/6] Running customer analysis...")
    from bss.analysis import customers
    all_results["customers"] = customers.run(cfg, dl)

    print("[2/6] Running product analysis...")
    from bss.analysis import products
    all_results["products"] = products.run(cfg, dl)

    print("[3/6] Running sales trend analysis...")
    from bss.analysis import sales
    all_results["sales"] = sales.run(cfg, dl)

    print("[4/6] Running search/demand analysis...")
    from bss.analysis import search
    all_results["search"] = search.run(cfg, dl)

    # Phase 3: ML
    print("[5/6] Running ML analyses...")
    from bss.ml import segmentation, basket, churn
    all_results["segmentation"] = segmentation.run(cfg, dl)
    all_results["basket"] = basket.run(cfg, dl)
    all_results["churn"] = churn.run(cfg, dl)

    if not args.no_forecast:
        print("       Fitting Prophet + ARIMA (this may take a minute)...")
        from bss.ml import forecasting
        all_results["forecasting"] = forecasting.run(cfg, dl)
    else:
        print("       Skipping forecast (--no-forecast)")
        all_results["forecasting"] = {}

    # Phase 4: Report
    print("[6/6] Generating strategic report...")
    from bss.reporting import report_generator
    report_path = report_generator.generate(cfg, all_results)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Report: {report_path}")
    print(f"Figures: {cfg.paths.figures_dir}")
    print(f"Tables: {cfg.paths.tables_dir}")


if __name__ == "__main__":
    main()
