from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class PathsConfig:
    data_dir: Path
    output_dir: Path
    figures_dir: Path
    tables_dir: Path
    report_path: Path


@dataclass
class DataFilesConfig:
    orders: str
    orders_hourly: str
    products: str
    categories: str
    search: str
    search_all: str


@dataclass
class AnalysisConfig:
    top_n_customers: int = 10
    top_n_products: int = 10
    top_n_searches: int = 15
    rfm_segments: int = 4
    fuzzy_match_threshold: int = 70


@dataclass
class SegmentationConfig:
    max_clusters: int = 8
    random_state: int = 42


@dataclass
class BasketConfig:
    min_support: float = 0.02
    min_lift: float = 1.0
    use_categories: bool = True


@dataclass
class ForecastingConfig:
    prophet_changepoint_prior: float = 0.05
    forecast_days: int = 90
    test_days: int = 30
    arima_seasonal: bool = True


@dataclass
class ChurnConfig:
    recency_multiplier: float = 2.0


@dataclass
class MLConfig:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    basket: BasketConfig = field(default_factory=BasketConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    churn: ChurnConfig = field(default_factory=ChurnConfig)


@dataclass
class PlotConfig:
    style: str = "seaborn-v0_8-whitegrid"
    figsize: tuple = (12, 6)
    dpi: int = 150
    palette: str = "Set2"


@dataclass
class Settings:
    paths: PathsConfig
    data_files: DataFilesConfig
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


def load_settings(config_path=None) -> Settings:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    raw = yaml.safe_load(config_path.read_text())

    project_root = config_path.resolve().parent.parent
    p = raw["paths"]
    paths = PathsConfig(
        data_dir=project_root / p["data_dir"],
        output_dir=project_root / p["output_dir"],
        figures_dir=project_root / p["figures_dir"],
        tables_dir=project_root / p["tables_dir"],
        report_path=project_root / p["report_path"],
    )

    df = raw["data_files"]
    data_files = DataFilesConfig(**df)

    analysis = AnalysisConfig(**raw.get("analysis", {}))

    ml_raw = raw.get("ml", {})
    ml = MLConfig(
        segmentation=SegmentationConfig(**ml_raw.get("segmentation", {})),
        basket=BasketConfig(**ml_raw.get("basket", {})),
        forecasting=ForecastingConfig(**ml_raw.get("forecasting", {})),
        churn=ChurnConfig(**ml_raw.get("churn", {})),
    )

    plot_raw = raw.get("plot", {})
    figsize = tuple(plot_raw.pop("figsize", [12, 6]))
    plot = PlotConfig(figsize=figsize, **plot_raw)

    return Settings(paths=paths, data_files=data_files, analysis=analysis, ml=ml, plot=plot)
