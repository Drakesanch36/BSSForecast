import pandas as pd
import numpy as np
from functools import lru_cache
from bss.config import Settings


class DataLoader:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self._cache = {}

    def _path(self, filename: str):
        return self.cfg.paths.data_dir / filename

    def _get(self, key: str, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    # --- Raw loaders ---

    def orders_raw(self) -> pd.DataFrame:
        def _load():
            df = pd.read_csv(self._path(self.cfg.data_files.orders), low_memory=False)
            df["Paid at"] = pd.to_datetime(df["Paid at"], format="mixed", dayfirst=False)
            return df
        return self._get("orders_raw", _load)

    def orders_hourly_raw(self) -> pd.DataFrame:
        def _load():
            df = pd.read_csv(self._path(self.cfg.data_files.orders_hourly), low_memory=False)
            df["Paid at"] = pd.to_datetime(df["Paid at"], format="mixed", dayfirst=False)
            return df
        return self._get("orders_hourly_raw", _load)

    def products_raw(self) -> pd.DataFrame:
        def _load():
            return pd.read_csv(self._path(self.cfg.data_files.products), encoding="unicode_escape")
        return self._get("products_raw", _load)

    def categories_raw(self) -> pd.DataFrame:
        def _load():
            return pd.read_csv(self._path(self.cfg.data_files.categories), encoding="unicode_escape")
        return self._get("categories_raw", _load)

    def search_data(self) -> pd.DataFrame:
        return self._get("search", lambda: pd.read_csv(self._path(self.cfg.data_files.search)))

    def search_all_data(self) -> pd.DataFrame:
        return self._get("search_all", lambda: pd.read_csv(self._path(self.cfg.data_files.search_all)))

    # --- Enriched / computed ---

    def line_items(self) -> pd.DataFrame:
        """Orders with Sales column and time features computed."""
        def _load():
            df = self.orders_hourly_raw().copy()
            df = df.dropna(subset=["Lineitem quantity", "Lineitem price", "Paid at"])
            df["Sales"] = df["Lineitem quantity"] * df["Lineitem price"]
            df["Year"] = df["Paid at"].dt.year
            df["Month"] = df["Paid at"].dt.month
            df["Day"] = df["Paid at"].dt.day
            df["DayOfWeek"] = df["Paid at"].dt.dayofweek
            df["WeekOfYear"] = df["Paid at"].dt.isocalendar().week.astype(int)
            df["Hour"] = df["Paid at"].dt.hour
            df.rename(columns={"Lineitem sku": "sku"}, inplace=True)
            return df
        return self._get("line_items", _load)

    def products_with_margins(self) -> pd.DataFrame:
        """Products enriched with profit margin from category table."""
        def _load():
            prods = self.products_raw()[["sku", "product_name", "category_name"]].dropna()
            cats = self.categories_raw()
            merged = prods.merge(cats, left_on="category_name", right_on="Category", how="left")
            merged.drop(columns=["Category"], inplace=True, errors="ignore")
            merged.rename(columns={"Profit Margin": "profit_margin"}, inplace=True)
            return merged
        return self._get("products_with_margins", _load)

    def enriched_orders(self) -> pd.DataFrame:
        """Line items merged with product info and profit margins."""
        def _load():
            li = self.line_items()
            pm = self.products_with_margins()
            merged = li.merge(pm[["sku", "category_name", "profit_margin"]], on="sku", how="left")
            merged["Profit"] = merged["Sales"] * merged["profit_margin"].fillna(0)
            return merged
        return self._get("enriched_orders", _load)

    def daily_sales(self) -> pd.DataFrame:
        """Daily aggregated sales."""
        def _load():
            li = self.line_items()
            daily = li.groupby(li["Paid at"].dt.date).agg(
                Sales=("Sales", "sum"),
                Orders=("Name", "nunique"),
                Units=("Lineitem quantity", "sum"),
            ).reset_index()
            daily.rename(columns={"Paid at": "date"}, inplace=True)
            daily["date"] = pd.to_datetime(daily["date"])
            return daily.sort_values("date").reset_index(drop=True)
        return self._get("daily_sales", _load)

    def monthly_sales(self) -> pd.DataFrame:
        """Monthly aggregated sales."""
        def _load():
            daily = self.daily_sales()
            monthly = daily.set_index("date").resample("MS").agg({
                "Sales": "sum", "Orders": "sum", "Units": "sum"
            }).reset_index()
            monthly.rename(columns={"date": "Month"}, inplace=True)
            return monthly
        return self._get("monthly_sales", _load)

    def customer_rfm(self) -> pd.DataFrame:
        """Customer-level RFM features."""
        def _load():
            li = self.line_items()
            max_date = li["Paid at"].max()

            # Aggregate at order level first (Name = order ID)
            orders = li.groupby(["Name", "Company"]).agg(
                order_date=("Paid at", "max"),
                order_total=("Sales", "sum"),
            ).reset_index()

            # Customer-level aggregation
            cust = orders.groupby("Company").agg(
                recency=("order_date", lambda x: (max_date - x.max()).days),
                frequency=("Name", "nunique"),
                monetary=("order_total", "sum"),
                first_order=("order_date", "min"),
                last_order=("order_date", "max"),
            ).reset_index()

            cust["avg_order_value"] = cust["monetary"] / cust["frequency"]
            active_days = (cust["last_order"] - cust["first_order"]).dt.days.clip(lower=1)
            cust["avg_days_between_orders"] = active_days / cust["frequency"].clip(lower=1)
            return cust
        return self._get("customer_rfm", _load)

    def date_range_months(self) -> int:
        """Actual number of months in the dataset (replaces hardcoded /12)."""
        li = self.line_items()
        date_range = li["Paid at"].max() - li["Paid at"].min()
        return max(1, round(date_range.days / 30.44))
