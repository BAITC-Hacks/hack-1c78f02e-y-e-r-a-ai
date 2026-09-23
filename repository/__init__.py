"""Слой доступа к SQLite. UI и core не ходят в БД напрямую."""

from repository.data import (
    approve_orders,
    fetch_categories,
    fetch_current_stock,
    fetch_dashboard_metrics,
    fetch_monthly_sales_history,
    fetch_monthly_stock,
    fetch_products,
    fetch_recommended_orders,
    fetch_sales_for_calculation,
    fetch_seasonality,
    fetch_stockout_periods,
    fetch_suppliers,
    fetch_transit,
    get_connection,
    persist_outliers,
    save_recommended_orders,
)

__all__ = [
    "approve_orders",
    "fetch_categories",
    "fetch_current_stock",
    "fetch_dashboard_metrics",
    "fetch_monthly_sales_history",
    "fetch_monthly_stock",
    "fetch_products",
    "fetch_recommended_orders",
    "fetch_sales_for_calculation",
    "fetch_seasonality",
    "fetch_stockout_periods",
    "fetch_suppliers",
    "fetch_transit",
    "get_connection",
    "persist_outliers",
    "save_recommended_orders",
]
