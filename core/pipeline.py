"""Оркестрация одного прогона расчёта: outliers → persist → recommendations → save."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pandas as pd

from core.demand import calculate_recommendations
from core.outliers import detect_outliers
from repository import (
    fetch_monthly_sales_history,
    fetch_monthly_stock,
    fetch_products,
    fetch_sales_for_calculation,
    fetch_seasonality,
    fetch_stockout_periods,
    fetch_transit,
    get_connection,
    persist_outliers,
    save_recommended_orders,
)


def run_replenishment_calculation(
    *,
    supplier_id: int | None = None,
    category: str | None = None,
    safety_days: int = 14,
) -> dict:
    """
    Полный цикл Y.E.R.A. AI (детерминированный Python, без LLM-расчёта qty).

    Возвращает dict: run_id, outliers_found, orders (DataFrame), saved.
    """
    conn = get_connection()
    try:
        products = fetch_products(supplier_id, category, conn)
        if products.empty:
            return {
                "run_id": None,
                "outliers_found": 0,
                "orders": pd.DataFrame(),
                "saved": 0,
            }

        sales = fetch_sales_for_calculation(conn)
        monthly_sales = fetch_monthly_sales_history(supplier_id, category, conn)
        monthly_stock = fetch_monthly_stock(products["product_id"].tolist(), conn)
        stockouts = fetch_stockout_periods(conn)
        transit = fetch_transit(conn)
        seasonality = fetch_seasonality(conn)

        outlier_flags = detect_outliers(sales)
        n_out = persist_outliers(conn, outlier_flags)
        sales = fetch_sales_for_calculation(conn)

        orders = calculate_recommendations(
            products,
            sales,
            monthly_sales,
            monthly_stock,
            stockouts,
            transit,
            seasonality,
            safety_days=safety_days,
        )

        run_id = f"yera-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
        saved = save_recommended_orders(conn, run_id, orders) if not orders.empty else 0

        return {
            "run_id": run_id,
            "outliers_found": n_out,
            "orders": orders,
            "saved": saved,
        }
    finally:
        conn.close()
