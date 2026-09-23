"""Чтение/запись storage.db. Только поля из схемы .cursorrules."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from database import DB_PATH, init_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    if not path.exists():
        init_db(path)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _df(conn: sqlite3.Connection, sql: str, params: tuple | list = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def fetch_suppliers(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    own = conn is None
    conn = conn or get_connection()
    try:
        return _df(
            conn,
            """
            SELECT supplier_id, supplier_name, lead_time_days
            FROM suppliers
            ORDER BY supplier_name
            """,
        )
    finally:
        if own:
            conn.close()


def fetch_categories(
    supplier_id: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[str]:
    own = conn is None
    conn = conn or get_connection()
    try:
        if supplier_id is None:
            df = _df(
                conn,
                """
                SELECT DISTINCT category FROM products
                WHERE category IS NOT NULL AND is_active = 1
                ORDER BY category
                """,
            )
        else:
            df = _df(
                conn,
                """
                SELECT DISTINCT category FROM products
                WHERE category IS NOT NULL AND is_active = 1 AND supplier_id = ?
                ORDER BY category
                """,
                (supplier_id,),
            )
        return df["category"].dropna().tolist()
    finally:
        if own:
            conn.close()


def fetch_products(
    supplier_id: int | None = None,
    category: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    own = conn is None
    conn = conn or get_connection()
    try:
        clauses = ["p.is_active = 1"]
        params: list = []
        if supplier_id is not None:
            clauses.append("p.supplier_id = ?")
            params.append(supplier_id)
        if category:
            clauses.append("p.category = ?")
            params.append(category)
        where = " AND ".join(clauses)
        return _df(
            conn,
            f"""
            SELECT
                p.product_id, p.code_1c, p.supplier_article, p.name, p.unit,
                p.category, p.supplier_id, p.min_ship_qty, p.multiplicity,
                s.supplier_name, s.lead_time_days
            FROM products p
            JOIN suppliers s ON s.supplier_id = p.supplier_id
            WHERE {where}
            ORDER BY s.supplier_name, p.name
            """,
            params,
        )
    finally:
        if own:
            conn.close()


def fetch_dashboard_metrics(
    supplier_id: int | None = None,
    category: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict:
    """
    Метрики дашборда:
    - число активных товаров
    - позиции с критической срочностью (последний run) или низким покрытием остатком
    - суммарный объём в пути
    """
    own = conn is None
    conn = conn or get_connection()
    try:
        products = fetch_products(supplier_id, category, conn)
        product_ids = products["product_id"].tolist()
        n_products = len(product_ids)

        if not product_ids:
            return {
                "products_count": 0,
                "critical_count": 0,
                "transit_qty": 0.0,
                "draft_orders": 0,
            }

        placeholders = ",".join("?" * len(product_ids))

        transit = _df(
            conn,
            f"""
            SELECT COALESCE(SUM(quantity), 0) AS total
            FROM transit_orders
            WHERE product_id IN ({placeholders})
            """,
            product_ids,
        )
        transit_qty = float(transit.iloc[0]["total"])

        critical = _df(
            conn,
            f"""
            SELECT COUNT(*) AS cnt
            FROM recommended_orders ro
            WHERE ro.product_id IN ({placeholders})
              AND ro.urgency = 'критическая'
              AND ro.run_id = (
                  SELECT run_id FROM recommended_orders
                  ORDER BY created_at DESC LIMIT 1
              )
            """,
            product_ids,
        )
        critical_count = int(critical.iloc[0]["cnt"])

        # Если расчёта ещё не было — оцениваем риск по остатку < 20% средней месячной продажи
        if critical_count == 0:
            low_stock = _df(
                conn,
                f"""
                WITH latest_stock AS (
                    SELECT ms.product_id, ms.stock_qty,
                           ROW_NUMBER() OVER (
                               PARTITION BY ms.product_id
                               ORDER BY ms.year DESC, ms.month DESC
                           ) AS rn
                    FROM monthly_stock ms
                    WHERE ms.product_id IN ({placeholders})
                ),
                avg_sales AS (
                    SELECT product_id, AVG(qty_sold) AS avg_qty
                    FROM monthly_sales
                    WHERE product_id IN ({placeholders})
                    GROUP BY product_id
                )
                SELECT COUNT(*) AS cnt
                FROM latest_stock ls
                JOIN avg_sales a ON a.product_id = ls.product_id
                WHERE ls.rn = 1
                  AND COALESCE(ls.stock_qty, 0) < 0.2 * COALESCE(a.avg_qty, 0)
                """,
                product_ids + product_ids,
            )
            critical_count = int(low_stock.iloc[0]["cnt"])

        drafts = _df(
            conn,
            f"""
            SELECT COUNT(*) AS cnt
            FROM recommended_orders
            WHERE product_id IN ({placeholders}) AND status = 'draft'
            """,
            product_ids,
        )

        return {
            "products_count": n_products,
            "critical_count": critical_count,
            "transit_qty": transit_qty,
            "draft_orders": int(drafts.iloc[0]["cnt"]),
        }
    finally:
        if own:
            conn.close()


def fetch_current_stock(
    supplier_id: int | None = None,
    category: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    own = conn is None
    conn = conn or get_connection()
    try:
        products = fetch_products(supplier_id, category, conn)
        if products.empty:
            return pd.DataFrame()
        ids = products["product_id"].tolist()
        placeholders = ",".join("?" * len(ids))
        stock = _df(
            conn,
            f"""
            SELECT product_id, warehouse, year, month, stock_qty
            FROM monthly_stock
            WHERE product_id IN ({placeholders})
            """,
            ids,
        )
        if stock.empty:
            return products.assign(current_stock=0.0, warehouse=None)

        stock = stock.sort_values(["product_id", "year", "month"])
        latest = stock.groupby("product_id", as_index=False).tail(1)
        out = products.merge(
            latest[["product_id", "warehouse", "stock_qty", "year", "month"]],
            on="product_id",
            how="left",
        )
        out = out.rename(columns={"stock_qty": "current_stock"})
        return out
    finally:
        if own:
            conn.close()


def fetch_monthly_sales_history(
    supplier_id: int | None = None,
    category: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    own = conn is None
    conn = conn or get_connection()
    try:
        products = fetch_products(supplier_id, category, conn)
        if products.empty:
            return pd.DataFrame()
        ids = products["product_id"].tolist()
        placeholders = ",".join("?" * len(ids))
        sales = _df(
            conn,
            f"""
            SELECT product_id, year, month, qty_sold
            FROM monthly_sales
            WHERE product_id IN ({placeholders})
            ORDER BY product_id, year, month
            """,
            ids,
        )
        return sales.merge(
            products[["product_id", "code_1c", "supplier_article", "name", "unit", "category"]],
            on="product_id",
            how="left",
        )
    finally:
        if own:
            conn.close()


def fetch_sales_for_calculation(conn: sqlite3.Connection) -> pd.DataFrame:
    return _df(
        conn,
        """
        SELECT
            transaction_id, sale_date, doc_number, product_id, unit, warehouse,
            quantity, client_hash, is_outlier, outlier_reason
        FROM sales_transactions
        ORDER BY product_id, sale_date
        """,
    )


def fetch_monthly_stock(
    product_ids: list[int] | None = None,
    conn: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    own = conn is None
    conn = conn or get_connection()
    try:
        if product_ids:
            placeholders = ",".join("?" * len(product_ids))
            return _df(
                conn,
                f"""
                SELECT product_id, warehouse, year, month, stock_qty
                FROM monthly_stock
                WHERE product_id IN ({placeholders})
                ORDER BY product_id, year, month
                """,
                product_ids,
            )
        return _df(
            conn,
            """
            SELECT product_id, warehouse, year, month, stock_qty
            FROM monthly_stock
            ORDER BY product_id, year, month
            """,
        )
    finally:
        if own:
            conn.close()


def fetch_stockout_periods(conn: sqlite3.Connection) -> pd.DataFrame:
    return _df(
        conn,
        """
        SELECT product_id, warehouse, start_date, end_date, days_out, detection_method
        FROM stockout_periods
        """,
    )


def fetch_transit(conn: sqlite3.Connection) -> pd.DataFrame:
    return _df(
        conn,
        """
        SELECT product_id, po_number, po_date, expected_arrival, quantity
        FROM transit_orders
        """,
    )


def fetch_seasonality(conn: sqlite3.Connection) -> pd.DataFrame:
    return _df(
        conn,
        """
        SELECT supplier_id, category, year, month, total_value
        FROM seasonality_reference
        ORDER BY supplier_id, category, year, month
        """,
    )


def persist_outliers(conn: sqlite3.Connection, outliers: pd.DataFrame) -> int:
    """
    Записывает флаги is_outlier / outlier_reason.
    Сначала сбрасывает все флаги, затем помечает найденные выбросы.
    """
    conn.execute(
        "UPDATE sales_transactions SET is_outlier = 0, outlier_reason = NULL"
    )
    if outliers.empty:
        conn.commit()
        return 0

    rows = outliers[["transaction_id", "outlier_reason"]].itertuples(index=False)
    conn.executemany(
        """
        UPDATE sales_transactions
        SET is_outlier = 1, outlier_reason = ?
        WHERE transaction_id = ?
        """,
        [(reason, tid) for tid, reason in rows],
    )
    conn.commit()
    return len(outliers)


def save_recommended_orders(conn: sqlite3.Connection, run_id: str, orders: pd.DataFrame) -> int:
    if orders.empty:
        return 0
    payload = [
        (
            run_id,
            int(r.product_id),
            int(r.supplier_id),
            float(r.base_demand) if pd.notna(r.base_demand) else None,
            float(r.seasonality_adj) if pd.notna(r.seasonality_adj) else None,
            float(r.stockout_adj) if pd.notna(r.stockout_adj) else None,
            float(r.outlier_excluded_qty) if pd.notna(r.outlier_excluded_qty) else 0.0,
            float(r.current_stock) if pd.notna(r.current_stock) else None,
            float(r.in_transit_qty) if pd.notna(r.in_transit_qty) else None,
            float(r.recommended_qty),
            r.urgency,
            r.justification,
            "draft",
        )
        for r in orders.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO recommended_orders (
            run_id, product_id, supplier_id, base_demand, seasonality_adj,
            stockout_adj, outlier_excluded_qty, current_stock, in_transit_qty,
            recommended_qty, urgency, justification, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def fetch_recommended_orders(
    run_id: str | None = None,
    supplier_id: int | None = None,
    category: str | None = None,
    status: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    own = conn is None
    conn = conn or get_connection()
    try:
        clauses = ["1=1"]
        params: list = []
        if run_id:
            clauses.append("ro.run_id = ?")
            params.append(run_id)
        else:
            # последний прогон
            clauses.append(
                """
                ro.run_id = (
                    SELECT run_id FROM recommended_orders
                    ORDER BY created_at DESC LIMIT 1
                )
                """
            )
        if supplier_id is not None:
            clauses.append("ro.supplier_id = ?")
            params.append(supplier_id)
        if category:
            clauses.append("p.category = ?")
            params.append(category)
        if status:
            clauses.append("ro.status = ?")
            params.append(status)

        where = " AND ".join(clauses)
        return _df(
            conn,
            f"""
            SELECT
                ro.id, ro.run_id, ro.product_id, ro.supplier_id,
                p.code_1c, p.supplier_article, p.name, p.unit, p.category,
                s.supplier_name,
                ro.base_demand, ro.seasonality_adj, ro.stockout_adj,
                ro.outlier_excluded_qty, ro.current_stock, ro.in_transit_qty,
                ro.recommended_qty, ro.urgency, ro.justification,
                ro.status, ro.created_at, ro.approved_by, ro.approved_at
            FROM recommended_orders ro
            JOIN products p ON p.product_id = ro.product_id
            JOIN suppliers s ON s.supplier_id = ro.supplier_id
            WHERE {where}
            ORDER BY
                CASE ro.urgency
                    WHEN 'критическая' THEN 1
                    WHEN 'высокая' THEN 2
                    WHEN 'средняя' THEN 3
                    ELSE 4
                END,
                ro.recommended_qty DESC
            """,
            params,
        )
    finally:
        if own:
            conn.close()


def approve_orders(
    order_ids: list[int],
    approved_by: str,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Меняет status draft → approved. Без человека заказ не «отправляется»."""
    if not order_ids:
        return 0
    own = conn is None
    conn = conn or get_connection()
    try:
        placeholders = ",".join("?" * len(order_ids))
        cur = conn.execute(
            f"""
            UPDATE recommended_orders
            SET status = 'approved',
                approved_by = ?,
                approved_at = datetime('now')
            WHERE id IN ({placeholders})
              AND status = 'draft'
            """,
            [approved_by, *order_ids],
        )
        conn.commit()
        return cur.rowcount
    finally:
        if own:
            conn.close()
