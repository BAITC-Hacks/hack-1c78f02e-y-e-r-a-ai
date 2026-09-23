"""
Детекция разовых крупных заказов (ТЗ п.7.4).

Помечает транзакции, но не удаляет их из истории — только исключает
из базы regular demand через флаг is_outlier.
"""

from __future__ import annotations

import pandas as pd


def detect_outliers(
    sales: pd.DataFrame,
    *,
    iqr_multiplier: float = 3.0,
    min_abs_qty: float = 50.0,
    client_share_threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Находит выбросы в sales_transactions.

    Правила (детерминированные):
    1) По каждому product_id: |quantity| > Q3 + iqr_multiplier * IQR
       (считаем по абсолютным отгрузкам; отгрузки клиенту обычно < 0).
    2) Дополнительно: отгрузка одному client_hash составляет ≥ client_share_threshold
       от суммы |qty| по продукту за тот же календарный месяц И |qty| ≥ min_abs_qty
       и |qty| ≥ 5 × медиана отгрузки по продукту.

    Возвращает DataFrame с колонками transaction_id, outlier_reason
    (только помеченные строки). Не пишет в БД.
    """
    if sales.empty:
        return pd.DataFrame(columns=["transaction_id", "outlier_reason"])

    df = sales.copy()
    df["abs_qty"] = df["quantity"].abs()
    # Только расход (отгрузки); приходы/возвраты с +qty не считаем выбросами спроса
    ship = df[df["quantity"] < 0].copy()
    if ship.empty:
        return pd.DataFrame(columns=["transaction_id", "outlier_reason"])

    flagged: dict[int, str] = {}

    for product_id, grp in ship.groupby("product_id"):
        q1 = grp["abs_qty"].quantile(0.25)
        q3 = grp["abs_qty"].quantile(0.75)
        iqr = max(q3 - q1, 0.0)
        fence = q3 + iqr_multiplier * iqr
        median_qty = float(grp["abs_qty"].median()) if len(grp) else 0.0

        for row in grp.itertuples(index=False):
            tid = int(row.transaction_id)
            aq = float(row.abs_qty)
            reasons: list[str] = []

            if aq >= min_abs_qty and aq > fence and iqr > 0:
                reasons.append(
                    f"IQR: |qty|={aq:.0f} > Q3+{iqr_multiplier}*IQR={fence:.0f} "
                    f"(product_id={product_id})"
                )

            if reasons:
                flagged[tid] = "; ".join(reasons)

        # Правило «крупный разовый заказ одному клиенту в месяце»
        tmp = grp.copy()
        tmp["ym"] = pd.to_datetime(tmp["sale_date"]).dt.to_period("M").astype(str)
        for (_pid, ym), month_grp in tmp.groupby(["product_id", "ym"]):
            month_total = float(month_grp["abs_qty"].sum())
            if month_total <= 0:
                continue
            for row in month_grp.itertuples(index=False):
                aq = float(row.abs_qty)
                share = aq / month_total
                tid = int(row.transaction_id)
                if (
                    aq >= min_abs_qty
                    and median_qty > 0
                    and aq >= 5 * median_qty
                    and share >= client_share_threshold
                ):
                    reason = (
                        f"разовый клиент: |qty|={aq:.0f} = {share:.0%} месяца {ym}, "
                        f"≥5×median={median_qty:.0f} (product_id={product_id}, "
                        f"client_hash={str(row.client_hash)[:12]}…)"
                    )
                    flagged[tid] = (
                        f"{flagged[tid]}; {reason}" if tid in flagged else reason
                    )

    if not flagged:
        return pd.DataFrame(columns=["transaction_id", "outlier_reason"])

    return pd.DataFrame(
        [
            {"transaction_id": tid, "outlier_reason": reason}
            for tid, reason in flagged.items()
        ]
    )
