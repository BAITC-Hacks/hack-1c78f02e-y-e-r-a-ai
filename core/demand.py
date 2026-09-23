"""
Детерминированный расчёт рекомендованных заказов.

Вход/выход — DataFrame/dict, без обращений к БД (ТЗ / .cursorrules п.5).
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd


def _seasonal_index(
    seasonality: pd.DataFrame,
    supplier_id: int,
    category: str | None,
    year: int,
    month: int,
) -> float:
    """
    seasonal_index = month_value / avg_month_value_that_year.
    Сначала ищем строку по категории, иначе — NULL (вся компания/поставщик).
    """
    if seasonality.empty:
        return 1.0

    def _index_for(cat: str | None) -> float | None:
        if cat is None:
            mask = (
                (seasonality["supplier_id"] == supplier_id)
                & (seasonality["category"].isna())
                & (seasonality["year"] == year)
            )
        else:
            mask = (
                (seasonality["supplier_id"] == supplier_id)
                & (seasonality["category"] == cat)
                & (seasonality["year"] == year)
            )
        subset = seasonality.loc[mask]
        if subset.empty:
            return None
        avg = float(subset["total_value"].mean())
        if avg <= 0:
            return 1.0
        month_row = subset[subset["month"] == month]
        if month_row.empty:
            return 1.0
        return float(month_row.iloc[0]["total_value"]) / avg

    idx = _index_for(category)
    if idx is None:
        idx = _index_for(None)
    return float(idx) if idx is not None else 1.0


def _trend_factor(monthly_sales: pd.DataFrame, product_id: int) -> float:
    """Устойчивый рост: отношение среднего за последние 3 мес к предыдущим 3."""
    hist = monthly_sales[monthly_sales["product_id"] == product_id].copy()
    if hist.empty or len(hist) < 6:
        return 1.0
    hist = hist.sort_values(["year", "month"])
    recent = hist.tail(3)["qty_sold"].mean()
    prev = hist.iloc[-6:-3]["qty_sold"].mean()
    if prev <= 0:
        return 1.0
    ratio = recent / prev
    # Ограничиваем влияние тренда
    return float(min(1.35, max(0.75, ratio)))


def _round_to_moq(qty: float, min_ship: int, multiplicity: int) -> float:
    if qty <= 0:
        return 0.0
    mult = max(int(multiplicity or 1), 1)
    minimum = max(int(min_ship or 1), 1)
    rounded = math.ceil(qty / mult) * mult
    if rounded < minimum:
        # поднимаем до min_ship с учётом кратности
        rounded = math.ceil(minimum / mult) * mult
    return float(rounded)


def _urgency(days_cover: float, recommended: float) -> str:
    """Срочность по дням покрытия остатком+в пути (даже если заказ = 0)."""
    if days_cover < 7:
        return "критическая"
    if days_cover < 14:
        return "высокая"
    if days_cover < 30:
        return "средняя"
    if recommended > 0:
        return "средняя"
    return "низкая"


def build_justification(
    *,
    code_1c: str,
    unit: str,
    base_monthly: float,
    seasonal_index: float,
    trend: float,
    stockout_adj: float,
    stockout_days: int,
    outlier_excluded_qty: float,
    current_stock: float,
    in_transit: float,
    lead_time_days: int,
    coverage_days: int,
    raw_need: float,
    recommended_qty: float,
    urgency: str,
) -> str:
    """Человекочитаемое обоснование только из посчитанных цифр (трассируемо)."""
    parts = [
        f"{code_1c}: база {base_monthly:.0f} {unit}/мес (история без выбросов)",
        f"сезонность x{seasonal_index:.2f}",
        f"тренд x{trend:.2f}",
        f"горизонт {coverage_days} дн (lead time {lead_time_days} + запас)",
    ]
    if stockout_adj > 0:
        parts.append(f"+{stockout_adj:.0f} {unit} компенсация stockout ({stockout_days} дн)")
    if outlier_excluded_qty > 0:
        parts.append(
            f"исключено выбросов {outlier_excluded_qty:.0f} {unit} (is_outlier=1)"
        )
    parts.append(f"остаток {current_stock:.0f} {unit}")
    parts.append(f"в пути {in_transit:.0f} {unit}")
    parts.append(f"сырая потребность {raw_need:.0f} → заказ {recommended_qty:.0f} {unit}")
    parts.append(f"срочность: {urgency}")
    return "; ".join(parts) + "."


def calculate_recommendations(
    products: pd.DataFrame,
    sales: pd.DataFrame,
    monthly_sales: pd.DataFrame,
    monthly_stock: pd.DataFrame,
    stockouts: pd.DataFrame,
    transit: pd.DataFrame,
    seasonality: pd.DataFrame,
    *,
    target_date: date | None = None,
    safety_days: int = 14,
) -> pd.DataFrame:
    """
    Считает recommended_qty по каждому активному товару.

    Формула (упрощённо):
      base_monthly = среднее |qty| по месяцам из sales WHERE is_outlier=0
      demand = base_monthly * seasonal_index * trend * (coverage_days/30)
               + stockout_adj
      need   = demand - current_stock - in_transit
      recommended_qty = round_moq(max(need, 0))
    """
    target = target_date or date.today()
    target_year, target_month = target.year, target.month
    # Типичный цикл заказа поставщику (между прогонами расчёта)
    review_cycle_days = 30

    if products.empty:
        return pd.DataFrame()

    sales = sales.copy()
    sales["abs_qty"] = sales["quantity"].abs()
    sales["ym"] = pd.to_datetime(sales["sale_date"]).dt.to_period("M").astype(str)
    clean = sales[sales["is_outlier"].fillna(0).astype(int) == 0]
    clean = clean[clean["quantity"] < 0]

    # Исключённый объём выбросов по продукту
    outliers = sales[sales["is_outlier"].fillna(0).astype(int) == 1]
    outlier_by_product = (
        outliers.groupby("product_id")["abs_qty"].sum()
        if not outliers.empty
        else pd.Series(dtype=float)
    )

    # Помесячный regular demand из очищенных транзакций
    if clean.empty:
        monthly_clean = pd.DataFrame(columns=["product_id", "ym", "qty"])
    else:
        monthly_clean = (
            clean.groupby(["product_id", "ym"], as_index=False)["abs_qty"]
            .sum()
            .rename(columns={"abs_qty": "qty"})
        )

    transit_sum = (
        transit.groupby("product_id")["quantity"].sum()
        if not transit.empty
        else pd.Series(dtype=float)
    )

    stock_latest = pd.DataFrame()
    if not monthly_stock.empty:
        stock_latest = (
            monthly_stock.sort_values(["product_id", "year", "month"])
            .groupby("product_id", as_index=False)
            .tail(1)
            .set_index("product_id")
        )

    rows: list[dict] = []

    for p in products.itertuples(index=False):
        pid = int(p.product_id)
        lead = int(p.lead_time_days or 21)
        # Горизонт: lead time + страховой запас + цикл заказа
        coverage_days = lead + safety_days + review_cycle_days

        prod_months = monthly_clean[monthly_clean["product_id"] == pid]
        if not prod_months.empty:
            # последние до 12 месяцев
            base_monthly = float(prod_months.sort_values("ym").tail(12)["qty"].mean())
        else:
            # fallback на monthly_sales
            ms = monthly_sales[monthly_sales["product_id"] == pid]
            base_monthly = float(ms["qty_sold"].mean()) if not ms.empty else 0.0

        seasonal = _seasonal_index(
            seasonality, int(p.supplier_id), getattr(p, "category", None),
            target_year, target_month,
        )
        trend = _trend_factor(monthly_sales, pid)

        # Stockout compensation: avg daily * days_out (по регулярному спросу)
        so = stockouts[stockouts["product_id"] == pid] if not stockouts.empty else pd.DataFrame()
        stockout_days = int(so["days_out"].fillna(0).sum()) if not so.empty else 0
        daily = base_monthly / 30.0 if base_monthly > 0 else 0.0
        stockout_adj = round(daily * stockout_days * seasonal, 1)

        demand = base_monthly * seasonal * trend * (coverage_days / 30.0) + stockout_adj

        current_stock = 0.0
        if not stock_latest.empty and pid in stock_latest.index:
            current_stock = float(stock_latest.loc[pid, "stock_qty"] or 0.0)

        in_transit = float(transit_sum.get(pid, 0.0) or 0.0)
        outlier_ex = float(outlier_by_product.get(pid, 0.0) or 0.0)

        raw_need = demand - current_stock - in_transit
        recommended = _round_to_moq(
            raw_need,
            int(getattr(p, "min_ship_qty", 1) or 1),
            int(getattr(p, "multiplicity", 1) or 1),
        )

        days_cover = (
            (current_stock + in_transit) / daily if daily > 0 else 999.0
        )
        urg = _urgency(days_cover, recommended)

        # Нулевой заказ с низкой срочностью не показываем; остальное — для менеджера
        if recommended <= 0 and urg == "низкая":
            continue

        unit = getattr(p, "unit", None) or "шт"
        justification = build_justification(
            code_1c=p.code_1c,
            unit=unit,
            base_monthly=base_monthly,
            seasonal_index=seasonal,
            trend=trend,
            stockout_adj=stockout_adj,
            stockout_days=stockout_days,
            outlier_excluded_qty=outlier_ex,
            current_stock=current_stock,
            in_transit=in_transit,
            lead_time_days=lead,
            coverage_days=coverage_days,
            raw_need=max(raw_need, 0.0),
            recommended_qty=recommended,
            urgency=urg,
        )

        rows.append(
            {
                "product_id": pid,
                "supplier_id": int(p.supplier_id),
                "code_1c": p.code_1c,
                "supplier_article": getattr(p, "supplier_article", None),
                "name": p.name,
                "unit": unit,
                "category": getattr(p, "category", None),
                "supplier_name": getattr(p, "supplier_name", None),
                "base_demand": round(base_monthly, 2),
                "seasonality_adj": round(seasonal, 4),
                "stockout_adj": round(stockout_adj, 2),
                "outlier_excluded_qty": round(outlier_ex, 2),
                "current_stock": round(current_stock, 2),
                "in_transit_qty": round(in_transit, 2),
                "recommended_qty": recommended,
                "urgency": urg,
                "justification": justification,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    urgency_order = {"критическая": 0, "высокая": 1, "средняя": 2, "низкая": 3}
    result["_u"] = result["urgency"].map(urgency_order)
    result = result.sort_values(["_u", "recommended_qty"], ascending=[True, False])
    return result.drop(columns=["_u"]).reset_index(drop=True)
