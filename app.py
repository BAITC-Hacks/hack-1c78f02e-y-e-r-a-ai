"""
Y.E.R.A. AI — Streamlit UI для рекомендованных заказов поставщикам.

Запуск:
  streamlit run app.py

Страницы: Дашборд / Остатки и продажи / Расчёт / Заказы / Настройки.
Расчёт qty — только детерминированный core/, без LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import run_replenishment_calculation
from database import DB_PATH, init_db, seed_test_data
from repository import (
    approve_orders,
    fetch_categories,
    fetch_current_stock,
    fetch_dashboard_metrics,
    fetch_monthly_sales_history,
    fetch_recommended_orders,
    fetch_suppliers,
)

# ---------------------------------------------------------------------------
# Page config & styles
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Y.E.R.A. AI · Электрокомплект",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #0f2744 0%, #163a5f 100%);
        border: 1px solid #2a5a8a;
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #9ec3e8 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f2f7fc !important;
        font-weight: 700;
    }
    .yera-hero {
        background: linear-gradient(120deg, #0b1f36 0%, #1a4a7a 55%, #0d9488 100%);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        color: #f8fafc;
        margin-bottom: 1rem;
        border: 1px solid #2dd4bf33;
    }
    .yera-hero h1 { margin: 0 0 0.35rem 0; font-size: 1.65rem; letter-spacing: 0.02em; }
    .yera-hero p { margin: 0; opacity: 0.9; font-size: 0.95rem; }
    .urgency-критическая { color: #fecaca; background: #7f1d1d; padding: 2px 8px; border-radius: 6px; }
    .urgency-высокая { color: #ffedd5; background: #9a3412; padding: 2px 8px; border-radius: 6px; }
    .urgency-средняя { color: #fef9c3; background: #854d0e; padding: 2px 8px; border-radius: 6px; }
    .urgency-низкая { color: #dcfce7; background: #166534; padding: 2px 8px; border-radius: 6px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached data loaders (тяжёлые запросы не на каждый rerender)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def _cached_suppliers() -> pd.DataFrame:
    return fetch_suppliers()


@st.cache_data(ttl=30, show_spinner=False)
def _cached_categories(supplier_id: int | None) -> list[str]:
    return fetch_categories(supplier_id)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_metrics(supplier_id: int | None, category: str | None) -> dict:
    return fetch_dashboard_metrics(supplier_id, category)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_stock(supplier_id: int | None, category: str | None) -> pd.DataFrame:
    return fetch_current_stock(supplier_id, category)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_sales_history(supplier_id: int | None, category: str | None) -> pd.DataFrame:
    return fetch_monthly_sales_history(supplier_id, category)


@st.cache_data(ttl=15, show_spinner=False)
def _cached_orders(
    run_id: str | None,
    supplier_id: int | None,
    category: str | None,
    status: str | None,
) -> pd.DataFrame:
    return fetch_recommended_orders(run_id, supplier_id, category, status)


def _ensure_db() -> None:
    if not DB_PATH.exists():
        init_db(DB_PATH)
        seed_test_data(clear=True)


def _fmt_int(n: float | int) -> str:
    return f"{int(round(n)):,}".replace(",", " ")


URGENCY_COLORS = {
    "критическая": "#b91c1c",
    "высокая": "#c2410c",
    "средняя": "#a16207",
    "низкая": "#15803d",
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, int | None, str | None]:
    with st.sidebar:
        st.markdown("### Y.E.R.A. AI")
        st.caption("Электрокомплект · закуп")

        page = st.radio(
            "Раздел",
            ["Дашборд", "Остатки и продажи", "Расчёт", "Заказы", "Настройки"],
            label_visibility="collapsed",
        )
        st.divider()

        suppliers = _cached_suppliers()
        supplier_names = ["Все поставщики"] + suppliers["supplier_name"].tolist()
        chosen = st.selectbox("Поставщик", supplier_names)
        supplier_id: int | None = None
        if chosen != "Все поставщики":
            supplier_id = int(
                suppliers.loc[suppliers["supplier_name"] == chosen, "supplier_id"].iloc[0]
            )

        cats = _cached_categories(supplier_id)
        cat_options = ["Все категории"] + cats
        cat_chosen = st.selectbox("Категория", cat_options)
        category = None if cat_chosen == "Все категории" else cat_chosen

        st.divider()
        st.caption(f"БД: `{DB_PATH.name}`")
        if st.button("Обновить кэш данных", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return page, supplier_id, category


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_dashboard(supplier_id: int | None, category: str | None) -> None:
    st.markdown(
        """
        <div class="yera-hero">
            <h1>Y.E.R.A. AI · Дашборд закупа</h1>
            <p>Рекомендованные заказы на основе истории продаж, сезонности, stockout и очистки выбросов.
            Отправка поставщику — только после утверждения менеджером.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = _cached_metrics(supplier_id, category)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Товаров в выборке", _fmt_int(metrics["products_count"]))
    c2.metric("Критический дефицит", _fmt_int(metrics["critical_count"]))
    c3.metric("Объём в пути", _fmt_int(metrics["transit_qty"]))
    c4.metric("Черновики заказов", _fmt_int(metrics["draft_orders"]))

    st.subheader("Последний расчёт (топ срочности)")
    orders = _cached_orders(None, supplier_id, category, None)
    if orders.empty:
        st.info("Расчётов ещё нет. Перейдите в раздел «Расчёт» и запустите Y.E.R.A. AI.")
        return

    preview = orders.head(12)[
        [
            "supplier_name",
            "supplier_article",
            "name",
            "recommended_qty",
            "unit",
            "urgency",
            "status",
            "justification",
        ]
    ].rename(
        columns={
            "supplier_name": "Поставщик",
            "supplier_article": "Артикул",
            "name": "Номенклатура",
            "recommended_qty": "К заказу",
            "unit": "Ед.",
            "urgency": "Срочность",
            "status": "Статус",
            "justification": "Обоснование",
        }
    )
    st.dataframe(preview, use_container_width=True, hide_index=True, height=360)


def page_stock_sales(supplier_id: int | None, category: str | None) -> None:
    st.header("Остатки и история продаж")
    tab_stock, tab_sales = st.tabs(["Текущие остатки", "Помесячные продажи"])

    with tab_stock:
        stock = _cached_stock(supplier_id, category)
        if stock.empty:
            st.warning("Нет данных по остаткам для выбранных фильтров.")
        else:
            view = stock[
                [
                    "supplier_name",
                    "code_1c",
                    "supplier_article",
                    "name",
                    "category",
                    "unit",
                    "warehouse",
                    "current_stock",
                    "year",
                    "month",
                ]
            ].rename(
                columns={
                    "supplier_name": "Поставщик",
                    "code_1c": "Код 1С",
                    "supplier_article": "Артикул",
                    "name": "Номенклатура",
                    "category": "Категория",
                    "unit": "Ед.",
                    "warehouse": "Склад",
                    "current_stock": "Остаток",
                    "year": "Год",
                    "month": "Месяц",
                }
            )
            st.dataframe(
                view,
                use_container_width=True,
                hide_index=True,
                height=480,
                column_config={
                    "Остаток": st.column_config.NumberColumn(format="%.0f"),
                },
            )

    with tab_sales:
        sales = _cached_sales_history(supplier_id, category)
        if sales.empty:
            st.warning("Нет истории продаж.")
        else:
            sales = sales.copy()
            sales["period"] = (
                sales["year"].astype(str) + "-" + sales["month"].astype(str).str.zfill(2)
            )
            pivot = sales.pivot_table(
                index=["code_1c", "supplier_article", "name", "unit"],
                columns="period",
                values="qty_sold",
                aggfunc="sum",
            ).reset_index()
            pivot.columns.name = None
            st.dataframe(pivot, use_container_width=True, hide_index=True, height=480)

            chart_src = (
                sales.groupby("period", as_index=False)["qty_sold"].sum().sort_values("period")
            )
            st.caption("Суммарные продажи по месяцам (выборка)")
            st.bar_chart(chart_src, x="period", y="qty_sold", height=240)


def page_calculation(supplier_id: int | None, category: str | None) -> None:
    st.header("Интеллектуальный расчёт пополнения")
    st.markdown(
        "Алгоритм: **базовая потребность** → **сезонность и тренд** → "
        "**компенсация stockout** → **исключение выбросов** (`detect_outliers`) → "
        "вычет остатка и товара в пути. Количество считает Python, не LLM."
    )

    col_a, col_b = st.columns([2, 1])
    with col_b:
        safety_days = st.slider("Страховой запас, дней", 7, 45, 14, 1)

    st.write("")
    run_clicked = st.button(
        "Запустить интеллектуальный расчет пополнения Y.E.R.A. AI",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        with st.spinner("Y.E.R.A. AI считает потребность…"):
            result = run_replenishment_calculation(
                supplier_id=supplier_id,
                category=category,
                safety_days=safety_days,
            )
        st.cache_data.clear()
        st.session_state["last_run"] = result
        st.success(
            f"Готово. Run `{result['run_id']}`: "
            f"выбросов помечено **{result['outliers_found']}**, "
            f"позиций сохранено **{result['saved']}** (status=draft)."
        )

    result = st.session_state.get("last_run")
    orders = pd.DataFrame()
    if result and result.get("run_id"):
        orders = _cached_orders(result["run_id"], supplier_id, category, None)
    else:
        orders = _cached_orders(None, supplier_id, category, None)

    if orders.empty:
        st.info("Нажмите кнопку выше, чтобы сформировать рекомендованные заказы.")
        return

    st.subheader("Рекомендованные заказы")
    display = orders[
        [
            "id",
            "supplier_name",
            "supplier_article",
            "name",
            "recommended_qty",
            "unit",
            "urgency",
            "justification",
            "status",
            "current_stock",
            "in_transit_qty",
            "outlier_excluded_qty",
        ]
    ].rename(
        columns={
            "id": "ID",
            "supplier_name": "Поставщик",
            "supplier_article": "Артикул",
            "name": "Номенклатура",
            "recommended_qty": "Рекомендуемое кол-во",
            "unit": "Ед.",
            "urgency": "Срочность",
            "justification": "Обоснование",
            "status": "Статус",
            "current_stock": "Остаток",
            "in_transit_qty": "В пути",
            "outlier_excluded_qty": "Исключено выбросов",
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=440,
        column_config={
            "Рекомендуемое кол-во": st.column_config.NumberColumn(format="%.0f"),
            "Обоснование": st.column_config.TextColumn(width="large"),
            "Срочность": st.column_config.TextColumn(width="small"),
        },
    )

    # Утверждение прямо со страницы расчёта
    drafts = orders[orders["status"] == "draft"]
    if not drafts.empty:
        st.divider()
        st.markdown("#### Утверждение менеджером")
        manager = st.text_input(
            "ФИО / логин утверждающего",
            value=st.session_state.get("manager_name", ""),
            key="calc_manager",
        )
        selected = st.multiselect(
            "Позиции к утверждению",
            options=drafts["id"].tolist(),
            format_func=lambda i: (
                f"#{i} · {drafts.loc[drafts['id']==i, 'supplier_article'].iloc[0]} · "
                f"{int(drafts.loc[drafts['id']==i, 'recommended_qty'].iloc[0])} "
                f"{drafts.loc[drafts['id']==i, 'unit'].iloc[0]}"
            ),
            default=drafts["id"].tolist(),
        )
        if st.button("Утвердить заказ", type="primary", key="approve_from_calc"):
            if not manager.strip():
                st.error("Укажите, кто утверждает заказ (approved_by).")
            elif not selected:
                st.warning("Выберите хотя бы одну позицию.")
            else:
                n = approve_orders(selected, manager.strip())
                st.session_state["manager_name"] = manager.strip()
                st.cache_data.clear()
                st.success(
                    f"Утверждено позиций: {n}. Статус draft → approved. "
                    "Автоотправка поставщику не выполняется."
                )
                st.rerun()


def page_orders(supplier_id: int | None, category: str | None) -> None:
    st.header("Список заказов по поставщикам")
    status_filter = st.selectbox(
        "Статус",
        ["Все", "draft", "approved", "rejected", "sent"],
        index=0,
    )
    status = None if status_filter == "Все" else status_filter
    orders = _cached_orders(None, supplier_id, category, status)

    if orders.empty:
        st.info("Нет заказов для выбранных фильтров.")
        return

    for supplier_name, grp in orders.groupby("supplier_name", sort=True):
        with st.expander(f"{supplier_name} · {len(grp)} поз.", expanded=True):
            view = grp[
                [
                    "id",
                    "supplier_article",
                    "name",
                    "recommended_qty",
                    "unit",
                    "urgency",
                    "status",
                    "justification",
                    "approved_by",
                ]
            ].rename(
                columns={
                    "id": "ID",
                    "supplier_article": "Артикул",
                    "name": "Номенклатура",
                    "recommended_qty": "Кол-во",
                    "unit": "Ед.",
                    "urgency": "Срочность",
                    "status": "Статус",
                    "justification": "Обоснование",
                    "approved_by": "Утвердил",
                }
            )
            st.dataframe(view, use_container_width=True, hide_index=True)

            drafts = grp[grp["status"] == "draft"]
            if not drafts.empty:
                manager = st.text_input(
                    f"Утверждающий ({supplier_name})",
                    value=st.session_state.get("manager_name", ""),
                    key=f"mgr_{supplier_name}",
                )
                if st.button(
                    f"Утвердить заказ · {supplier_name}",
                    key=f"approve_{supplier_name}",
                    type="primary",
                ):
                    if not manager.strip():
                        st.error("Укажите approved_by.")
                    else:
                        n = approve_orders(drafts["id"].tolist(), manager.strip())
                        st.session_state["manager_name"] = manager.strip()
                        st.cache_data.clear()
                        st.success(f"{supplier_name}: утверждено {n} позиций.")
                        st.rerun()


def page_settings() -> None:
    st.header("Настройки")
    st.write(f"**Путь к БД:** `{DB_PATH}`")
    st.write(f"**Файл существует:** {DB_PATH.exists()}")

    st.subheader("Демо-данные")
    st.caption(
        "Пересоздаёт storage.db с тестовыми товарами ИЭК / Systeme Electric "
        "и аномальными крупными заказами для проверки выбросов."
    )
    if st.button("Пересоздать тестовую БД", type="secondary"):
        init_db(DB_PATH)
        stats = seed_test_data(clear=True)
        st.cache_data.clear()
        st.success(f"БД обновлена: {stats}")

    st.subheader("Методология (кратко)")
    st.markdown(
        """
1. **Базовая потребность** — среднее месячных отгрузок без `is_outlier`.
2. **Сезонность** — `month_value / avg` из `seasonality_reference` × тренд 3/3 мес.
3. **Stockout** — `avg_daily × days_out` добавляется к потребности.
4. **Выбросы** — `detect_outliers()` (IQR + доля клиента в месяце), флаг в БД.
5. **Итог** — спрос на горизонт lead time + safety − остаток − в пути, округление по MOQ/кратности.
6. Заказ **не отправляется** автоматически: только `draft` → `approved` человеком.
        """
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_db()
    page, supplier_id, category = render_sidebar()

    if page == "Дашборд":
        page_dashboard(supplier_id, category)
    elif page == "Остатки и продажи":
        page_stock_sales(supplier_id, category)
    elif page == "Расчёт":
        page_calculation(supplier_id, category)
    elif page == "Заказы":
        page_orders(supplier_id, category)
    else:
        page_settings()


if __name__ == "__main__":
    main()
