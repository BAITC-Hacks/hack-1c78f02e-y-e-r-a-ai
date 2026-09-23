"""
Y.E.R.A. AI — Streamlit UI для рекомендованных заказов поставщикам.

Запуск:
  streamlit run app.py

Навигация (горизонтальные вкладки):
  Дашборд закупа / Рекомендации / ИИ-Ассистент.
Настройки API — компактный expander в шапке.
Расчёт qty — только детерминированный core/, без LLM.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import run_replenishment_calculation
from database import (
    DB_PATH,
    apply_moq_rounding,
    init_db,
    seed_test_data,
)
from repository import (
    approve_orders,
    fetch_categories,
    fetch_current_stock,
    fetch_dashboard_metrics,
    fetch_monthly_sales_history,
    fetch_products,
    fetch_recommended_orders,
    fetch_suppliers,
    get_connection,
)

try:
    from database import AGENT_SCHEMA_SUMMARY, execute_readonly_query
except ImportError as _imp_err:  # pragma: no cover
    raise ImportError(
        "Не удалось импортировать AGENT_SCHEMA_SUMMARY / execute_readonly_query "
        "из database.py. Сохраните database.py и перезапустите Streamlit "
        f"(streamlit run app.py). Исходная ошибка: {_imp_err}"
    ) from _imp_err

# ---------------------------------------------------------------------------
# Page config & premium dark theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Y.E.R.A. AI · Электрокомплект",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    /* ——— База: угольно-графитовый фон ——— */
    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
    /* Полностью скрыть sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    section[data-testid="stSidebar"] {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
    }
    /* Текст */
    h1, h2, h3, h4, p, label, span, .stMarkdown, .stCaption {
        color: #F8FAFC !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #94A3B8 !important;
    }
    /* Верхний бренд-бар */
    .yera-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.85rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #111827 0%, #1F2937 55%, #0F172A 100%);
        border: 1px solid rgba(249, 115, 22, 0.35);
        box-shadow: 0 0 24px rgba(239, 68, 68, 0.12);
    }
    .yera-topbar .brand {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #FFF !important;
    }
    .yera-topbar .brand span { color: #F97316; }
    .yera-topbar .tagline {
        color: #94A3B8 !important;
        font-size: 0.88rem;
        margin: 0;
    }
    /* Hero */
    .yera-hero {
        background: linear-gradient(120deg, #111827 0%, #1F2937 50%, #7C2D12 140%);
        border-radius: 14px;
        padding: 1.35rem 1.5rem;
        color: #F8FAFC;
        margin-bottom: 1.1rem;
        border: 1px solid rgba(249, 115, 22, 0.28);
        box-shadow: 0 0 28px rgba(249, 115, 22, 0.08);
    }
    .yera-hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 1.55rem;
        letter-spacing: 0.02em;
        color: #FFF !important;
    }
    .yera-hero p { margin: 0; color: #CBD5E1 !important; font-size: 0.95rem; }
    /* KPI-карточки */
    div[data-testid="stMetric"] {
        background: #1F2937 !important;
        border: 1px solid rgba(249, 115, 22, 0.28);
        border-radius: 12px;
        padding: 14px 16px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 18px rgba(239, 68, 68, 0.1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 28px rgba(249, 115, 22, 0.22);
        border-color: #F97316;
    }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #FDBA74 !important;
    }
    /* Dataframes */
    div[data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] > div {
        background: #1F2937 !important;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.05);
    }
    /* Горизонтальные вкладки */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: #111827;
        padding: 0.45rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin-bottom: 0.75rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1rem !important;
        font-weight: 600;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #EF4444 0%, #F97316 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.35);
    }
    /* Кнопки — красно-оранжевый акцент */
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #EF4444 0%, #F97316 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease-in-out !important;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.28);
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        filter: brightness(1.12);
        transform: translateY(-3px);
        box-shadow: 0 10px 22px rgba(249, 115, 22, 0.35);
        color: #FFFFFF !important;
    }
    div[data-testid="stButton"] > button p,
    div[data-testid="stDownloadButton"] > button p {
        color: #FFFFFF !important;
    }
    /* Инпуты */
    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div, .stTextArea textarea {
        background-color: #1F2937 !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
        border-color: rgba(148, 163, 184, 0.3) !important;
    }
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        border-radius: 12px;
        padding: 0.35rem 0.25rem !important;
    }
    /* Окно чата: bordered container со скроллом */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #1E293B !important;
        border: 1px solid rgba(249, 115, 22, 0.32) !important;
        border-radius: 16px !important;
        padding: 15px !important;
        box-shadow: 0 0 24px rgba(56, 189, 248, 0.08);
    }
    div[data-testid="stChatMessageContent"] {
        background: #0F172A !important;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 12px 14px !important;
    }
    /* Пользователь vs ассистент — разные тона пузырей */
    div[data-testid="stChatMessage"]:has(img[alt="👤"]) [data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([aria-label="👤"]) [data-testid="stChatMessageContent"] {
        background: #334155 !important;
        border-color: rgba(249, 115, 22, 0.4);
    }
    div[data-testid="stChatMessage"]:has(img[alt="⚡"]) [data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([aria-label="⚡"]) [data-testid="stChatMessageContent"] {
        background: #0F172A !important;
        border-color: rgba(56, 189, 248, 0.35);
    }
    /* Поле ввода чата у нижнего края */
    [data-testid="stChatInput"] {
        background: #1E293B !important;
        border: 1px solid rgba(249, 115, 22, 0.3) !important;
        border-radius: 12px !important;
        padding: 8px !important;
        margin-top: 0.75rem !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #F8FAFC !important;
    }
    .yera-api-ok {
        color: #86EFAC !important;
        font-size: 0.85rem;
        margin: 0.25rem 0 0 0;
    }
    .urgency-критическая { color: #fecaca; background: #7f1d1d; padding: 2px 8px; border-radius: 6px; }
    .urgency-высокая { color: #ffedd5; background: #9a3412; padding: 2px 8px; border-radius: 6px; }
    .urgency-средняя { color: #fef9c3; background: #854d0e; padding: 2px 8px; border-radius: 6px; }
    .urgency-низкая { color: #dcfce7; background: #166534; padding: 2px 8px; border-radius: 6px; }
    hr { border-color: rgba(148, 163, 184, 0.2) !important; }
    div[data-testid="stExpander"] {
        background: #1F2937;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached data loaders
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


@st.cache_data(ttl=30, show_spinner=False)
def _cached_moq_reference(
    supplier_id: int | None = None,
    category: str | None = None,
) -> pd.DataFrame:
    products = fetch_products(supplier_id, category)
    if products.empty:
        return pd.DataFrame(columns=["product_id", "min_ship_qty", "multiplicity"])
    return products[["product_id", "min_ship_qty", "multiplicity"]].copy()


def _ensure_db() -> None:
    if not DB_PATH.exists():
        init_db(DB_PATH)
        seed_test_data(clear=True)


def _fmt_int(n: float | int) -> str:
    return f"{int(round(n)):,}".replace(",", " ")


# ---------------------------------------------------------------------------
# Top bar + filters + API settings (вместо sidebar / вкладки Настройки)
# ---------------------------------------------------------------------------

def _env_openai_key() -> str:
    """Ключ из системного окружения — главный источник истины."""
    return (os.getenv("OPENAI_API_KEY") or "").strip()


def _get_openai_api_key() -> str:
    """
    Порядок поиска ключа:
    1) os.getenv("OPENAI_API_KEY") — приоритет, переживает Ctrl+R
    2) st.session_state["openai_api_key"] — ручной ввод из expander
    3) st.secrets
    """
    env_key = _env_openai_key()
    if env_key:
        return env_key

    session_key = str(st.session_state.get("openai_api_key") or "").strip()
    if session_key:
        return session_key

    try:
        return str(st.secrets.get("OPENAI_API_KEY", "") or "").strip()
    except Exception:
        return ""


def render_api_settings_panel() -> None:
    """Компактный expander в шапке: ключ API + служебные действия."""
    with st.expander("⚙️ Настройки API", expanded=False):
        env_key = _env_openai_key()
        if env_key:
            st.markdown(
                '<p class="yera-api-ok">✓ OPENAI_API_KEY найден в системном окружении '
                f"(…{env_key[-4:]}). Ввод не требуется.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "Ключ не найден в окружении. Задайте OPENAI_API_KEY в системе "
                "или введите ниже (сохранится до перезапуска Streamlit)."
            )
            manual = st.text_input(
                "API-ключ OpenAI (fallback)",
                value=st.session_state.get("openai_api_key", ""),
                type="password",
                key="openai_api_key_input",
                placeholder="sk-…",
            )
            if manual.strip():
                st.session_state["openai_api_key"] = manual.strip()

        st.caption(f"БД: `{DB_PATH}` · существует: {DB_PATH.exists()}")
        if st.button("Пересоздать тестовую БД", key="reseed_db"):
            init_db(DB_PATH)
            stats = seed_test_data(clear=True)
            st.cache_data.clear()
            st.success(f"БД обновлена: {stats}")


def render_topbar() -> None:
    col_brand, col_settings = st.columns([4, 1.4])
    with col_brand:
        st.markdown(
            """
            <div class="yera-topbar">
                <div>
                    <div class="brand">Y.E.R.A. <span>AI</span></div>
                    <p class="tagline">Электрокомплект · премиум-логистика закупа</p>
                </div>
                <div class="tagline">storage.db · рекомендованные заказы</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_settings:
        render_api_settings_panel()


def render_filters() -> tuple[int | None, str | None]:
    """Горизонтальные фильтры поставщик / категория (вместо sidebar)."""
    suppliers = _cached_suppliers()
    supplier_names = ["Все поставщики"] + suppliers["supplier_name"].tolist()

    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        chosen = st.selectbox("Поставщик", supplier_names, key="filter_supplier")
    supplier_id: int | None = None
    if chosen != "Все поставщики" and not suppliers.empty:
        supplier_id = int(
            suppliers.loc[suppliers["supplier_name"] == chosen, "supplier_id"].iloc[0]
        )

    cats = _cached_categories(supplier_id)
    with f2:
        cat_chosen = st.selectbox(
            "Категория",
            ["Все категории"] + cats,
            key="filter_category",
        )
    category = None if cat_chosen == "Все категории" else cat_chosen

    with f3:
        st.write("")
        st.write("")
        if st.button("↻ Обновить данные", use_container_width=True, key="refresh_cache"):
            st.cache_data.clear()
            st.rerun()

    return supplier_id, category


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_dashboard(supplier_id: int | None, category: str | None) -> None:
    st.markdown(
        """
        <div class="yera-hero">
            <h1>📊 Дашборд закупа</h1>
            <p>Сводка по остаткам, дефициту и последним рекомендованным заказам.
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

    st.markdown("#### Последний расчёт (топ срочности)")
    orders = _cached_orders(None, supplier_id, category, None)
    if orders.empty:
        st.info("Расчётов ещё нет. Откройте вкладку «Рекомендации» и запустите Y.E.R.A. AI.")
    else:
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

    st.markdown("---")
    page_stock_sales(supplier_id, category)


def page_stock_sales(supplier_id: int | None, category: str | None) -> None:
    st.markdown("#### Остатки и история продаж")
    mode = st.radio(
        "Данные",
        ["Текущие остатки", "Помесячные продажи"],
        horizontal=True,
        label_visibility="collapsed",
        key="dash_stock_mode",
    )

    if mode == "Текущие остатки":
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
                height=420,
                column_config={
                    "Остаток": st.column_config.NumberColumn(format="%.0f"),
                },
            )
    else:
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
            st.dataframe(pivot, use_container_width=True, hide_index=True, height=420)

            chart_src = (
                sales.groupby("period", as_index=False)["qty_sold"].sum().sort_values("period")
            )
            st.caption("Суммарные продажи по месяцам (выборка)")
            st.bar_chart(chart_src, x="period", y="qty_sold", height=240)


def page_calculation(supplier_id: int | None, category: str | None) -> None:
    st.markdown(
        """
        <div class="yera-hero">
            <h1>📦 Рекомендации к заказу</h1>
            <p>Базовая потребность → сезонность и тренд → stockout → выбросы →
            вычет остатка и товара в пути. Количество считает Python, не LLM.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _col_a, col_b = st.columns([2, 1])
    with col_b:
        safety_days = st.slider("Страховой запас, дней", 7, 45, 14, 1)

    st.write("")
    run_clicked = st.button(
        "▶ Запустить интеллектуальный расчет пополнения Y.E.R.A. AI",
        type="primary",
        use_container_width=True,
        key="run_yera_calc",
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
    if result and result.get("run_id"):
        orders = _cached_orders(result["run_id"], supplier_id, category, None)
    else:
        orders = _cached_orders(None, supplier_id, category, None)

    if orders.empty:
        st.info("Нажмите кнопку выше, чтобы сформировать рекомендованные заказы.")
        page_orders(supplier_id, category)
        return

    moq_ref = _cached_moq_reference(supplier_id, category)
    orders = orders.merge(moq_ref, on="product_id", how="left")
    moq_pairs = [
        apply_moq_rounding(
            float(row.recommended_qty or 0),
            int(row.min_ship_qty or 1) if pd.notna(row.min_ship_qty) else 1,
            int(row.multiplicity or 1) if pd.notna(row.multiplicity) else 1,
        )
        for row in orders.itertuples(index=False)
    ]
    orders = orders.copy()
    orders["moq_final_qty"] = [pair[0] for pair in moq_pairs]
    orders["moq_note"] = [pair[1] for pair in moq_pairs]

    product_ids = orders["product_id"].dropna().astype(int).unique().tolist()
    costs_df = pd.DataFrame(columns=["product_id", "unit_cost"])
    if product_ids:
        placeholders = ",".join("?" * len(product_ids))
        conn = get_connection()
        try:
            costs_df = pd.read_sql_query(
                f"""
                SELECT product_id, COALESCE(unit_cost, 0) AS unit_cost
                FROM products
                WHERE product_id IN ({placeholders})
                """,
                conn,
                params=product_ids,
            )
        finally:
            conn.close()

    orders = orders.merge(costs_df, on="product_id", how="left")
    orders["unit_cost"] = orders["unit_cost"].fillna(0.0)
    qty_col = "moq_final_qty" if "moq_final_qty" in orders.columns else "recommended_qty"
    orders["line_cost"] = orders[qty_col].fillna(0) * orders["unit_cost"]

    total_purchase = float(orders["line_cost"].sum())
    urgent_mask = orders["urgency"].isin(["критическая", "высокая"])
    urgent_purchase = float(orders.loc[urgent_mask, "line_cost"].sum())

    st.markdown("---")
    st.markdown("#### Финансовый анализ и прогноз кассового разрыва")
    budget = st.number_input(
        "Доступный бюджет закупа, ₸",
        min_value=0.0,
        value=2_000_000.0,
        step=50_000.0,
        format="%.0f",
        key="purchase_budget",
    )
    cash_gap = max(total_purchase - float(budget), 0.0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Общая сумма закупа", f"{total_purchase:,.0f} ₸".replace(",", " "))
    m2.metric(
        "Из них срочно (крит. + высокая)",
        f"{urgent_purchase:,.0f} ₸".replace(",", " "),
    )
    m3.metric(
        "Кассовый разрыв",
        f"{cash_gap:,.0f} ₸".replace(",", " "),
        delta=None if cash_gap <= 0 else "превышение бюджета",
        delta_color="inverse",
    )

    if cash_gap > 0:
        st.warning(
            f"Кассовый разрыв **{cash_gap:,.0f} ₸**. "
            "Рекомендуется в первую очередь закупать только срочные позиции "
            "(срочность «критическая» и «высокая»), остальное — отложить "
            "до пополнения бюджета.".replace(",", " ")
        )
    else:
        st.success("Бюджета достаточно для полного рекомендованного закупа.")

    st.markdown("#### Рекомендованные заказы")
    display = orders[
        [
            "id",
            "supplier_name",
            "supplier_article",
            "name",
            "recommended_qty",
            "moq_final_qty",
            "unit",
            "urgency",
            "moq_note",
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
            "moq_final_qty": "Кол-во с MOQ",
            "unit": "Ед.",
            "urgency": "Срочность",
            "moq_note": "MOQ-проверка",
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

    csv_bytes = display.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button(
        label="📥 Скачать отчет закупа в CSV",
        data=csv_bytes,
        file_name="yera_recommended_orders.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
        help="UTF-8 с BOM — корректно открывается в Microsoft Excel",
        key="download_calc_orders_csv",
    )

    drafts = orders[orders["status"] == "draft"]
    if not drafts.empty:
        st.markdown("---")
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

    st.markdown("---")
    page_orders(supplier_id, category)


def page_orders(supplier_id: int | None, category: str | None) -> None:
    st.markdown("#### Заказы по поставщикам")
    status_filter = st.selectbox(
        "Статус",
        ["Все", "draft", "approved", "rejected", "sent"],
        index=0,
        key="orders_status_filter",
    )
    status = None if status_filter == "Все" else status_filter
    orders = _cached_orders(None, supplier_id, category, status)

    if orders.empty:
        st.info("Нет заказов для выбранных фильтров.")
        return

    for supplier_name, grp in orders.groupby("supplier_name", sort=True):
        with st.expander(f"{supplier_name} · {len(grp)} поз.", expanded=False):
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


# ---------------------------------------------------------------------------
# ИИ-Ассистент (Чат)
# ---------------------------------------------------------------------------

AGENT_PROMPT_PATH = ROOT / "agent_prompt.txt"
CHAT_SQL_INSTRUCTIONS = """
# РЕЖИМ SQL-АССИСТЕНТА
По вопросу менеджера ты ДОЛЖЕН сначала сформировать безопасный SQL-запрос
только типа SELECT (или WITH … SELECT) к SQLite storage.db, чтобы ответить
цифрами из БД. Не считай количества заказов сам — только читай данные.

Верни СТРОГО один JSON-объект без markdown-ограждений:
{
  "sql": "SELECT …",
  "reason": "кратко, зачем этот запрос"
}

Правила SQL:
- Только чтение: запрещены INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA/ATTACH.
- Используй только таблицы и поля из схемы ниже.
- JOIN по product_id / supplier_id где нужно.
- quantity в sales_transactions отрицательный при отгрузке — для объёма продаж
  используй ABS(quantity) или -quantity.
- client_hash — обезличенный id; не пытайся раскрыть клиента.
- Добавляй LIMIT (не больше 100), если выборка может быть большой.
- Не выдумывай таблицы и колонки.

""" + AGENT_SCHEMA_SUMMARY


def _load_agent_system_prompt() -> str:
    base = ""
    if AGENT_PROMPT_PATH.exists():
        base = AGENT_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return f"{base}\n\n{CHAT_SQL_INSTRUCTIONS}".strip()


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def _openai_chat(
    messages: list[dict],
    *,
    api_key: str,
    temperature: float = 0.2,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=st.session_state.get("openai_model", "gpt-4o-mini"),
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _dataframe_for_llm(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "(пусто: 0 строк)"
    return df.head(max_rows).to_csv(index=False)


def _run_yera_chat_turn(user_text: str, api_key: str) -> dict:
    system = _load_agent_system_prompt()
    plan_raw = _openai_chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Сформируй JSON с безопасным SELECT для ответа на вопрос.\n"
                    f"Вопрос менеджера: {user_text}"
                ),
            },
        ],
        api_key=api_key,
        temperature=0.0,
    )

    sql: str | None = None
    reason = ""
    try:
        plan = _extract_json_object(plan_raw)
        raw_sql = plan.get("sql")
        sql = str(raw_sql).strip() if raw_sql else None
        reason = str(plan.get("reason") or "")
    except (json.JSONDecodeError, TypeError, ValueError):
        m = re.search(
            r"(?is)\b(WITH\b[\s\S]+SELECT\b[\s\S]+|SELECT\b[\s\S]+)",
            plan_raw,
        )
        if m:
            sql = m.group(1).strip().rstrip(";")
        else:
            return {
                "content": plan_raw
                or "Не удалось сформировать SQL по вопросу. Переформулируйте запрос.",
                "dataframe": None,
                "sql": None,
            }

    df: pd.DataFrame | None = None
    sql_error: str | None = None
    if sql:
        try:
            df = execute_readonly_query(sql, limit=100)
        except Exception as exc:  # noqa: BLE001
            sql_error = str(exc)

    if sql_error:
        answer = _openai_chat(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Вопрос: {user_text}\nSQL: {sql}\nОшибка: {sql_error}\n"
                        "Объясни менеджеру кратко на русском, что пошло не так."
                    ),
                },
            ],
            api_key=api_key,
        )
        return {"content": answer, "dataframe": None, "sql": sql}

    data_block = _dataframe_for_llm(df) if df is not None else "(запрос к БД не выполнялся)"
    answer = _openai_chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Вопрос менеджера: {user_text}\n"
                    f"Зачем SQL: {reason or '—'}\n"
                    f"SQL:\n{sql or 'не использовался'}\n\n"
                    f"Результат из storage.db:\n{data_block}\n\n"
                    "Ответь по-русски кратко. Ссылайся на цифры из результата. "
                    "Не придумывай значения. Не предлагай автоотправку заказа."
                ),
            },
        ],
        api_key=api_key,
        temperature=0.3,
    )

    show_table = df is not None and not df.empty and len(df.columns) > 0
    return {
        "content": answer,
        "dataframe": df if show_table else None,
        "sql": sql,
    }


def _render_chat_bubble(msg: dict) -> None:
    """Одно сообщение с разными аватарками user / Y.E.R.A."""
    role = msg.get("role", "assistant")
    if role == "user":
        avatar = "👤"
    else:
        avatar = "⚡"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.get("content") or "")
        if msg.get("sql"):
            with st.expander("SQL-запрос", expanded=False):
                st.code(msg["sql"], language="sql")
        df = msg.get("dataframe")
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)


def page_chat_assistant() -> None:
    st.markdown(
        """
        <div class="yera-hero">
            <h1>💬 ИИ-Ассистент Y.E.R.A.</h1>
            <p>Вопросы о товарах, продажах и заказах на естественном языке.
            Ассистент переводит их в безопасный SELECT к storage.db.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Здравствуйте! Я **Y.E.R.A. AI**. Спросите, например: "
                    "«Какие товары ИЭК с остатком меньше 50?» или "
                    "«Покажи критические рекомендованные заказы»."
                ),
                "dataframe": None,
                "sql": None,
            }
        ]

    col_cfg, col_clear = st.columns([3, 1])
    with col_cfg:
        st.selectbox(
            "Модель OpenAI",
            ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
            index=0,
            key="openai_model",
        )
    with col_clear:
        st.write("")
        st.write("")
        if st.button("Очистить чат", use_container_width=True, key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()

    api_key = _get_openai_api_key()
    # Жёлтое предупреждение — ТОЛЬКО если ключа нет нигде (env / session / secrets)
    if not api_key:
        st.warning(
            "OPENAI_API_KEY не найден. Задайте переменную окружения "
            "или откройте «⚙️ Настройки API» в шапке."
        )

    # Фиксированное окно истории (как в мессенджере)
    try:
        chat_box = st.container(height=480, border=True)
    except TypeError:
        chat_box = st.container(border=True)

    with chat_box:
        st.markdown(
            """
            <div style="background:#1E293B;padding:15px;border-radius:12px;margin:-0.5rem;">
            <p style="color:#94A3B8;font-size:0.8rem;margin:0 0 0.75rem 0;">
            Y.E.R.A. AI · диалог с базой storage.db
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for msg in st.session_state.chat_messages:
            _render_chat_bubble(msg)

    # Поле ввода прижато к низу (Streamlit pin) — сразу под окном истории
    prompt = st.chat_input("Напишите вопрос о товарах, продажах, остатках…")
    if not prompt:
        return

    st.session_state.chat_messages.append(
        {"role": "user", "content": prompt, "dataframe": None, "sql": None}
    )

    if not api_key:
        err = (
            "Нет API-ключа OpenAI. Задайте `OPENAI_API_KEY` в окружении "
            "или в «⚙️ Настройки API»."
        )
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": err, "dataframe": None, "sql": None}
        )
        st.rerun()
        return

    with st.spinner("Y.E.R.A. AI анализирует данные…"):
        try:
            result = _run_yera_chat_turn(prompt, api_key)
        except Exception as exc:  # noqa: BLE001
            result = {
                "content": f"Ошибка обращения к OpenAI / БД: {exc}",
                "dataframe": None,
                "sql": None,
            }

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": result["content"],
            "dataframe": result.get("dataframe"),
            "sql": result.get("sql"),
        }
    )
    st.rerun()


# ---------------------------------------------------------------------------
# Main — горизонтальные вкладки (без «Настройки»)
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_db()
    render_topbar()
    supplier_id, category = render_filters()

    tab_dash, tab_reco, tab_chat = st.tabs(
        [
            "📊 Дашборд закупа",
            "📦 Рекомендации",
            "💬 ИИ-Ассистент",
        ]
    )

    with tab_dash:
        page_dashboard(supplier_id, category)
    with tab_reco:
        page_calculation(supplier_id, category)
    with tab_chat:
        page_chat_assistant()


if __name__ == "__main__":
    main()
