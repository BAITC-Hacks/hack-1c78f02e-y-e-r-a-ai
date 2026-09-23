"""
Инициализация SQLite-базы storage.db и заполнение реалистичными тестовыми данными.

При запуске:
  python database.py
создаёт storage.db (если нет), все таблицы по схеме из .cursorrules
и загружает демо-данные (ИЭК / Systeme Electric), включая аномально крупные
разовые заказы одного клиента для проверки detect_outliers().
"""

from __future__ import annotations

import hashlib
import random
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "storage.db"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    supplier_name   TEXT NOT NULL UNIQUE,
    lead_time_days  INTEGER,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS products (
    product_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    code_1c           TEXT NOT NULL UNIQUE,
    supplier_article  TEXT,
    name              TEXT NOT NULL,
    unit              TEXT,
    category          TEXT,
    supplier_id       INTEGER REFERENCES suppliers(supplier_id),
    min_ship_qty      INTEGER DEFAULT 1,
    multiplicity      INTEGER DEFAULT 1,
    is_active         INTEGER DEFAULT 1,
    created_at        TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_products_supplier ON products(supplier_id);

CREATE TABLE IF NOT EXISTS sales_transactions (
    transaction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    sale_date        TEXT NOT NULL,
    doc_number       TEXT,
    doc_name         TEXT,
    product_id       INTEGER NOT NULL REFERENCES products(product_id),
    unit             TEXT,
    warehouse        TEXT,
    quantity         REAL NOT NULL,
    client_hash      TEXT,
    is_outlier       INTEGER DEFAULT 0,
    outlier_reason   TEXT,
    imported_at      TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sales_product_date ON sales_transactions(product_id, sale_date);
CREATE INDEX IF NOT EXISTS idx_sales_outlier ON sales_transactions(is_outlier);

CREATE TABLE IF NOT EXISTS monthly_stock (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id   INTEGER NOT NULL REFERENCES products(product_id),
    warehouse    TEXT,
    year         INTEGER NOT NULL,
    month        INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    stock_qty    REAL,
    UNIQUE(product_id, warehouse, year, month)
);
CREATE INDEX IF NOT EXISTS idx_stock_product_period ON monthly_stock(product_id, year, month);

CREATE TABLE IF NOT EXISTS monthly_sales (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id   INTEGER NOT NULL REFERENCES products(product_id),
    year         INTEGER NOT NULL,
    month        INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    qty_sold     REAL,
    UNIQUE(product_id, year, month)
);
CREATE INDEX IF NOT EXISTS idx_msales_product_period ON monthly_sales(product_id, year, month);

CREATE TABLE IF NOT EXISTS stockout_periods (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id       INTEGER NOT NULL REFERENCES products(product_id),
    warehouse        TEXT,
    start_date       TEXT NOT NULL,
    end_date         TEXT,
    days_out         INTEGER,
    detection_method TEXT,
    created_at       TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_stockout_product ON stockout_periods(product_id);

CREATE TABLE IF NOT EXISTS transit_orders (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id       INTEGER NOT NULL REFERENCES products(product_id),
    po_number        TEXT,
    po_date          TEXT,
    expected_arrival TEXT,
    quantity         REAL NOT NULL,
    created_at       TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_transit_product ON transit_orders(product_id);

CREATE TABLE IF NOT EXISTS seasonality_reference (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    supplier_id  INTEGER REFERENCES suppliers(supplier_id),
    category     TEXT,
    year         INTEGER NOT NULL,
    month        INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    total_value  REAL NOT NULL,
    UNIQUE(supplier_id, category, year, month)
);

CREATE TABLE IF NOT EXISTS recommended_orders (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT NOT NULL,
    product_id           INTEGER NOT NULL REFERENCES products(product_id),
    supplier_id          INTEGER NOT NULL REFERENCES suppliers(supplier_id),
    base_demand          REAL,
    seasonality_adj      REAL,
    stockout_adj         REAL,
    outlier_excluded_qty REAL DEFAULT 0,
    current_stock        REAL,
    in_transit_qty       REAL,
    recommended_qty      REAL NOT NULL,
    urgency              TEXT CHECK (urgency IN ('низкая','средняя','высокая','критическая')),
    justification        TEXT NOT NULL,
    status               TEXT DEFAULT 'draft' CHECK (status IN ('draft','approved','rejected','sent')),
    created_at           TEXT DEFAULT (datetime('now')),
    approved_by          TEXT,
    approved_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_reco_run ON recommended_orders(run_id);
CREATE INDEX IF NOT EXISTS idx_reco_supplier ON recommended_orders(supplier_id);
"""


def hash_client_id(raw_client_id: str) -> str:
    """Обезличивание клиента: в БД попадает только sha256, никогда исходный ID."""
    return hashlib.sha256(raw_client_id.encode("utf-8")).hexdigest()


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Добавляет недостающие колонки в существующую БД (без поломки данных)."""
    product_cols = {row[1] for row in conn.execute("PRAGMA table_info(products)")}
    if "unit_cost" not in product_cols:
        conn.execute(
            "ALTER TABLE products ADD COLUMN unit_cost REAL DEFAULT 0"
        )
    conn.commit()


def init_db(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Создаёт файл БД и все таблицы по схеме."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(path)
    conn.executescript(SCHEMA_SQL)
    migrate_schema(conn)
    conn.commit()
    return conn


def _month_iter(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


def _seasonal_factor(month: int) -> float:
    """Условная сезонность электротехники: спад зимой, пик весна/осень."""
    factors = {
        1: 0.75,
        2: 0.80,
        3: 1.05,
        4: 1.15,
        5: 1.20,
        6: 1.00,
        7: 0.90,
        8: 0.95,
        9: 1.25,
        10: 1.30,
        11: 1.10,
        12: 0.85,
    }
    return factors[month]


def seed_test_data(
    conn: sqlite3.Connection | None = None,
    *,
    db_path: Path | str = DB_PATH,
    clear: bool = True,
    seed: int = 42,
) -> dict:
    """
    Заполняет таблицы реалистичными тестовыми данными.

    - Поставщики: ИЭК, Systeme Electric
    - Товары обеих марок с MOQ / кратностью
    - 24 месяца продаж и остатков (2024-01 … 2025-12)
    - Несколько аномально крупных отгрузок одному клиенту (is_outlier=0 —
      флаг ставит алгоритм detect_outliers, не сидер)
    - Периоды stockout, товар в пути, сезонность
    """
    rng = random.Random(seed)
    own_conn = conn is None
    if own_conn:
        conn = init_db(db_path)

    if clear:
        tables = [
            "recommended_orders",
            "seasonality_reference",
            "transit_orders",
            "stockout_periods",
            "monthly_sales",
            "monthly_stock",
            "sales_transactions",
            "products",
            "suppliers",
        ]
        for table in tables:
            conn.execute(f"DELETE FROM {table}")
        conn.commit()

    # --- Поставщики ---
    suppliers = [
        ("ИЭК", 21),
        ("Systeme Electric", 28),
    ]
    supplier_ids: dict[str, int] = {}
    for name, lead in suppliers:
        cur = conn.execute(
            "INSERT INTO suppliers (supplier_name, lead_time_days) VALUES (?, ?)",
            (name, lead),
        )
        supplier_ids[name] = cur.lastrowid

    iek_id = supplier_ids["ИЭК"]
    se_id = supplier_ids["Systeme Electric"]

    # --- Товары ---
    # (code_1c, article, name, unit, category, supplier_id, min_ship, multiplicity, base_monthly, unit_cost_kzt)
    product_defs = [
        # ИЭК
        ("IEK-0001", "MVA20-1-016-C", "Автоматический выключатель ВА47-29 1P 16А C", "шт", "Автоматика", iek_id, 12, 1, 180, 1850),
        ("IEK-0002", "MVA20-1-025-C", "Автоматический выключатель ВА47-29 1P 25А C", "шт", "Автоматика", iek_id, 12, 1, 140, 2100),
        ("IEK-0003", "MVA20-3-032-C", "Автоматический выключатель ВА47-29 3P 32А C", "шт", "Автоматика", iek_id, 6, 1, 90, 5200),
        ("IEK-0004", "MDV15-2-063-030", "УЗО ВД1-63 2P 63А 30мА", "шт", "УЗО", iek_id, 4, 1, 55, 9800),
        ("IEK-0005", "UKU10-V1-K01", "Корпус ЩРн-П 12 модулей IP41", "шт", "Щиты", iek_id, 1, 1, 35, 4200),
        ("IEK-0006", "UKA10-40-K03", "Корпус ЩРн 36 модулей IP31", "шт", "Щиты", iek_id, 1, 1, 22, 8900),
        ("IEK-0007", "UKP10-3-K01", "Клемма винтовая ЗНИ 2.5 мм²", "шт", "Клеммы", iek_id, 100, 1, 800, 95),
        ("IEK-0008", "YND10-00-K02", "Наконечник НШвИ 1.5-8", "шт", "Наконечники", iek_id, 100, 1, 1200, 35),
        # Systeme Electric
        ("SE-0001", "A9F74116", "Автомат iC60N 1P 16A C", "шт", "Автоматика", se_id, 1, 12, 95, 6500),
        ("SE-0002", "A9F74325", "Автомат iC60N 3P 25A C", "шт", "Автоматика", se_id, 1, 6, 48, 18500),
        ("SE-0003", "A9R41263", "УЗО iID 2P 63A 30mA AC", "шт", "УЗО", se_id, 1, 4, 30, 27500),
        ("SE-0004", "A9C20832", "Контактор iCT 25A 2NO 230V", "шт", "Контакторы", se_id, 1, 1, 40, 15200),
        ("SE-0005", "GV2ME08", "Пускатель TeSys GV2ME 2.5-4A", "шт", "Пускатели", se_id, 1, 1, 18, 24800),
        ("SE-0006", "NSX100N", "Автомат Compact NSX100N 3P 100A", "шт", "Силовая защита", se_id, 1, 1, 8, 125000),
        ("SE-0007", "LRE12", "Реле тепловое TeSys LRE 5.5-8A", "шт", "Реле", se_id, 1, 1, 25, 11200),
        ("SE-0008", "ZB5AA3", "Кнопка XB5 зелёная утопленная", "шт", "Кнопки", se_id, 1, 10, 60, 3200),
    ]

    product_meta: list[dict] = []
    for code, article, name, unit, category, sid, min_ship, mult, base, cost in product_defs:
        cur = conn.execute(
            """
            INSERT INTO products (
                code_1c, supplier_article, name, unit, category,
                supplier_id, min_ship_qty, multiplicity, is_active, unit_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (code, article, name, unit, category, sid, min_ship, mult, cost),
        )
        product_meta.append(
            {
                "product_id": cur.lastrowid,
                "code_1c": code,
                "unit": unit,
                "category": category,
                "supplier_id": sid,
                "base_monthly": base,
                "unit_cost": cost,
            }
        )

    warehouse = "Основной склад"
    history_start = date(2024, 1, 1)
    history_end = date(2025, 12, 1)

    # Обезличенные клиенты (в БД только хэш)
    regular_clients = [hash_client_id(f"client_regular_{i:03d}") for i in range(1, 16)]
    # Один клиент — источник аномально крупных разовых заказов
    outlier_client = hash_client_id("client_project_mega_001")

    monthly_sales_rows: list[tuple] = []
    monthly_stock_rows: list[tuple] = []
    sales_rows: list[tuple] = []
    doc_counter = 1000

    # Товары, по которым симулируем stockout (низкий остаток / ноль)
    stockout_product_codes = {"IEK-0004", "SE-0006"}

    for meta in product_meta:
        pid = meta["product_id"]
        base = meta["base_monthly"]
        unit = meta["unit"]
        stock = float(base * 1.1 + rng.randint(5, 40))
        trend = 1.0  # устойчивый рост ~1.2%/мес

        for year, month in _month_iter(history_start, history_end):
            trend *= 1.012
            seasonal = _seasonal_factor(month)
            noise = rng.uniform(0.85, 1.15)
            qty_sold = max(1.0, round(base * seasonal * trend * noise, 1))

            # Stockout: в мае–июне 2025 для выбранных SKU остаток = 0, продажи занижены
            is_stockout_month = (
                meta["code_1c"] in stockout_product_codes
                and year == 2025
                and month in (5, 6)
            )
            if is_stockout_month:
                stock_qty = 0.0
                qty_sold = round(qty_sold * 0.15, 1)  # упущенный спрос
            else:
                # Остаток на начало месяца до продаж
                stock_qty = max(0.0, round(stock, 1))
                # Пополнение слабее расхода → к концу истории остатки умеренные
                stock = stock - qty_sold + qty_sold * rng.uniform(0.55, 0.95)
                if stock < base * 0.15:
                    stock = base * rng.uniform(0.4, 0.9)
                # В конце 2025 у части SKU — низкий остаток (для демо дефицита)
                if year == 2025 and month >= 11:
                    stock = min(stock, base * rng.uniform(0.2, 0.7))

            monthly_sales_rows.append((pid, year, month, qty_sold))
            monthly_stock_rows.append((pid, warehouse, year, month, stock_qty))

            # Разбиваем месячные продажи на 4–8 транзакций (отрицательное qty = отгрузка)
            n_tx = rng.randint(4, 8)
            weights = [rng.random() for _ in range(n_tx)]
            wsum = sum(weights)
            day_base = date(year, month, 1)
            days_in_month = (
                date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
            ).day

            for i, w in enumerate(weights):
                part = max(1.0, round(qty_sold * (w / wsum), 1))
                day = min(days_in_month, 2 + i * (days_in_month // n_tx))
                sale_dt = datetime(year, month, day, rng.randint(9, 17), rng.randint(0, 59), 0)
                doc_counter += 1
                client = rng.choice(regular_clients)
                sales_rows.append(
                    (
                        sale_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        f"РН-{doc_counter}",
                        "Расходная накладная",
                        pid,
                        unit,
                        warehouse,
                        -part,  # отгрузка клиенту
                        client,
                        0,  # is_outlier — ставит алгоритм, не сидер
                        None,
                    )
                )

    # --- Аномально крупные разовые заказы одного клиента ---
    # Должны заметно отличаться от regular demand (x8–x15 от типичной отгрузки)
    outlier_specs = [
        # (code_1c, date, multiplier_of_base_monthly, reason_comment — только для логов сидера)
        ("IEK-0001", date(2025, 3, 14), 12.0),
        ("IEK-0003", date(2025, 4, 22), 10.0),
        ("IEK-0007", date(2025, 7, 8), 15.0),
        ("SE-0001", date(2025, 2, 18), 11.0),
        ("SE-0004", date(2025, 9, 5), 9.0),
        ("SE-0008", date(2025, 10, 11), 14.0),
    ]
    code_to_meta = {m["code_1c"]: m for m in product_meta}
    outlier_inserted = 0
    for code, sale_day, mult in outlier_specs:
        meta = code_to_meta[code]
        qty = -round(meta["base_monthly"] * mult, 1)
        doc_counter += 1
        sale_dt = datetime(sale_day.year, sale_day.month, sale_day.day, 11, 30, 0)
        sales_rows.append(
            (
                sale_dt.strftime("%Y-%m-%d %H:%M:%S"),
                f"РН-{doc_counter}",
                "Расходная накладная",
                meta["product_id"],
                meta["unit"],
                warehouse,
                qty,
                outlier_client,
                0,
                None,
            )
        )
        # Корректируем monthly_sales, чтобы агрегат совпадал с транзакциями
        y, m = sale_day.year, sale_day.month
        for i, (pid, yy, mm, q) in enumerate(monthly_sales_rows):
            if pid == meta["product_id"] and yy == y and mm == m:
                monthly_sales_rows[i] = (pid, yy, mm, round(q + abs(qty), 1))
                break
        outlier_inserted += 1

    conn.executemany(
        """
        INSERT INTO monthly_sales (product_id, year, month, qty_sold)
        VALUES (?, ?, ?, ?)
        """,
        monthly_sales_rows,
    )
    conn.executemany(
        """
        INSERT INTO monthly_stock (product_id, warehouse, year, month, stock_qty)
        VALUES (?, ?, ?, ?, ?)
        """,
        monthly_stock_rows,
    )
    conn.executemany(
        """
        INSERT INTO sales_transactions (
            sale_date, doc_number, doc_name, product_id, unit, warehouse,
            quantity, client_hash, is_outlier, outlier_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        sales_rows,
    )

    # --- Stockout periods (как будто уже детектированы алгоритмом) ---
    for code in stockout_product_codes:
        meta = code_to_meta[code]
        conn.execute(
            """
            INSERT INTO stockout_periods (
                product_id, warehouse, start_date, end_date, days_out, detection_method
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                meta["product_id"],
                warehouse,
                "2025-05-01",
                "2025-06-30",
                61,
                "stock=0 for N days",
            ),
        )

    # --- Товар в пути ---
    transit_specs = [
        ("IEK-0001", "УТ-8201", "2025-11-20", "поступление до 15.12.2025", 240),
        ("IEK-0004", "УТ-8215", "2025-11-28", "поступление до 20.12.2025", 80),
        ("SE-0001", "УТ-8231", "2025-12-01", "поступление до 28.12.2025", 120),
        ("SE-0006", "УТ-8240", "2025-11-10", "поступление до 10.12.2025", 12),
        ("IEK-0007", "УТ-8255", "2025-12-05", "поступление до 05.01.2026", 2000),
    ]
    for code, po, po_date, arrival, qty in transit_specs:
        meta = code_to_meta[code]
        conn.execute(
            """
            INSERT INTO transit_orders (
                product_id, po_number, po_date, expected_arrival, quantity
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (meta["product_id"], po, po_date, arrival, qty),
        )

    # --- Сезонность (уровень поставщика + несколько категорий) ---
    for sid in (iek_id, se_id):
        for year in (2024, 2025):
            yearly_base = 2_500_000 if sid == iek_id else 1_800_000
            if year == 2025:
                yearly_base *= 1.12
            month_values = []
            for month in range(1, 13):
                val = yearly_base / 12 * _seasonal_factor(month) * rng.uniform(0.97, 1.03)
                month_values.append(val)
                conn.execute(
                    """
                    INSERT INTO seasonality_reference (
                        supplier_id, category, year, month, total_value
                    ) VALUES (?, NULL, ?, ?, ?)
                    """,
                    (sid, year, month, round(val, 2)),
                )
            # Категория «Автоматика»
            for month in range(1, 13):
                val = month_values[month - 1] * 0.45
                conn.execute(
                    """
                    INSERT INTO seasonality_reference (
                        supplier_id, category, year, month, total_value
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (sid, "Автоматика", year, month, round(val, 2)),
                )

    # --- Пример recommended_orders (draft) для демо UI ---
    run_id = "demo-run-2025-12"
    demo_reco = [
        (
            "IEK-0001",
            iek_id,
            210.0,
            1.25,
            0.0,
            2160.0,
            95.0,
            240.0,
            180.0,
            "средняя",
            (
                "База 210 шт/мес (без выбросов); сезонность сент/окт ×1.25; "
                "исключён разовый заказ ~2160 шт (client_hash=…); "
                "остаток 95 шт; в пути 240 шт (УТ-8201)."
            ),
        ),
        (
            "SE-0006",
            se_id,
            9.0,
            1.10,
            6.0,
            0.0,
            2.0,
            12.0,
            8.0,
            "высокая",
            (
                "База 9 шт/мес; сезонность ×1.10; "
                "компенсация stockout май–июнь 2025 (+6 шт, 61 день); "
                "остаток 2 шт; в пути 12 шт (УТ-8240)."
            ),
        ),
    ]
    for code, sid, base_d, seas, stock_adj, out_ex, cur_st, transit, reco, urg, just in demo_reco:
        meta = code_to_meta[code]
        conn.execute(
            """
            INSERT INTO recommended_orders (
                run_id, product_id, supplier_id, base_demand, seasonality_adj,
                stockout_adj, outlier_excluded_qty, current_stock, in_transit_qty,
                recommended_qty, urgency, justification, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft')
            """,
            (
                run_id,
                meta["product_id"],
                sid,
                base_d,
                seas,
                stock_adj,
                out_ex,
                cur_st,
                transit,
                reco,
                urg,
                just,
            ),
        )

    conn.commit()

    stats = {
        "suppliers": len(supplier_ids),
        "products": len(product_meta),
        "monthly_sales": len(monthly_sales_rows),
        "monthly_stock": len(monthly_stock_rows),
        "sales_transactions": len(sales_rows),
        "outlier_candidate_tx": outlier_inserted,
        "outlier_client_hash": outlier_client,
        "db_path": str(Path(db_path).resolve()),
    }

    if own_conn:
        conn.close()

    return stats


def apply_moq_rounding(
    recommended_qty: float, min_ship_qty: int, multiplicity: int
) -> tuple[float, str]:
    """
    Проверяет и при необходимости округляет количество под условия поставщика:
    минимальная партия отгрузки (min_ship_qty) и кратность (multiplicity).
    Возвращает (итоговое_количество, человекочитаемое_пояснение).
    """
    qty = max(float(recommended_qty or 0), 0.0)
    multiplicity = max(int(multiplicity or 1), 1)
    min_ship_qty = max(int(min_ship_qty or 1), 1)

    if qty <= 0:
        return 0.0, "заказ не требуется"

    rounded = ((qty + multiplicity - 1) // multiplicity) * multiplicity
    rounded = max(rounded, min_ship_qty)

    if rounded != qty:
        return (
            float(rounded),
            f"округлено с {qty:.0f} до {rounded:.0f} "
            f"(мин. партия {min_ship_qty}, кратность {multiplicity})",
        )
    return float(rounded), "соответствует условиям поставщика"


# ---------------------------------------------------------------------------
# Read-only SQL для ИИ-ассистента (только SELECT)
# ---------------------------------------------------------------------------

_FORBIDDEN_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|TRUNCATE|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

AGENT_SCHEMA_SUMMARY = """
Таблицы SQLite (storage.db):
- suppliers(supplier_id, supplier_name, lead_time_days, created_at)
- products(product_id, code_1c, supplier_article, name, unit, category,
  supplier_id, min_ship_qty, multiplicity, is_active, unit_cost, created_at)
- sales_transactions(transaction_id, sale_date, doc_number, doc_name, product_id,
  unit, warehouse, quantity, client_hash, is_outlier, outlier_reason, imported_at)
  quantity < 0 = отгрузка клиенту; client_hash — обезличенный sha256
- monthly_stock(id, product_id, warehouse, year, month, stock_qty)
- monthly_sales(id, product_id, year, month, qty_sold)
- stockout_periods(id, product_id, warehouse, start_date, end_date, days_out,
  detection_method, created_at)
- transit_orders(id, product_id, po_number, po_date, expected_arrival, quantity,
  created_at)
- seasonality_reference(id, supplier_id, category, year, month, total_value)
- recommended_orders(id, run_id, product_id, supplier_id, base_demand,
  seasonality_adj, stockout_adj, outlier_excluded_qty, current_stock,
  in_transit_qty, recommended_qty, urgency, justification, status,
  created_at, approved_by, approved_at)
  urgency IN ('низкая','средняя','высокая','критическая');
  status IN ('draft','approved','rejected','sent')
""".strip()


def is_safe_select(sql: str) -> tuple[bool, str]:
    """Проверяет, что SQL — одиночный безопасный SELECT (без записи/DDL)."""
    if not sql or not str(sql).strip():
        return False, "пустой SQL"

    cleaned = str(sql).strip().rstrip(";").strip()
    if ";" in cleaned:
        return False, "разрешён только один statement"

    upper = cleaned.upper().lstrip()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "разрешены только SELECT / WITH … SELECT"

    if _FORBIDDEN_SQL_RE.search(cleaned):
        return False, "обнаружены запрещённые ключевые слова (только чтение)"

    return True, "ok"


def execute_readonly_query(
    sql: str,
    *,
    db_path: Path | str = DB_PATH,
    limit: int = 200,
):
    """
    Выполняет только безопасный SELECT и возвращает DataFrame (до `limit` строк).
    Бросает ValueError, если запрос не read-only.
    """
    import pandas as pd

    ok, reason = is_safe_select(sql)
    if not ok:
        raise ValueError(f"Небезопасный SQL: {reason}")

    cleaned = str(sql).strip().rstrip(";").strip()
    if "LIMIT" not in cleaned.upper():
        cleaned = f"{cleaned}\nLIMIT {int(limit)}"

    path = Path(db_path)
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return pd.read_sql_query(cleaned, conn)
    finally:
        conn.close()


def main() -> None:
    print(f"Инициализация БД: {DB_PATH}")
    conn = init_db(DB_PATH)
    stats = seed_test_data(conn, clear=True)
    conn.close()

    print("Готово. Загружено:")
    for key, value in stats.items():
        if key == "outlier_client_hash":
            print(f"  {key}: {value[:16]}…")
        else:
            print(f"  {key}: {value}")
    print(
        "\nАномально крупные заказы одного клиента добавлены с is_outlier=0 — "
        "их должен пометить detect_outliers()."
    )


if __name__ == "__main__":
    main()
