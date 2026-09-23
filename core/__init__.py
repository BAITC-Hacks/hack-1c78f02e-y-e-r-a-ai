"""Чистые функции расчёта потребности и детекции выбросов (без доступа к БД)."""

from core.outliers import detect_outliers
from core.demand import calculate_recommendations, build_justification
from core.pipeline import run_replenishment_calculation

__all__ = [
    "detect_outliers",
    "calculate_recommendations",
    "build_justification",
    "run_replenishment_calculation",
]
