"""Módulo de páginas de la aplicación"""
from .dashboard import show_dashboard
from .prediction import show_prediction
from .analytics import show_analytics
from .data_explorer import show_eda
from .about import show_about

__all__ = [
    'show_dashboard',
    'show_prediction',
    'show_analytics',
    'show_eda',
    'show_about'
]
