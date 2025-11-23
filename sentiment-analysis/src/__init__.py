"""
Sentiment Analysis - Módulo Principal
======================================

Módulos disponibles:
- data_loader: Carga y exploración de datos
- text_preprocessing: Limpieza y preprocesamiento de texto
- feature_extraction: Conversión de texto a features numéricas
- model: Modelos clásicos de ML (Naive Bayes, SVM)
- deep_model: Modelos de Deep Learning (LSTM)
- visualizations: Visualizaciones para NLP
- predictor: Predicción de sentimientos
"""

__version__ = '1.0.0'
__author__ = 'ML Learning Project'

# Exportar módulos para permitir: from src import data_loader
from . import data_loader
from . import text_preprocessing
from . import feature_extraction
from . import model
from . import deep_model
from . import visualizations
from . import predictor

# Definir qué se exporta con "from src import *"
__all__ = [
    'data_loader',
    'text_preprocessing',
    'feature_extraction',
    'model',
    'deep_model',
    'visualizations',
    'predictor'
]
