"""
Módulo para visualización de datos del proyecto de detección de fraude
"""
import sys
from pathlib import Path

# Agregar la raíz del proyecto al PATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config.config import FIGURES_DIR, TARGET_COLUMN, AMOUNT_COLUMN, STYLE

# Configurar estilo
sns.set_style(STYLE)


def plot_class_distribution(df, save=True):
    """
    Gráfico de barras con la distribución de clases
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Si True, guarda la figura
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Conteo absoluto
    class_counts = df[TARGET_COLUMN].value_counts()
    ax[0].bar(['Legítima', 'Fraude'], class_counts.values, color=['#2ecc71', '#e74c3c'])
    ax[0].set_ylabel('Número de Transacciones', fontsize=12)
    ax[0].set_title('Distribución de Clases (Valores Absolutos)', fontsize=14, fontweight='bold')
    ax[0].set_yscale('log')  # Escala logarítmica para ver mejor
    
    # Añadir etiquetas con valores
    for i, v in enumerate(class_counts.values):
        ax[0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Porcentaje
    class_percentages = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    ax[1].bar(['Legítima', 'Fraude'], class_percentages.values, color=['#2ecc71', '#e74c3c'])
    ax[1].set_ylabel('Porcentaje (%)', fontsize=12)
    ax[1].set_title('Distribución de Clases (Porcentajes)', fontsize=14, fontweight='bold')
    
    # Añadir etiquetas con porcentajes
    for i, v in enumerate(class_percentages.values):
        ax[1].text(i, v, f'{v:.3f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(FIGURES_DIR / 'class_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {FIGURES_DIR / 'class_distribution.png'}")
    
    plt.show()


def plot_amount_distribution(df, save=True):
    """
    Distribución de montos por clase (Legítima vs Fraude) - MEJORADO
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Si True, guarda la figura
    """
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    
    # Separar por clase
    legitimate = df[df[TARGET_COLUMN] == 0][AMOUNT_COLUMN]
    fraud = df[df[TARGET_COLUMN] == 1][AMOUNT_COLUMN]
    
    # FILA 1: Histogramas separados
    # Transacciones Legítimas
    ax[0, 0].hist(legitimate, bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax[0, 0].set_xlabel('Monto ($)', fontsize=12)
    ax[0, 0].set_ylabel('Frecuencia', fontsize=12)
    ax[0, 0].set_title('Distribución de Montos - LEGÍTIMAS', fontsize=14, fontweight='bold')
    ax[0, 0].set_xlim([0, 500])
    ax[0, 0].grid(axis='y', alpha=0.3)
    
    # Transacciones Fraudulentas
    ax[0, 1].hist(fraud, bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax[0, 1].set_xlabel('Monto ($)', fontsize=12)
    ax[0, 1].set_ylabel('Frecuencia', fontsize=12)
    ax[0, 1].set_title('Distribución de Montos - FRAUDES', fontsize=14, fontweight='bold')
    ax[0, 1].set_xlim([0, 500])
    ax[0, 1].grid(axis='y', alpha=0.3)
    
    # FILA 2: Boxplots y Estadísticas
    # Boxplot comparativo
    data_to_plot = [legitimate, fraud]
    bp = ax[1, 0].boxplot(data_to_plot, labels=['Legítima', 'Fraude'], patch_artist=True,
                          boxprops=dict(facecolor='lightblue'),
                          medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax[1, 0].set_ylabel('Monto ($)', fontsize=12)
    ax[1, 0].set_title('Comparación de Montos (Boxplot)', fontsize=14, fontweight='bold')
    ax[1, 0].set_ylim([0, 1000])
    ax[1, 0].grid(axis='y', alpha=0.3)
    
    # Estadísticas comparativas
    stats_text = f"""
    ESTADÍSTICAS COMPARATIVAS:
    
    Legítimas:
    • Media: ${legitimate.mean():.2f}
    • Mediana: ${legitimate.median():.2f}
    • Desv. Est: ${legitimate.std():.2f}
    • Max: ${legitimate.max():.2f}
    
    Fraudes:
    • Media: ${fraud.mean():.2f}
    • Mediana: ${fraud.median():.2f}
    • Desv. Est: ${fraud.std():.2f}
    • Max: ${fraud.max():.2f}
    """
    ax[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                  verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(FIGURES_DIR / 'amount_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {FIGURES_DIR / 'amount_distribution.png'}")
    
    plt.show()


def plot_correlation_matrix(df, save=True):
    """
    Matriz de correlación de las features
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Si True, guarda la figura
    """
    # Seleccionar solo algunas features para visualizar (todas sería muy denso)
    features_to_plot = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V17', 'V14', 'V12', 'V10', TARGET_COLUMN]
    df_subset = df[features_to_plot]
    
    # Calcular correlación
    corr = df_subset.corr()
    
    # Crear heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlación (Features seleccionadas)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save:
        plt.savefig(FIGURES_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {FIGURES_DIR / 'correlation_matrix.png'}")
    
    plt.show()


def plot_time_distribution(df, save=True):
    """
    Distribución de transacciones a lo largo del tiempo - MEJORADO
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Si True, guarda la figura
    """
    fig, ax = plt.subplots(3, 1, figsize=(16, 12))
    
    # Convertir Time a horas
    df['Hours'] = df['Time'] / 3600
    
    # Separar por clase
    legitimate_time = df[df[TARGET_COLUMN] == 0]['Hours']
    fraud_time = df[df[TARGET_COLUMN] == 1]['Hours']
    
    # Gráfico 1: Distribución general
    ax[0].hist(df['Hours'], bins=48, color='#3498db', edgecolor='black', alpha=0.7)
    ax[0].set_xlabel('Tiempo (Horas)', fontsize=12)
    ax[0].set_ylabel('Número de Transacciones', fontsize=12)
    ax[0].set_title('Distribución Temporal - TODAS las Transacciones', fontsize=14, fontweight='bold')
    ax[0].grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Solo Legítimas
    ax[1].hist(legitimate_time, bins=48, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax[1].set_xlabel('Tiempo (Horas)', fontsize=12)
    ax[1].set_ylabel('Número de Transacciones', fontsize=12)
    ax[1].set_title('Distribución Temporal - LEGÍTIMAS', fontsize=14, fontweight='bold')
    ax[1].grid(axis='y', alpha=0.3)
    
    # Gráfico 3: Solo Fraudes
    ax[2].hist(fraud_time, bins=48, color='#e74c3c', edgecolor='black', alpha=0.8)
    ax[2].set_xlabel('Tiempo (Horas)', fontsize=12)
    ax[2].set_ylabel('Número de Transacciones', fontsize=12)
    ax[2].set_title('Distribución Temporal - FRAUDES', fontsize=14, fontweight='bold')
    ax[2].grid(axis='y', alpha=0.3)
    
    # Añadir líneas verticales para marcar días
    for a in ax:
        a.axvline(x=24, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Día 2')
        a.legend()
    
    plt.tight_layout()
    
    if save:
        plt.savefig(FIGURES_DIR / 'time_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {FIGURES_DIR / 'time_distribution.png'}")
    
    plt.show()


def generate_all_plots(df):
    """
    Genera todos los gráficos del análisis exploratorio
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n🎨 Generando visualizaciones...")
    print("="*60)
    
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_time_distribution(df)
    plot_correlation_matrix(df)
    
    print("\n✅ Todas las visualizaciones generadas correctamente")
    print(f"📁 Guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    # Prueba del módulo
    from src.data.load import load_data
    
    data = load_data()
    if data is not None:
        generate_all_plots(data)