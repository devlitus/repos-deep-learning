# src/visualizations.py
import matplotlib.pyplot as plt
import os
from config import REPORTS_DIR, FEATURES

def plot_feature_vs_target(df, target='precio'):
    """Crea gráficas de características vs precio"""
    print("\n=== GENERANDO VISUALIZACIONES ===")
    
    features_to_plot = [f for f in FEATURES if f in df.columns]
    n_features = len(features_to_plot)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, feature in enumerate(features_to_plot[:4]):
        axes[i].scatter(df[feature], df[target], color=colors[i], alpha=0.6)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)
        axes[i].set_title(f'{feature} vs {target}')
    
    plt.tight_layout()
    
    # Guardar figura
    filepath = os.path.join(REPORTS_DIR, 'feature_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada en: {filepath}")
    
    plt.show()

def plot_predictions_vs_actual(y_test, y_pred):
    """Gráfica de predicciones vs valores reales"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, s=100)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Predicción perfecta')
    plt.xlabel('Precio Real', fontsize=12)
    plt.ylabel('Precio Predicho', fontsize=12)
    plt.title('Predicciones vs Precios Reales', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    filepath = os.path.join(REPORTS_DIR, 'predictions_vs_actual.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada en: {filepath}")
    
    plt.show()