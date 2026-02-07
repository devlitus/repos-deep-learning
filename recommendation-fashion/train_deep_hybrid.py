"""
Script independiente para entrenar solo el Deep Hybrid Recommender
Útil para experimentación rápida sin ejecutar todo el pipeline
"""

import sys
import io
from pathlib import Path

# Configurar UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# Asegurar imports
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'src'))

import config
import pandas as pd
import numpy as np


def main():
    """Entrena solo el Deep Hybrid Recommender"""

    print("\n" + "=" * 80)
    print("  🧠 ENTRENAMIENTO RÁPIDO - DEEP HYBRID RECOMMENDER")
    print("=" * 80)

    # ========== 1. Verificar dependencias ==========
    print("\n📦 Verificando dependencias...")

    try:
        import torch
        print(f"  ✅ PyTorch instalado (versión {torch.__version__})")
        print(f"  ✅ CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  ❌ PyTorch no instalado")
        print("  Instala con: pip install torch")
        return

    try:
        from surprise import KNNBasic, SVD, Dataset, Reader
        print("  ✅ Surprise instalado")
    except ImportError:
        print("  ❌ Surprise no instalado")
        print("  Instala con: pip install scikit-surprise")
        return

    # ========== 2. Cargar datos ==========
    print("\n📂 Cargando datos...")

    if not config.DATASET_FILE.exists():
        print(f"  ❌ Dataset no encontrado: {config.DATASET_FILE}")
        print("  Ejecuta: python download_fashion.py")
        return

    from src.data_loader import load_data, prepare_data

    df = load_data()
    df_clean = prepare_data(df)

    print(f"  ✅ Datos cargados: {len(df_clean):,} interacciones")
    print(f"  ✅ Usuarios: {df_clean['user_idx'].nunique():,}")
    print(f"  ✅ Productos: {df_clean['product_idx'].nunique():,}")

    # ========== 3. Entrenar modelos base con Surprise ==========
    print("\n🔄 Entrenando modelos base (User-CF, Item-CF, SVD)...")

    # Preparar datos para Surprise
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(
        df_clean[['userId', 'productId', 'rating']],
        reader
    )
    trainset = surprise_data.build_full_trainset()

    # User-Based CF
    print("  → User-Based Collaborative Filtering...")
    user_cf = KNNBasic(
        sim_options={'name': 'cosine', 'user_based': True},
        k=config.USER_CF_K_NEIGHBORS
    )
    user_cf.fit(trainset)
    print("    ✅ Completado")

    # Item-Based CF
    print("  → Item-Based Collaborative Filtering...")
    item_cf = KNNBasic(
        sim_options={'name': 'cosine', 'user_based': False},
        k=config.ITEM_CF_K_NEIGHBORS
    )
    item_cf.fit(trainset)
    print("    ✅ Completado")

    # SVD
    print("  → Matrix Factorization (SVD)...")
    svd = SVD(
        n_factors=config.SVD_K_FACTORS,
        n_epochs=20,
        lr_all=0.005,
        reg_all=config.SVD_REGULARIZATION
    )
    svd.fit(trainset)
    print("    ✅ Completado")

    # ========== 4. Entrenar Deep Hybrid ==========
    print("\n🧠 Entrenando Deep Hybrid Recommender...")

    from src.deep_hybrid_recommender import run_deep_hybrid_system, visualize_deep_hybrid_results

    results = run_deep_hybrid_system(
        df_clean=df_clean,
        user_cf_model=user_cf,
        item_cf_model=item_cf,
        svd_model=svd
    )

    # ========== 5. Mostrar resultados ==========
    print("\n" + "=" * 80)
    print("  📊 RESULTADOS FINALES")
    print("=" * 80)

    eval_metrics = results['evaluation']

    print(f"\n  📈 Métricas de Evaluación:")
    print(f"    RMSE: {eval_metrics['RMSE']:.4f}")
    print(f"    MAE: {eval_metrics['MAE']:.4f}")

    print(f"\n  🎯 Pesos Aprendidos por Atención:")
    attention = eval_metrics['attention_weights']
    components = ['User-CF', 'Item-CF', 'SVD', 'NCF']
    for comp, weight in zip(components, attention):
        print(f"    {comp:12s}: {weight:.3f} ({weight*100:.1f}%)")

    print(f"\n  📁 Modelo guardado en:")
    print(f"    {config.MODELS_DIR / 'deep_hybrid_model.pth'}")

    # ========== 6. Generar visualizaciones ==========
    print(f"\n  📊 Generando visualizaciones...")

    visualize_deep_hybrid_results(results)

    print(f"\n  ✅ Gráficas guardadas en: reports/")

    # ========== 7. Comparación con pesos manuales ==========
    print(f"\n  🔍 Comparación con Híbrido Tradicional:")
    print(f"    Pesos manuales en config.py:")
    print(f"      User-CF: {config.HYBRID_WEIGHT_USER_CF:.3f}")
    print(f"      Item-CF: {config.HYBRID_WEIGHT_ITEM_CF:.3f}")
    print(f"      SVD: {config.HYBRID_WEIGHT_SVD:.3f}")
    print(f"      NCF: 0.000 (no incluido)")

    print(f"\n    Pesos aprendidos (Deep Hybrid):")
    for comp, weight in zip(components, attention):
        print(f"      {comp:12s}: {weight:.3f}")

    # ========== 8. Recomendaciones de uso ==========
    print("\n" + "=" * 80)
    print("  💡 PRÓXIMOS PASOS")
    print("=" * 80)

    print("\n  1️⃣  Revisar gráficas en reports/:")
    print("     - deep_hybrid_ncf_training.png (curvas de entrenamiento)")
    print("     - deep_hybrid_attention_weights.png (pesos aprendidos)")
    print("     - deep_hybrid_predictions.png (predicciones vs real)")

    print("\n  2️⃣  Hacer predicciones:")
    print("     python -c \"from src.deep_hybrid_recommender import *; ...\"")

    print("\n  3️⃣  Ejecutar pipeline completo:")
    print("     python main.py")

    print("\n  4️⃣  Explorar notebooks:")
    print("     jupyter notebook notebooks/deep_learning_recommendations.ipynb")

    print("\n" + "=" * 80)
    print("  ✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
