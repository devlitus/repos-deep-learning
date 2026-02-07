"""
Pipeline Principal - Sistema de Recomendación de Moda (Amazon Fashion Reviews)
Ejecuta el pipeline completo: carga, EDA, CF, SVD, Híbrido
"""

import sys
import io
import time
from pathlib import Path

# Configurar UTF-8 para Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# Asegurar que el directorio raíz del proyecto esté en el path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'src'))


def print_banner():
    """Imprime el banner del proyecto"""
    print("\n" + "=" * 80)
    print("  👕 SISTEMA DE RECOMENDACIÓN DE MODA")
    print("  Amazon Fashion Reviews - Pipeline Completo")
    print("=" * 80)


def print_section(number, title):
    """Imprime un encabezado de sección"""
    print(f"\n\n{'█' * 80}")
    print(f"  PASO {number}: {title}")
    print(f"{'█' * 80}")


def print_error(module, error):
    """Imprime un error de módulo"""
    print(f"\n  ❌ Error en {module}: {error}")
    print(f"  ⚠️  Continuando con los demás módulos...\n")


def main():
    """Ejecuta el pipeline completo del proyecto"""
    start_time = time.time()
    print_banner()

    # =========================================================================
    # PASO 1: VALIDAR CONFIGURACIÓN
    # =========================================================================
    print_section(1, "VALIDAR CONFIGURACIÓN")

    from config import validate_config, print_config_summary

    success, issues, warnings_list = validate_config()

    if issues:
        print("\n  ❌ Problemas encontrados:")
        for issue in issues:
            print(f"    {issue}")

    if warnings_list:
        print("\n  ⚠️  Advertencias:")
        for warning in warnings_list:
            print(f"    {warning}")

    if success:
        print("\n  ✅ Configuración válida")
        print_config_summary()
    else:
        print("\n  ❌ Configuración inválida. Corrige los errores antes de continuar.")
        return

    # =========================================================================
    # PASO 2: CARGAR DATOS
    # =========================================================================
    print_section(2, "CARGAR DATOS")

    try:
        from src.data_loader import load_data
        df = load_data()
    except Exception as e:
        print_error("Carga de datos", e)
        return  # Sin datos no podemos continuar

    # =========================================================================
    # PASO 3: EXPLORAR DATOS
    # =========================================================================
    print_section(3, "EXPLORAR DATOS")

    try:
        from src.data_loader import explore_data
        explore_data(df)
    except Exception as e:
        print_error("Exploración de datos", e)

    # =========================================================================
    # PASO 4: PREPARAR DATOS
    # =========================================================================
    print_section(4, "PREPARAR DATOS")

    try:
        from src.data_loader import prepare_data
        df_clean = prepare_data(df)
    except Exception as e:
        print_error("Preparación de datos", e)
        return  # Sin datos limpios no podemos continuar

    # =========================================================================
    # PASO 5: CREAR MATRIZ USUARIO-PRODUCTO
    # =========================================================================
    print_section(5, "CREAR MATRIZ USUARIO-PRODUCTO")

    try:
        from src.data_loader import get_user_item_matrix
        rating_matrix = get_user_item_matrix(df_clean)
    except Exception as e:
        print_error("Creación de matriz", e)
        return  # Sin matriz no podemos continuar

    # =========================================================================
    # PASO 6: ANÁLISIS EXPLORATORIO (EDA)
    # =========================================================================
    print_section(6, "ANÁLISIS EXPLORATORIO (EDA)")

    try:
        from src.exploratory_analysis import run_full_eda
        eda_results = run_full_eda(df_clean)
    except Exception as e:
        print_error("Análisis exploratorio", e)
        eda_results = None

    # =========================================================================
    # PASO 7: ANÁLISIS DE DISPERSIÓN (SPARSITY)
    # =========================================================================
    print_section(7, "ANÁLISIS DE DISPERSIÓN (SPARSITY)")

    try:
        from src.sparsity_analysis import run_sparsity_analysis
        sparsity_results = run_sparsity_analysis(df_clean, rating_matrix)
    except Exception as e:
        print_error("Análisis de sparsity", e)
        sparsity_results = None

    # =========================================================================
    # PASO 8: USER-BASED COLLABORATIVE FILTERING
    # =========================================================================
    print_section(8, "USER-BASED COLLABORATIVE FILTERING")

    user_sim = None
    user_cf_eval = None
    try:
        from src.user_based_collaborative_filtering import run_user_based_cf
        ub_results = run_user_based_cf(df_clean, rating_matrix)
        user_sim = ub_results['similarity_df']
        user_cf_eval = ub_results['evaluation']
    except Exception as e:
        print_error("User-Based CF", e)

    # =========================================================================
    # PASO 9: ITEM-BASED COLLABORATIVE FILTERING
    # =========================================================================
    print_section(9, "ITEM-BASED COLLABORATIVE FILTERING")

    product_sim = None
    item_cf_eval = None
    try:
        from src.item_based_collaborative_filtering import run_item_based_cf
        ib_results = run_item_based_cf(df_clean, rating_matrix)
        product_sim = ib_results['similarity_df']
        item_cf_eval = ib_results['evaluation']
    except Exception as e:
        print_error("Item-Based CF", e)

    # =========================================================================
    # PASO 10: MATRIX FACTORIZATION (SVD)
    # =========================================================================
    print_section(10, "MATRIX FACTORIZATION (SVD)")

    U = sigma = Vt = None
    svd_eval = None
    try:
        from src.matrix_factorization_svd import run_svd_analysis
        svd_results = run_svd_analysis(df_clean, rating_matrix)
        U = svd_results['U']
        sigma = svd_results['sigma']
        Vt = svd_results['Vt']
        svd_eval = svd_results['evaluation']
    except Exception as e:
        print_error("SVD", e)

    # =========================================================================
    # PASO 11: SISTEMA HÍBRIDO
    # =========================================================================
    print_section(11, "SISTEMA HÍBRIDO DE RECOMENDACIÓN")

    hybrid_eval = None
    if user_sim is not None and product_sim is not None and U is not None:
        try:
            from src.hybrid_recommender_system import run_hybrid_system, compare_all_methods, visualize_hybrid
            hybrid_results = run_hybrid_system(
                df_clean, rating_matrix, user_sim, product_sim, U, sigma, Vt
            )
            hybrid_eval = hybrid_results['evaluation']
        except Exception as e:
            print_error("Sistema Híbrido", e)
    else:
        print("\n  ⚠️  Sistema Híbrido omitido: faltan componentes previos")
        print("    Requiere: User-Based CF + Item-Based CF + SVD")

    # =========================================================================
    # PASO 12: DEEP HYBRID RECOMMENDER (Deep Learning)
    # =========================================================================
    print_section(12, "DEEP HYBRID RECOMMENDER (Deep Learning)")

    deep_hybrid_eval = None
    deep_hybrid_results = None

    # Verificar que PyTorch esté disponible
    try:
        import torch
        pytorch_available = True
    except ImportError:
        pytorch_available = False
        print("\n  ⚠️  PyTorch no está instalado. Instala con: pip install torch")

    # Verificar que surprise esté disponible para modelos tradicionales
    try:
        from surprise import KNNBasic, SVD
        surprise_available = True
    except ImportError:
        surprise_available = False
        print("\n  ⚠️  Surprise no está instalado. Instala con: pip install scikit-surprise")

    if pytorch_available and surprise_available:
        # Verificar que tengamos los modelos base necesarios
        if user_sim is not None and product_sim is not None and U is not None:
            try:
                print("\n  🔄 Entrenando modelos base con Surprise para Deep Hybrid...")

                from surprise import Dataset, Reader
                from surprise.model_selection import train_test_split as surprise_split

                # Preparar datos para Surprise
                reader = Reader(rating_scale=(1, 5))
                surprise_data = Dataset.load_from_df(
                    df_clean[['userId', 'productId', 'rating']],
                    reader
                )
                trainset = surprise_data.build_full_trainset()

                # Entrenar modelos base con Surprise
                print("    Entrenando User-Based CF...")
                from surprise import KNNBasic
                user_cf_surprise = KNNBasic(
                    sim_options={'name': 'cosine', 'user_based': True},
                    k=20
                )
                user_cf_surprise.fit(trainset)

                print("    Entrenando Item-Based CF...")
                item_cf_surprise = KNNBasic(
                    sim_options={'name': 'cosine', 'user_based': False},
                    k=20
                )
                item_cf_surprise.fit(trainset)

                print("    Entrenando SVD...")
                from surprise import SVD as SurpriseSVD
                svd_surprise = SurpriseSVD(
                    n_factors=50,
                    n_epochs=20,
                    lr_all=0.005,
                    reg_all=0.02
                )
                svd_surprise.fit(trainset)

                # Ejecutar Deep Hybrid
                from src.deep_hybrid_recommender import run_deep_hybrid_system, visualize_deep_hybrid_results

                deep_hybrid_results = run_deep_hybrid_system(
                    df_clean=df_clean,
                    user_cf_model=user_cf_surprise,
                    item_cf_model=item_cf_surprise,
                    svd_model=svd_surprise
                )

                deep_hybrid_eval = deep_hybrid_results['evaluation']

                print("\n  ✅ Deep Hybrid System entrenado exitosamente")
                print(f"    RMSE: {deep_hybrid_eval['RMSE']:.4f}")
                print(f"    MAE: {deep_hybrid_eval['MAE']:.4f}")

            except Exception as e:
                print_error("Deep Hybrid System", e)
                import traceback
                traceback.print_exc()
        else:
            print("\n  ⚠️  Deep Hybrid omitido: faltan componentes previos")
            print("    Requiere: User-Based CF + Item-Based CF + SVD")
    else:
        print("\n  ⚠️  Deep Hybrid omitido: dependencias faltantes")
        if not pytorch_available:
            print("    - Instala PyTorch: pip install torch")
        if not surprise_available:
            print("    - Instala Surprise: pip install scikit-surprise")

    # =========================================================================
    # PASO 13: COMPARACIÓN FINAL DE TODOS LOS MÉTODOS
    # =========================================================================
    print_section(13, "COMPARACIÓN FINAL DE TODOS LOS MÉTODOS")

    # Recopilar resultados
    all_results = {}
    if user_cf_eval and user_cf_eval.get('RMSE') is not None:
        all_results['User-Based CF'] = user_cf_eval
    if item_cf_eval and item_cf_eval.get('RMSE') is not None:
        all_results['Item-Based CF'] = item_cf_eval
    if svd_eval and svd_eval.get('RMSE') is not None:
        all_results['SVD'] = svd_eval
    if hybrid_eval and hybrid_eval.get('RMSE') is not None:
        all_results['Hybrid'] = hybrid_eval
    if deep_hybrid_eval and deep_hybrid_eval.get('RMSE') is not None:
        all_results['Deep Hybrid'] = deep_hybrid_eval

    if all_results:
        try:
            from src.hybrid_recommender_system import compare_all_methods, visualize_hybrid
            comparison = compare_all_methods(all_results)

            # Generar visualización comparativa
            if hybrid_eval:
                visualize_hybrid(hybrid_eval, comparison)

            # Visualizar resultados de Deep Hybrid
            if deep_hybrid_results:
                from src.deep_hybrid_recommender import visualize_deep_hybrid_results
                visualize_deep_hybrid_results(deep_hybrid_results, comparison)
        except Exception as e:
            print_error("Comparación final", e)
    else:
        print("\n  ⚠️  No hay métricas disponibles para comparar")

    # =========================================================================
    # PASO 14: RESUMEN Y CONCLUSIONES
    # =========================================================================
    print_section(14, "RESUMEN Y CONCLUSIONES")

    elapsed_time = time.time() - start_time

    print(f"\n  {'=' * 60}")
    print(f"  📊 RESUMEN DEL PIPELINE")
    print(f"  {'=' * 60}")

    print(f"\n  📁 Dataset:")
    print(f"    Reviews originales: {len(df):,}")
    print(f"    Reviews procesadas: {len(df_clean):,}")
    print(f"    Usuarios: {df_clean['user_idx'].nunique():,}")
    print(f"    Productos: {df_clean['product_idx'].nunique():,}")

    if all_results:
        print(f"\n  📈 Métricas de los Modelos:")
        for name, metrics in all_results.items():
            print(f"    {name}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

    # Reportes generados
    from config import REPORTS_DIR
    reports = list(REPORTS_DIR.glob('*.png'))
    print(f"\n  📄 Reportes generados: {len(reports)} archivos en reports/")
    for report in sorted(reports):
        print(f"    - {report.name}")

    print(f"\n  ⏱️  Tiempo total: {elapsed_time:.1f} segundos ({elapsed_time/60:.1f} minutos)")

    print(f"\n  {'=' * 60}")
    print(f"  ✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"  {'=' * 60}")

    print(f"\n  📌 Próximos pasos:")
    print(f"    1. Revisar gráficos en reports/")
    print(f"    2. Ejecutar la app web: streamlit run web/app.py")
    print(f"    3. Explorar notebooks en notebooks/")
    print(f"\n  {'=' * 60}\n")


if __name__ == '__main__':
    main()
