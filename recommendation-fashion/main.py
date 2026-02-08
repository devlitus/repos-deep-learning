"""
Pipeline Principal - Sistema de Recomendación de Moda (Amazon Fashion Reviews)
Ejecuta el pipeline completo: carga, EDA, análisis de sparsity, NCF (Deep Learning)
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

    from config import validate_config, print_config_summary, MODELS_DIR

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
    # PASO 8: ENTRENAMIENTO NCF (Deep Learning)
    # =========================================================================
    print_section(8, "NEURAL COLLABORATIVE FILTERING (Deep Learning)")

    ncf_eval = None

    try:
        import torch
        pytorch_available = True
    except ImportError:
        pytorch_available = False
        print("\n  ⚠️  PyTorch no está instalado. Instala con: pip install torch")

    if pytorch_available:
        print("\n  ℹ️  Para entrenar el modelo NCF, ejecuta:")
        print("    python train_ncf_only.py")
        print("\n  Este script entrena el modelo de Deep Learning de forma independiente.")

        # Verificar si ya existe un modelo entrenado
        ncf_model_path = MODELS_DIR / 'ncf_model.pth'
        ncf_metrics_path = REPORTS_DIR / 'ncf_metrics.json'

        if ncf_model_path.exists():
            print(f"\n  ✅ Modelo NCF encontrado: {ncf_model_path.name}")
            if ncf_metrics_path.exists():
                import json
                with open(str(ncf_metrics_path), 'r') as f:
                    ncf_eval = json.load(f)
                if 'test_rmse' in ncf_eval:
                    print(f"    RMSE: {ncf_eval['test_rmse']:.4f}")
        else:
            print(f"\n  ⚠️  No se encontró modelo NCF entrenado")
    else:
        print("\n  ⚠️  NCF omitido: PyTorch no disponible")

    # =========================================================================
    # PASO 9: RESUMEN Y CONCLUSIONES
    # =========================================================================
    print_section(9, "RESUMEN Y CONCLUSIONES")

    elapsed_time = time.time() - start_time

    print(f"\n  {'=' * 60}")
    print(f"  📊 RESUMEN DEL PIPELINE")
    print(f"  {'=' * 60}")

    print(f"\n  📁 Dataset:")
    print(f"    Reviews originales: {len(df):,}")
    print(f"    Reviews procesadas: {len(df_clean):,}")
    print(f"    Usuarios: {df_clean['user_idx'].nunique():,}")
    print(f"    Productos: {df_clean['product_idx'].nunique():,}")

    if ncf_eval:
        print(f"\n  📈 Modelo NCF:")
        if 'test_rmse' in ncf_eval:
            print(f"    Test RMSE: {ncf_eval['test_rmse']:.4f}")

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
    print(f"    1. Entrenar modelo NCF: python train_ncf_only.py")
    print(f"    2. Revisar gráficos en reports/")
    print(f"    3. Ejecutar la app web: streamlit run web/app.py")
    print(f"    4. Explorar notebooks en notebooks/")
    print(f"\n  {'=' * 60}\n")


if __name__ == '__main__':
    main()
