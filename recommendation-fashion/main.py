"""
Sistema de Recomendación de Moda - Pipeline Completo
Ejecuta todos los análisis en orden
"""
import subprocess
import sys
from pathlib import Path

def print_banner(text):
    """Imprime un banner decorativo"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def run_module(module_name, description):
    """Ejecuta un módulo Python"""
    print_banner(description)

    module_path = Path(__file__).parent / 'src' / f'{module_name}.py'

    if not module_path.exists():
        print(f"❌ Error: No se encontró {module_name}.py")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(module_path)],
            cwd=Path(__file__).parent,
            capture_output=False
        )

        if result.returncode == 0:
            print(f"\n✅ {description} completado exitosamente")
            return True
        else:
            print(f"\n❌ Error ejecutando {description}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Ejecuta el pipeline completo"""

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "👕 SISTEMA DE RECOMENDACIÓN DE MODA 👕" + " " * 20 + "║")
    print("║" + " " * 10 + "Amazon Fashion Reviews - Collaborative Filtering" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")

    # Step 0: Download dataset
    print_banner("PASO 0: DESCARGAR DATASET")
    print("Para descargar el dataset Amazon Fashion Reviews, ejecuta:")
    print("  python download_fashion.py")
    print("\n⏸️  Por favor, descarga el dataset primero si aún no lo has hecho.")
    print("Continuaré con el análisis...")

    # Step 1: Exploratory Analysis
    print_banner("PASO 1: ANÁLISIS EXPLORATORIO")
    success_1 = run_module('exploratory_analysis', '📊 Análisis Exploratorio de Datos')

    if not success_1:
        print("\n⚠️  El análisis exploratorio falló. Verifica que el dataset esté descargado.")
        return

    # Step 2: User-Based CF
    print_banner("PASO 2: FILTRADO COLABORATIVO BASADO EN USUARIOS")
    success_2 = run_module('user_based_collaborative_filtering', '👥 User-Based Collaborative Filtering')

    # Step 3: Item-Based CF
    print_banner("PASO 3: FILTRADO COLABORATIVO BASADO EN PRODUCTOS")
    success_3 = run_module('item_based_collaborative_filtering', '👕 Item-Based Collaborative Filtering')

    # Step 4: Matrix Factorization SVD
    print_banner("PASO 4: FACTORIZACIÓN DE MATRICES (SVD)")
    success_4 = run_module('matrix_factorization_svd', '📊 Matrix Factorization (SVD)')

    # Step 5: Hybrid System
    print_banner("PASO 5: SISTEMA HÍBRIDO")
    success_5 = run_module('hybrid_recommender_system', '🎯 Sistema Híbrido de Recomendación')

    # Final Summary
    print_banner("📋 RESUMEN DEL PIPELINE")

    print("Módulos ejecutados:")
    print(f"  1. Análisis Exploratorio: {'✅ OK' if success_1 else '❌ FALLO'}")
    print(f"  2. User-Based CF: {'✅ OK' if success_2 else '❌ FALLO'}")
    print(f"  3. Item-Based CF: {'✅ OK' if success_3 else '❌ FALLO'}")
    print(f"  4. Matrix Factorization SVD: {'✅ OK' if success_4 else '❌ FALLO'}")
    print(f"  5. Hybrid System: {'✅ OK' if success_5 else '❌ FALLO'}")

    print("\n📊 Resultados guardados en:")
    print("  - reports/: Visualizaciones y gráficos")
    print("  - data/processed/: Datos procesados")

    print("\n🎯 Próximos pasos:")
    print("  - Revisar los gráficos en la carpeta 'reports/'")
    print("  - Analizar las métricas de cada algoritmo")
    print("  - Comparar rendimiento de User-Based vs Item-Based vs SVD")

    print("\n✅ Pipeline completado!\n")

if __name__ == '__main__':
    main()
