#!/usr/bin/env python3
"""
Generador de Dataset Sintético Grande
Genera dataset de prueba con más datos para mejor entrenamiento de Deep Learning
"""

import sys
import io
import json
import random
from pathlib import Path
from datetime import datetime

# Configurar UTF-8
if sys.platform == 'win32' and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# Importar configuración
from config import DATASET_FILE, RANDOM_STATE

def generate_synthetic_dataset(num_reviews=50000):
    """
    Genera un dataset sintético con distribución realista
    
    Args:
        num_reviews: Número de reviews a generar (default: 50,000)
    """
    print("=" * 80)
    print(f"  GENERANDO DATASET SINTÉTICO - {num_reviews:,} REVIEWS")
    print("=" * 80)
    
    random.seed(RANDOM_STATE)
    
    # Calcular número de usuarios y productos basado en reviews
    # Ratio típico: 1 usuario → 20-30 reviews, 1 producto → 8-12 reviews
    num_users = max(200, num_reviews // 25)
    num_products = max(500, num_reviews // 10)
    
    print(f"\nParámetros:")
    print(f"  - Reviews: {num_reviews:,}")
    print(f"  - Usuarios: {num_users:,} (aprox {num_reviews/num_users:.1f} reviews/usuario)")
    print(f"  - Productos: {num_products:,} (aprox {num_reviews/num_products:.1f} reviews/producto)")
    
    # Generar IDs
    print(f"\n[1/4] Generando IDs de usuarios y productos...")
    user_ids = [f"user_{i:05d}" for i in range(num_users)]
    product_ids = [f"B{random.randint(100000000, 999999999):09d}" for _ in range(num_products)]
    
    # Distribución de ratings más realista (sesgo hacia ratings altos)
    rating_distribution = [1.0]*5 + [2.0]*10 + [3.0]*20 + [4.0]*30 + [5.0]*35
    
    print(f"[2/4] Generando reviews...")
    reviews = []
    base_time = int(datetime(2020, 1, 1).timestamp())
    
    for i in range(num_reviews):
        if (i + 1) % 10000 == 0:
            print(f"  Progreso: {i+1:,}/{num_reviews:,} ({(i+1)/num_reviews*100:.1f}%)")
        
        review = {
            "reviewerID": random.choice(user_ids),
            "asin": random.choice(product_ids),
            "overall": random.choice(rating_distribution),
            "reviewText": f"Synthetic review #{i+1} for fashion product",
            "summary": f"Review summary #{i+1}",
            "unixReviewTime": base_time + random.randint(0, 4*365*24*3600),  # 4 años
            "verified": random.choice([True, True, True, False])  # 75% verificadas
        }
        reviews.append(review)
    
    print(f"[3/4] Guardando archivo JSON...")
    
    # Guardar archivo
    with open(DATASET_FILE, 'w', encoding='utf-8') as f:
        for i, review in enumerate(reviews):
            if (i + 1) % 10000 == 0:
                print(f"  Guardadas: {i+1:,}/{num_reviews:,}")
            json.dump(review, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"[4/4] Validando archivo...")
    
    # Estadísticas finales
    file_size_mb = DATASET_FILE.stat().st_size / (1024 * 1024)
    
    # Contar líneas reales
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    
    print("\n" + "=" * 80)
    print("  DATASET GENERADO EXITOSAMENTE")
    print("=" * 80)
    print(f"\nEstadísticas:")
    print(f"  - Archivo: {DATASET_FILE.name}")
    print(f"  - Ubicación: {DATASET_FILE.parent}")
    print(f"  - Reviews: {line_count:,}")
    print(f"  - Usuarios únicos: {num_users:,}")
    print(f"  - Productos únicos: {num_products:,}")
    print(f"  - Tamaño: {file_size_mb:.2f} MB")
    print(f"\nSiguientes pasos:")
    print(f"  1. python main.py                (ejecutar pipeline completo)")
    print(f"  2. python train_ncf_only.py      (entrenar solo NCF)")
    print(f"  3. streamlit run web/app.py      (ver app web)")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar dataset sintético grande')
    parser.add_argument('--size', type=int, default=50000,
                      help='Número de reviews a generar (default: 50000)')
    
    args = parser.parse_args()
    
    print(f"\nNota: Generando {args.size:,} reviews sintéticas...")
    print("Para dataset real, ver DEEP_HYBRID_GUIDE.md\n")
    
    try:
        generate_synthetic_dataset(args.size)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
