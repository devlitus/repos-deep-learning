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
    Genera un dataset sintético con PATRONES REALISTAS para aprendizaje

    Simula comportamientos reales:
    - Usuarios con preferencias consistentes (optimistas vs críticos)
    - Productos con calidad intrínseca (buenos vs malos)
    - Categorías de productos (Fashion, Shoes, Accessories)
    - Afinidad usuario-categoría

    Args:
        num_reviews: Número de reviews a generar (default: 50,000)
    """
    print("=" * 80)
    print(f"  GENERANDO DATASET CON PATRONES REALISTAS - {num_reviews:,} REVIEWS")
    print("=" * 80)

    random.seed(RANDOM_STATE)
    import numpy as np
    np.random.seed(RANDOM_STATE)

    # Calcular número de usuarios y productos
    num_users = max(200, num_reviews // 25)
    num_products = max(500, num_reviews // 10)

    print(f"\nParámetros:")
    print(f"  - Reviews: {num_reviews:,}")
    print(f"  - Usuarios: {num_users:,} (aprox {num_reviews/num_users:.1f} reviews/usuario)")
    print(f"  - Productos: {num_products:,} (aprox {num_reviews/num_products:.1f} reviews/producto)")

    # ===== CREAR USUARIOS CON PERSONALIDADES =====
    print(f"\n[1/5] Creando usuarios con preferencias...")

    categories = ['Fashion', 'Shoes', 'Accessories']

    users = {}
    for i in range(num_users):
        user_id = f"user_{i:05d}"

        # Personalidad del usuario (distribución normal)
        # bias: qué tan optimista/crítico es (-2 a +2)
        # variance: qué tan consistente es (0.2 a 0.8)
        users[user_id] = {
            'bias': np.random.normal(0, 0.8),  # -2 a +2 aprox
            'variance': np.random.uniform(0.2, 0.8),
            'favorite_category': random.choice(categories),
            'activity': np.random.exponential(25)  # Algunos usuarios más activos
        }

    # ===== CREAR PRODUCTOS CON CALIDAD =====
    print(f"[2/5] Creando productos con calidad intrínseca...")

    products = {}
    for i in range(num_products):
        product_id = f"B{random.randint(100000000, 999999999):09d}"

        # Calidad del producto (distribución sesgada hacia buenos productos)
        # 60% buenos (3.5-5.0), 30% mediocres (2.5-3.5), 10% malos (1.0-2.5)
        quality_tier = np.random.choice(['good', 'medium', 'bad'], p=[0.6, 0.3, 0.1])

        if quality_tier == 'good':
            base_quality = np.random.uniform(3.5, 5.0)
        elif quality_tier == 'medium':
            base_quality = np.random.uniform(2.5, 3.5)
        else:
            base_quality = np.random.uniform(1.0, 2.5)

        products[product_id] = {
            'quality': base_quality,
            'category': random.choice(categories),
            'popularity': np.random.exponential(10)
        }

    user_ids = list(users.keys())
    product_ids = list(products.keys())

    # ===== GENERAR REVIEWS CON PATRONES =====
    print(f"[3/5] Generando reviews con patrones realistas...")
    reviews = []
    base_time = int(datetime(2020, 1, 1).timestamp())

    for i in range(num_reviews):
        if (i + 1) % 10000 == 0:
            print(f"  Progreso: {i+1:,}/{num_reviews:,} ({(i+1)/num_reviews*100:.1f}%)")

        # Seleccionar usuario (ponderado por actividad)
        user_id = random.choice(user_ids)
        user = users[user_id]

        # Seleccionar producto (ponderado por popularidad)
        product_id = random.choice(product_ids)
        product = products[product_id]

        # CALCULAR RATING REALISTA
        # Base = calidad del producto
        rating = product['quality']

        # Ajustar por bias del usuario (optimista/crítico)
        rating += user['bias']

        # Bonus si es la categoría favorita del usuario (+0.5 promedio)
        if product['category'] == user['favorite_category']:
            rating += np.random.uniform(0, 1.0)

        # Agregar ruido aleatorio (varianza del usuario)
        rating += np.random.normal(0, user['variance'])

        # Clip al rango válido [1.0, 5.0]
        rating = max(1.0, min(5.0, rating))

        # Redondear a .0 o .5 (más realista)
        rating = round(rating * 2) / 2

        review = {
            "reviewerID": user_id,
            "asin": product_id,
            "overall": rating,
            "reviewText": f"Review for {product['category']} product (quality: {product['quality']:.1f})",
            "summary": f"{product['category']} review",
            "unixReviewTime": base_time + random.randint(0, 4*365*24*3600),
            "verified": random.choice([True, True, True, False])
        }
        reviews.append(review)

    print(f"[4/5] Guardando archivo JSON...")

    # Guardar archivo
    with open(DATASET_FILE, 'w', encoding='utf-8') as f:
        for i, review in enumerate(reviews):
            if (i + 1) % 10000 == 0:
                print(f"  Guardadas: {i+1:,}/{num_reviews:,}")
            json.dump(review, f, ensure_ascii=False)
            f.write('\n')

    print(f"[5/5] Validando patrones generados...")

    # Estadísticas finales
    file_size_mb = DATASET_FILE.stat().st_size / (1024 * 1024)

    # Contar líneas y validar patrones
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)

    # Calcular estadísticas de patrones
    import pandas as pd
    df = pd.DataFrame(reviews)

    # Consistencia de usuarios
    user_consistency = df.groupby('reviewerID')['overall'].std().mean()

    # Consistencia de productos
    product_consistency = df.groupby('asin')['overall'].std().mean()

    print("\n" + "=" * 80)
    print("  DATASET CON PATRONES GENERADO EXITOSAMENTE")
    print("=" * 80)
    print(f"\n📊 Estadísticas:")
    print(f"  - Archivo: {DATASET_FILE.name}")
    print(f"  - Ubicación: {DATASET_FILE.parent}")
    print(f"  - Reviews: {line_count:,}")
    print(f"  - Usuarios únicos: {num_users:,}")
    print(f"  - Productos únicos: {num_products:,}")
    print(f"  - Tamaño: {file_size_mb:.2f} MB")

    print(f"\n🎯 Calidad de Patrones:")
    print(f"  - Consistencia de usuarios: {user_consistency:.3f}")
    print(f"    {'✅ BUENO' if user_consistency < 1.0 else '⚠️  MEJORABLE'} (<1.0 = patrones fuertes)")
    print(f"  - Consistencia de productos: {product_consistency:.3f}")
    print(f"    {'✅ BUENO' if product_consistency < 1.0 else '⚠️  MEJORABLE'} (<1.0 = patrones fuertes)")

    print(f"\n📈 Distribución de ratings:")
    rating_dist = df['overall'].value_counts().sort_index()
    for rating in sorted(rating_dist.index):
        count = rating_dist[rating]
        pct = count / len(df) * 100
        bar = '█' * int(pct / 2)
        print(f"  {rating}: {count:6,} ({pct:5.1f}%) {bar}")

    print(f"\n🚀 Siguientes pasos:")
    print(f"  1. python train_ncf_only.py      (entrenar NCF con patrones)")
    print(f"  2. python main.py                (pipeline completo)")
    print(f"  3. streamlit run web/app.py      (app web)")
    print("\n💡 Nota: Este dataset tiene PATRONES REALES que el modelo puede aprender")
    print("  - Usuarios consistentes en sus preferencias")
    print("  - Productos con calidad intrínseca")
    print("  - Afinidad usuario-categoría")
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
