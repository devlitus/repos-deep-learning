"""
Script avanzado para generar datasets personalizados con control fino
Permite ajustar todos los parámetros de generación según necesidades específicas

Casos de uso:
- Generar datasets pequeños para pruebas rápidas
- Crear datasets balanceados o desbalanceados
- Simular escenarios específicos (alta sparsity, cold start, etc.)
- Generar datasets con características específicas
"""

import sys
import io
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import random

# Configurar UTF-8 para Windows
if sys.platform == 'win32' and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

from config import DATA_RAW_DIR, RANDOM_STATE

# =====================================
# PERFILES PREDEFINIDOS
# =====================================

PROFILES = {
    'mini': {
        'description': 'Dataset pequeño para pruebas rápidas (5k reviews)',
        'num_interactions': 5000,
        'num_users': 100,
        'num_products': 300,
        'user_alpha': 1.5,
        'product_alpha': 2.0,
        'verified_prob': 0.6,
        'rating_distribution': 'normal'  # Más balanceado
    },
    'balanced': {
        'description': 'Dataset balanceado para entrenamiento (50k reviews)',
        'num_interactions': 50000,
        'num_users': 300,
        'num_products': 1000,
        'user_alpha': 1.7,
        'product_alpha': 2.0,
        'verified_prob': 0.65,
        'rating_distribution': 'balanced'
    },
    'realistic': {
        'description': 'Dataset realista tipo e-commerce (100k reviews)',
        'num_interactions': 100000,
        'num_users': 500,
        'num_products': 2000,
        'user_alpha': 1.8,
        'product_alpha': 2.2,
        'verified_prob': 0.65,
        'rating_distribution': 'skewed_positive'  # Sesgado hacia 4-5 estrellas
    },
    'large': {
        'description': 'Dataset grande para producción (500k reviews)',
        'num_interactions': 500000,
        'num_users': 2000,
        'num_products': 5000,
        'user_alpha': 1.9,
        'product_alpha': 2.3,
        'verified_prob': 0.68,
        'rating_distribution': 'skewed_positive'
    },
    'cold_start': {
        'description': 'Dataset para simular cold start (muchos usuarios/productos con pocas interacciones)',
        'num_interactions': 30000,
        'num_users': 1000,  # Muchos usuarios
        'num_products': 3000,  # Muchos productos
        'user_alpha': 1.3,  # Distribución más plana
        'product_alpha': 1.5,  # Distribución más plana
        'verified_prob': 0.5,
        'rating_distribution': 'balanced'
    },
    'sparse': {
        'description': 'Dataset muy sparse para testing (alta dispersión)',
        'num_interactions': 20000,
        'num_users': 800,
        'num_products': 4000,  # Muchos productos, pocas interacciones
        'user_alpha': 1.2,
        'product_alpha': 1.3,
        'verified_prob': 0.55,
        'rating_distribution': 'balanced'
    }
}

# =====================================
# DISTRIBUCIONES DE RATINGS
# =====================================

RATING_DISTRIBUTIONS = {
    'skewed_positive': {  # Realista: usuarios dejan más reviews positivas
        1.0: 0.05, 1.5: 0.03, 2.0: 0.05, 2.5: 0.04,
        3.0: 0.08, 3.5: 0.10, 4.0: 0.25, 4.5: 0.20, 5.0: 0.20
    },
    'balanced': {  # Más equilibrado para entrenamiento
        1.0: 0.08, 1.5: 0.07, 2.0: 0.10, 2.5: 0.10,
        3.0: 0.15, 3.5: 0.15, 4.0: 0.15, 4.5: 0.10, 5.0: 0.10
    },
    'normal': {  # Distribución más normal alrededor de 3
        1.0: 0.05, 1.5: 0.08, 2.0: 0.12, 2.5: 0.15,
        3.0: 0.20, 3.5: 0.15, 4.0: 0.12, 4.5: 0.08, 5.0: 0.05
    },
    'skewed_negative': {  # Para testing: más reviews negativas
        1.0: 0.20, 1.5: 0.15, 2.0: 0.15, 2.5: 0.15,
        3.0: 0.15, 3.5: 0.08, 4.0: 0.07, 4.5: 0.03, 5.0: 0.02
    },
    'polarized': {  # Reviews muy polarizadas (muy buenas o muy malas)
        1.0: 0.25, 1.5: 0.05, 2.0: 0.05, 2.5: 0.05,
        3.0: 0.10, 3.5: 0.05, 4.0: 0.05, 4.5: 0.15, 5.0: 0.25
    }
}

# =====================================
# FUNCIONES DE GENERACIÓN
# =====================================

def generate_power_law_distribution(num_items, num_samples, alpha=2.0):
    """Genera distribución de ley de potencia"""
    ranks = np.arange(1, num_items + 1)
    probabilities = 1.0 / (ranks ** alpha)
    probabilities = probabilities / probabilities.sum()
    samples = np.random.choice(num_items, size=num_samples, p=probabilities)
    return samples.tolist()

def generate_user_id(user_index):
    """Genera ID de usuario"""
    return f"user_{user_index:05d}"

def generate_product_id(product_index):
    """Genera ASIN de producto"""
    random_part = random.randint(100000000, 999999999)
    return f"B{random_part}"

def generate_timestamp(start_date, end_date):
    """Genera timestamp aleatorio"""
    time_delta = end_date - start_date
    random_days = random.randint(0, time_delta.days)
    random_seconds = random.randint(0, 86400)
    random_date = start_date + timedelta(days=random_days, seconds=random_seconds)
    return int(random_date.timestamp())

def generate_rating(distribution_name):
    """Genera rating según la distribución especificada"""
    distribution = RATING_DISTRIBUTIONS[distribution_name]
    ratings = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(ratings, weights=weights)[0]

def generate_review_text(rating):
    """Genera texto de review basado en rating"""
    if rating >= 4.0:
        templates = [
            "Excelente producto, muy satisfecho con la compra.",
            "Muy buena calidad, lo recomiendo totalmente.",
            "Perfecto, justo lo que buscaba.",
            "Gran producto, volveré a comprar.",
            "Me encantó, superó mis expectativas."
        ]
    elif rating >= 3.0:
        templates = [
            "Producto aceptable, cumple con lo básico.",
            "Está bien por el precio que tiene.",
            "No está mal, aunque esperaba más.",
            "Regular, tiene algunos detalles mejorables.",
            "Es correcto, nada extraordinario."
        ]
    else:
        templates = [
            "No cumplió mis expectativas.",
            "Mala calidad, no lo recomiendo.",
            "Decepcionante, tuve que devolverlo.",
            "No vale la pena, hay mejores opciones.",
            "Producto defectuoso, muy insatisfecho."
        ]
    return random.choice(templates)

def generate_summary(rating):
    """Genera resumen basado en rating"""
    if rating >= 4.0:
        return random.choice(["Excelente", "Muy bueno", "Recomendado", "Perfecto", "Me encantó"])
    elif rating >= 3.0:
        return random.choice(["Aceptable", "Está bien", "Regular", "Cumple", "Normal"])
    else:
        return random.choice(["Decepcionante", "Mala calidad", "No recomendado", "Malo", "Insatisfecho"])

def generate_dataset(config, output_file, mode='overwrite'):
    """
    Genera dataset con la configuración especificada

    Args:
        config: Diccionario con parámetros de generación
        output_file: Path del archivo de salida
        mode: 'overwrite' o 'append'
    """
    print("=" * 70)
    print("🎲 GENERANDO DATASET PERSONALIZADO")
    print("=" * 70)
    print(f"\n📊 Configuración:")
    for key, value in config.items():
        print(f"  - {key}: {value}")

    # Semillas
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Extraer parámetros
    num_interactions = config['num_interactions']
    num_users = config['num_users']
    num_products = config['num_products']
    user_alpha = config.get('user_alpha', 1.8)
    product_alpha = config.get('product_alpha', 2.2)
    verified_prob = config.get('verified_prob', 0.65)
    rating_dist = config.get('rating_distribution', 'skewed_positive')

    # Rango temporal
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=365 * 3)

    # Calcular sparsity esperada
    sparsity = 1 - (num_interactions / (num_users * num_products))
    print(f"\n🕸️  Sparsity esperada: {sparsity * 100:.2f}%")

    # Generar distribuciones
    print("\n🔄 Generando distribuciones...")
    user_distribution = generate_power_law_distribution(num_users, num_interactions, user_alpha)
    product_distribution = generate_power_law_distribution(num_products, num_interactions, product_alpha)

    # Crear IDs
    print("🆔 Creando IDs...")
    user_ids = {i: generate_user_id(i) for i in range(num_users)}
    product_ids = {i: generate_product_id(i) for i in range(num_products)}

    # Generar reviews
    print("📝 Generando reviews...")
    reviews = []

    for i in range(num_interactions):
        user_idx = user_distribution[i]
        product_idx = product_distribution[i]
        rating = generate_rating(rating_dist)

        review = {
            'reviewerID': user_ids[user_idx],
            'asin': product_ids[product_idx],
            'overall': rating,
            'reviewText': generate_review_text(rating),
            'summary': generate_summary(rating),
            'unixReviewTime': generate_timestamp(start_date, end_date),
            'verified': random.random() < verified_prob
        }

        reviews.append(review)

        if (i + 1) % 10000 == 0:
            print(f"  ✓ {i + 1:,} / {num_interactions:,}")

    # Estadísticas
    print("\n" + "=" * 70)
    print("📊 ESTADÍSTICAS DEL DATASET")
    print("=" * 70)

    unique_users = len(set(r['reviewerID'] for r in reviews))
    unique_products = len(set(r['asin'] for r in reviews))
    ratings = [r['overall'] for r in reviews]
    avg_rating = sum(ratings) / len(ratings)
    verified_count = sum(1 for r in reviews if r['verified'])

    print(f"\n👥 Usuarios únicos: {unique_users:,}")
    print(f"📦 Productos únicos: {unique_products:,}")
    print(f"⭐ Total interacciones: {len(reviews):,}")
    print(f"📊 Rating promedio: {avg_rating:.2f}")
    print(f"✓ Reviews verificadas: {verified_count / len(reviews) * 100:.1f}%")

    # Distribución de ratings
    print(f"\n📈 Distribución de ratings:")
    for rating in sorted(set(ratings)):
        count = ratings.count(rating)
        pct = (count / len(ratings)) * 100
        bar = '█' * int(pct / 2)
        print(f"  {rating:.1f} ⭐: {bar} {pct:.1f}%")

    # Guardar
    print("\n" + "=" * 70)
    print("💾 GUARDANDO DATASET")
    print("=" * 70)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_mode = 'w' if mode == 'overwrite' else 'a'

    with open(output_file, write_mode, encoding='utf-8') as f:
        for review in reviews:
            f.write(json.dumps(review, ensure_ascii=False) + '\n')

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n✅ Guardado: {output_file.name}")
    print(f"📊 Tamaño: {file_size_mb:.2f} MB")

    return reviews

# =====================================
# INTERFAZ DE LÍNEA DE COMANDOS
# =====================================

def main():
    parser = argparse.ArgumentParser(
        description='Generador avanzado de datasets sintéticos para sistemas de recomendación',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Perfiles disponibles:
  mini        - Dataset pequeño para pruebas (5k reviews)
  balanced    - Dataset balanceado (50k reviews)
  realistic   - Dataset realista e-commerce (100k reviews)
  large       - Dataset grande (500k reviews)
  cold_start  - Simula problema de cold start
  sparse      - Dataset muy disperso

Distribuciones de rating:
  skewed_positive - Sesgado hacia ratings altos (realista)
  balanced        - Distribución equilibrada
  normal          - Distribución normal alrededor de 3
  skewed_negative - Sesgado hacia ratings bajos
  polarized       - Reviews polarizadas (muy buenas o muy malas)

Ejemplos:
  # Usar perfil predefinido
  python generate_custom_dataset.py --profile realistic

  # Configuración personalizada
  python generate_custom_dataset.py --interactions 75000 --users 400 --products 1500

  # Dataset para cold start
  python generate_custom_dataset.py --profile cold_start --output cold_start_data.json

  # Añadir al dataset existente
  python generate_custom_dataset.py --profile mini --mode append
        '''
    )

    parser.add_argument('--profile', '-p', choices=list(PROFILES.keys()),
                       help='Usar perfil predefinido')

    parser.add_argument('--interactions', '-n', type=int,
                       help='Número de interacciones')
    parser.add_argument('--users', '-u', type=int,
                       help='Número de usuarios únicos')
    parser.add_argument('--products', '-i', type=int,
                       help='Número de productos únicos')

    parser.add_argument('--user-alpha', type=float,
                       help='Parámetro alpha para distribución de usuarios')
    parser.add_argument('--product-alpha', type=float,
                       help='Parámetro alpha para distribución de productos')

    parser.add_argument('--verified-prob', type=float,
                       help='Probabilidad de review verificada (0-1)')

    parser.add_argument('--rating-dist', choices=list(RATING_DISTRIBUTIONS.keys()),
                       help='Distribución de ratings')

    parser.add_argument('--output', '-o', type=str,
                       default='fashion_reviews.json',
                       help='Nombre del archivo de salida')

    parser.add_argument('--mode', '-m', choices=['overwrite', 'append'],
                       default='overwrite',
                       help='Modo de escritura')

    parser.add_argument('--list-profiles', action='store_true',
                       help='Listar perfiles disponibles y salir')

    args = parser.parse_args()

    # Listar perfiles
    if args.list_profiles:
        print("\n" + "=" * 70)
        print("📋 PERFILES DISPONIBLES")
        print("=" * 70)
        for name, profile in PROFILES.items():
            print(f"\n🔹 {name}")
            print(f"   {profile['description']}")
            print(f"   Interacciones: {profile['num_interactions']:,}")
            print(f"   Usuarios: {profile['num_users']:,}")
            print(f"   Productos: {profile['num_products']:,}")
            print(f"   Sparsity: {(1 - profile['num_interactions'] / (profile['num_users'] * profile['num_products'])) * 100:.2f}%")
        return 0

    # Determinar configuración
    if args.profile:
        config = PROFILES[args.profile].copy()
        print(f"\n📋 Usando perfil: {args.profile}")
        print(f"   {config['description']}")
    else:
        # Configuración personalizada
        if not all([args.interactions, args.users, args.products]):
            parser.error("Debes especificar --profile O proporcionar --interactions, --users y --products")

        config = {
            'num_interactions': args.interactions,
            'num_users': args.users,
            'num_products': args.products,
            'user_alpha': args.user_alpha or 1.8,
            'product_alpha': args.product_alpha or 2.2,
            'verified_prob': args.verified_prob or 0.65,
            'rating_distribution': args.rating_dist or 'skewed_positive'
        }

    # Sobrescribir con argumentos CLI si se proporcionan
    if args.user_alpha:
        config['user_alpha'] = args.user_alpha
    if args.product_alpha:
        config['product_alpha'] = args.product_alpha
    if args.verified_prob:
        config['verified_prob'] = args.verified_prob
    if args.rating_dist:
        config['rating_distribution'] = args.rating_dist

    # Archivo de salida
    output_file = DATA_RAW_DIR / args.output

    print("\n" + "=" * 70)
    print("  🎨 GENERADOR AVANZADO DE DATASETS SINTÉTICOS")
    print("=" * 70)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Generar
    try:
        generate_dataset(config, output_file, mode=args.mode)

        print("\n" + "=" * 70)
        print("✅ GENERACIÓN COMPLETADA")
        print("=" * 70)
        print(f"\n📁 Archivo: {output_file}")
        print(f"💡 Siguiente paso: python main.py")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR")
        print("=" * 70)
        print(f"\n{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
