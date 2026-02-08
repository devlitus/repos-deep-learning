"""
Script para generar datos sintéticos realistas de reviews de productos de moda
Simula comportamientos realistas de usuarios y productos en un sistema de recomendaciones

Características:
- Distribución de ley de potencia para usuarios activos (pocos muy activos, muchos casuales)
- Distribución de ley de potencia para productos populares (pocos bestsellers, muchos de nicho)
- Ratings sesgados hacia valores positivos (distribución realista)
- Timestamps distribuidos en un rango temporal
- Textos de reviews y summaries generados
- Verificación de productos con probabilidad realista
"""

import sys
import io
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Configurar UTF-8 para Windows
if sys.platform == 'win32' and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# Importar configuración
from config import DATASET_FILE, DATA_RAW_DIR, RANDOM_STATE

# =====================================
# PARÁMETROS DE GENERACIÓN
# =====================================

# Semilla para reproducibilidad
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Configuración de generación
NUM_INTERACTIONS = 100000  # Número de interacciones a generar
NUM_USERS = 500  # Número de usuarios únicos
NUM_PRODUCTS = 2000  # Número de productos únicos

# Rango temporal (últimos 3 años)
END_DATE = datetime(2024, 12, 31)
START_DATE = END_DATE - timedelta(days=365 * 3)

# Probabilidad de que un producto esté verificado (compra verificada)
VERIFIED_PROBABILITY = 0.65

# Distribución de ratings (sesgada hacia valores altos)
# En sistemas reales, las personas tienden a dejar reviews cuando están muy satisfechas o muy insatisfechas
RATING_WEIGHTS = {
    1.0: 0.05,  # 5% - Muy insatisfecho
    1.5: 0.03,
    2.0: 0.05,  # 5% - Insatisfecho
    2.5: 0.04,
    3.0: 0.08,  # 8% - Regular
    3.5: 0.10,
    4.0: 0.25,  # 25% - Satisfecho
    4.5: 0.20,
    5.0: 0.20   # 20% - Muy satisfecho
}

# Textos de ejemplo para reviews (se combinarán aleatoriamente)
REVIEW_TEMPLATES = [
    "Excelente producto, muy cómodo y de buena calidad.",
    "Me encantó, justo lo que esperaba. Muy recomendable.",
    "Buena relación calidad-precio. Lo volvería a comprar.",
    "No cumplió mis expectativas, la calidad es regular.",
    "Producto defectuoso, tuve que devolverlo.",
    "Material de buena calidad, talla perfecta.",
    "El color es más claro/oscuro de lo que se ve en las fotos.",
    "Llegó rápido y bien empaquetado.",
    "No es lo que esperaba, pero cumple su función.",
    "Perfecto para el precio que tiene.",
    "Muy satisfecho con la compra.",
    "No lo recomendaría, hay mejores opciones.",
    "Diseño bonito pero incómodo.",
    "Excelente para uso diario.",
    "Decepcionante, no vale la pena.",
]

SUMMARY_TEMPLATES = [
    "Excelente compra",
    "Muy recomendable",
    "Buena calidad",
    "No lo recomiendo",
    "Decepcionante",
    "Perfecto",
    "Justo lo que buscaba",
    "No cumplió expectativas",
    "Gran producto",
    "Regular",
    "Satisfecho",
    "No vale la pena",
    "Cómodo y práctico",
    "Mala calidad",
    "Me encantó",
]

# =====================================
# FUNCIONES DE GENERACIÓN
# =====================================

def generate_power_law_distribution(num_items, num_samples, alpha=2.0):
    """
    Genera una distribución de ley de potencia para simular usuarios/productos

    Args:
        num_items: Número de items únicos (usuarios o productos)
        num_samples: Número total de muestras a generar
        alpha: Parámetro de la distribución (mayor = más concentración)

    Returns:
        Lista con los índices de items seleccionados
    """
    # Generar probabilidades según ley de potencia
    ranks = np.arange(1, num_items + 1)
    probabilities = 1.0 / (ranks ** alpha)
    probabilities = probabilities / probabilities.sum()

    # Muestrear con reemplazo
    samples = np.random.choice(num_items, size=num_samples, p=probabilities)
    return samples.tolist()

def generate_user_id(user_index):
    """Genera un ID de usuario formateado"""
    return f"user_{user_index:05d}"

def generate_product_id(product_index):
    """Genera un ID de producto (ASIN) realista"""
    # Formato: B + 9 dígitos (similar a ASINs de Amazon)
    random_part = random.randint(100000000, 999999999)
    return f"B{random_part}"

def generate_timestamp():
    """Genera un timestamp aleatorio en el rango definido"""
    time_delta = END_DATE - START_DATE
    random_days = random.randint(0, time_delta.days)
    random_seconds = random.randint(0, 86400)  # Segundos en un día

    random_date = START_DATE + timedelta(days=random_days, seconds=random_seconds)
    return int(random_date.timestamp())

def generate_rating():
    """Genera un rating siguiendo la distribución definida"""
    ratings = list(RATING_WEIGHTS.keys())
    weights = list(RATING_WEIGHTS.values())
    return random.choices(ratings, weights=weights)[0]

def generate_review_text(rating):
    """Genera un texto de review basado en el rating"""
    # Seleccionar templates apropiados según el rating
    if rating >= 4.0:
        # Reviews positivas
        templates = [t for t in REVIEW_TEMPLATES if any(word in t.lower() for word in
                    ['excelente', 'encantó', 'recomendable', 'buena', 'perfecto', 'satisfecho'])]
    elif rating >= 3.0:
        # Reviews neutrales
        templates = [t for t in REVIEW_TEMPLATES if any(word in t.lower() for word in
                    ['regular', 'cumple', 'precio', 'esperaba'])]
    else:
        # Reviews negativas
        templates = [t for t in REVIEW_TEMPLATES if any(word in t.lower() for word in
                    ['no', 'defectuoso', 'decepcionante', 'regular', 'insatisfecho'])]

    if not templates:
        templates = REVIEW_TEMPLATES

    return random.choice(templates)

def generate_summary(rating):
    """Genera un resumen basado en el rating"""
    if rating >= 4.0:
        summaries = [s for s in SUMMARY_TEMPLATES if any(word in s.lower() for word in
                    ['excelente', 'recomendable', 'buena', 'perfecto', 'encantó', 'satisfecho'])]
    elif rating >= 3.0:
        summaries = [s for s in SUMMARY_TEMPLATES if 'regular' in s.lower() or 'cumplió' in s.lower()]
    else:
        summaries = [s for s in SUMMARY_TEMPLATES if any(word in s.lower() for word in
                    ['no', 'decepcionante', 'mala', 'vale'])]

    if not summaries:
        summaries = SUMMARY_TEMPLATES

    return random.choice(summaries)

def generate_synthetic_reviews(num_interactions, num_users, num_products):
    """
    Genera reviews sintéticas con distribuciones realistas

    Args:
        num_interactions: Número total de interacciones a generar
        num_users: Número de usuarios únicos
        num_products: Número de productos únicos

    Returns:
        Lista de diccionarios con las reviews
    """
    print("=" * 70)
    print("🎲 GENERANDO DATOS SINTÉTICOS REALISTAS")
    print("=" * 70)
    print(f"📊 Parámetros:")
    print(f"  - Interacciones: {num_interactions:,}")
    print(f"  - Usuarios: {num_users:,}")
    print(f"  - Productos: {num_products:,}")
    print(f"  - Rango temporal: {START_DATE.date()} a {END_DATE.date()}")
    print()

    # Generar distribuciones de usuarios y productos
    print("🔄 Generando distribuciones de ley de potencia...")
    user_distribution = generate_power_law_distribution(num_users, num_interactions, alpha=1.8)
    product_distribution = generate_power_law_distribution(num_products, num_interactions, alpha=2.2)

    # Crear mapeos de índices a IDs
    print("🆔 Creando IDs únicos de usuarios y productos...")
    user_ids = {i: generate_user_id(i) for i in range(num_users)}
    product_ids = {i: generate_product_id(i) for i in range(num_products)}

    # Generar reviews
    print("📝 Generando reviews...")
    reviews = []

    for i in range(num_interactions):
        user_idx = user_distribution[i]
        product_idx = product_distribution[i]

        rating = generate_rating()

        review = {
            'reviewerID': user_ids[user_idx],
            'asin': product_ids[product_idx],
            'overall': rating,
            'reviewText': generate_review_text(rating),
            'summary': generate_summary(rating),
            'unixReviewTime': generate_timestamp(),
            'verified': random.random() < VERIFIED_PROBABILITY
        }

        reviews.append(review)

        # Mostrar progreso
        if (i + 1) % 10000 == 0:
            print(f"  ✓ Generadas {i + 1:,} / {num_interactions:,} reviews")

    print(f"\n✅ Generación completada: {len(reviews):,} reviews")
    return reviews

def save_reviews_to_json(reviews, output_file, mode='append'):
    """
    Guarda las reviews en formato JSON línea por línea

    Args:
        reviews: Lista de diccionarios con las reviews
        output_file: Path del archivo de salida
        mode: 'append' para añadir al archivo existente, 'overwrite' para sobrescribir
    """
    print("\n" + "=" * 70)
    print("💾 GUARDANDO DATOS")
    print("=" * 70)

    # Crear directorio si no existe
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determinar modo de escritura
    if mode == 'overwrite' or not output_file.exists():
        write_mode = 'w'
        action = "Creando nuevo archivo"
    else:
        write_mode = 'a'
        action = "Añadiendo al archivo existente"

    print(f"📁 Archivo: {output_file.name}")
    print(f"🔧 Modo: {action}")

    # Escribir reviews
    count = 0
    with open(output_file, write_mode, encoding='utf-8') as f:
        for review in reviews:
            f.write(json.dumps(review, ensure_ascii=False) + '\n')
            count += 1

    print(f"✅ Guardadas {count:,} reviews")

    # Mostrar tamaño del archivo
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"📊 Tamaño del archivo: {file_size_mb:.2f} MB")

def analyze_generated_data(reviews):
    """Muestra estadísticas de los datos generados"""
    print("\n" + "=" * 70)
    print("📊 ESTADÍSTICAS DE DATOS GENERADOS")
    print("=" * 70)

    # Usuarios y productos únicos
    unique_users = len(set(r['reviewerID'] for r in reviews))
    unique_products = len(set(r['asin'] for r in reviews))

    print(f"\n👥 Usuarios únicos: {unique_users:,}")
    print(f"📦 Productos únicos: {unique_products:,}")
    print(f"⭐ Total de interacciones: {len(reviews):,}")

    # Distribución de ratings
    ratings = [r['overall'] for r in reviews]
    print(f"\n📈 Distribución de ratings:")
    for rating in sorted(set(ratings)):
        count = ratings.count(rating)
        percentage = (count / len(ratings)) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {rating:.1f} ⭐: {bar} {percentage:.1f}% ({count:,})")

    # Rating promedio
    avg_rating = sum(ratings) / len(ratings)
    print(f"\n📊 Rating promedio: {avg_rating:.2f} ⭐")

    # Productos verificados
    verified_count = sum(1 for r in reviews if r['verified'])
    verified_pct = (verified_count / len(reviews)) * 100
    print(f"\n✓ Reviews verificadas: {verified_pct:.1f}% ({verified_count:,})")

    # Sparsity
    sparsity = 1 - (len(reviews) / (unique_users * unique_products))
    print(f"\n🕸️  Sparsity: {sparsity * 100:.2f}%")

    # Top usuarios más activos
    from collections import Counter
    user_activity = Counter(r['reviewerID'] for r in reviews)
    print(f"\n👑 Top 5 usuarios más activos:")
    for user_id, count in user_activity.most_common(5):
        print(f"  - {user_id}: {count} reviews")

    # Top productos más reseñados
    product_popularity = Counter(r['asin'] for r in reviews)
    print(f"\n🏆 Top 5 productos más reseñados:")
    for product_id, count in product_popularity.most_common(5):
        print(f"  - {product_id}: {count} reviews")

# =====================================
# MAIN
# =====================================

def main():
    """Función principal"""
    print("\n" + "=" * 70)
    print("  🎨 GENERADOR DE DATOS SINTÉTICOS - FASHION RECOMMENDATIONS")
    print("=" * 70)
    print(f"📅 Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Preguntar al usuario el modo de operación
    print("🔧 Opciones de generación:")
    print("  1. Sobrescribir archivo existente (nuevo dataset)")
    print("  2. Añadir al archivo existente (expandir dataset)")
    print()

    while True:
        choice = input("Selecciona una opción (1 o 2): ").strip()
        if choice in ['1', '2']:
            break
        print("❌ Opción inválida. Por favor ingresa 1 o 2.")

    mode = 'overwrite' if choice == '1' else 'append'

    # Preguntar cantidad de interacciones
    print()
    while True:
        try:
            num_interactions_input = input(f"Número de interacciones a generar [{NUM_INTERACTIONS}]: ").strip()
            if num_interactions_input == '':
                num_interactions = NUM_INTERACTIONS
            else:
                num_interactions = int(num_interactions_input)

            if num_interactions > 0:
                break
            print("❌ El número debe ser mayor que 0.")
        except ValueError:
            print("❌ Por favor ingresa un número válido.")

    print()

    try:
        # Generar reviews
        reviews = generate_synthetic_reviews(num_interactions, NUM_USERS, NUM_PRODUCTS)

        # Analizar datos generados
        analyze_generated_data(reviews)

        # Guardar en archivo
        save_reviews_to_json(reviews, DATASET_FILE, mode=mode)

        print("\n" + "=" * 70)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print(f"\n📁 Archivo generado: {DATASET_FILE}")
        print(f"📊 Total de reviews: {len(reviews):,}")
        print(f"\n💡 Siguiente paso: Ejecuta 'python main.py' para entrenar los modelos")
        print()

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR EN LA GENERACIÓN")
        print("=" * 70)
        print(f"\n{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
