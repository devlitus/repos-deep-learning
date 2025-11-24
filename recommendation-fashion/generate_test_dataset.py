"""
Generador de Dataset de Prueba para Recommendation Fashion
Crea un dataset JSON simulado de Fashion Reviews para testing
"""
import json
import random
from pathlib import Path

def generate_test_dataset(output_dir='data/raw', num_reviews=5000, num_users=500, num_products=800):
    """
    Genera un dataset de prueba simulado en formato JSON (línea por línea)

    Args:
        output_dir: Directorio donde guardar el dataset
        num_reviews: Número total de reviews a generar
        num_users: Número de usuarios únicos
        num_products: Número de productos únicos
    """

    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_file = Path(output_dir) / 'fashion_reviews.json'

    print("=" * 60)
    print("🧪 GENERANDO DATASET DE PRUEBA")
    print("=" * 60)
    print(f"\n📊 Parámetros:")
    print(f"  - Número de reviews: {num_reviews:,}")
    print(f"  - Usuarios únicos: {num_users:,}")
    print(f"  - Productos únicos: {num_products:,}")
    print(f"  - Dispersión estimada: {(1 - num_reviews/(num_users*num_products))*100:.1f}%")

    print(f"\n📝 Generando reviews...")

    product_prefixes = ['shirt', 'pants', 'dress', 'jacket', 'shoes', 'hat', 'sweater', 'jeans', 'coat', 'blouse']
    summaries = [
        'Excellent product!',
        'Great quality and fast shipping',
        'Very satisfied',
        'Love it!',
        'Worth the price',
        'Poor quality',
        'Not as described',
        'Disappointing',
        'Average quality',
        'Good value for money'
    ]

    reviews_list = []

    # Generar reviews aleatorios
    for i in range(num_reviews):
        user_id = f'A{random.randint(1000, 1000+num_users):05d}'
        product_id = f'B{random.randint(0, num_products):08d}'
        rating = random.choices([1, 2, 3, 4, 5], weights=[10, 15, 25, 25, 25])[0]
        timestamp = random.randint(1400000000, 1600000000)

        review = {
            'reviewerID': user_id,
            'asin': product_id,
            'overall': rating,
            'summary': random.choice(summaries),
            'reviewText': f'This {random.choice(product_prefixes)} is ' +
                         ('great!' if rating >= 4 else 'not great.' if rating <= 2 else 'okay.'),
            'unixReviewTime': timestamp
        }

        reviews_list.append(review)

        if (i + 1) % 1000 == 0:
            print(f"  ✓ {i+1:,}/{num_reviews:,} reviews generados...")

    # Guardar en formato JSON (línea por línea)
    print(f"\n💾 Guardando en {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for review in reviews_list:
            f.write(json.dumps(review) + '\n')

    print(f"✅ Dataset de prueba creado exitosamente!")
    print(f"\n📊 Archivo creado:")
    print(f"  - Ruta: {output_file}")
    print(f"  - Tamaño: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print(f"  - Total de reviews: {len(reviews_list):,}")

    print("\n🎯 Próximos pasos:")
    print(f"  1. El archivo está listo en: {output_file}")
    print(f"  2. Ejecuta: python main.py")
    print(f"  3. El sistema ejecutará todos los análisis automáticamente")

    return output_file

if __name__ == '__main__':
    generate_test_dataset()
