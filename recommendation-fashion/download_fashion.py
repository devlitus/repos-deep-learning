"""
Script para descargar el dataset Amazon Fashion Reviews desde Hugging Face
Fuente: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
"""
import os
import json
from pathlib import Path

def download_fashion_dataset(output_dir='data/raw'):
    """
    Descarga el dataset Amazon Fashion Reviews desde Hugging Face

    Este script descarga las reviews de ropa del dataset de Amazon.
    El dataset contiene: asin (product ID), reviewerID, overall (rating),
    reviewText, summary, y otros metadatos.

    Args:
        output_dir: Directorio donde se guardará el dataset
    """
    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    json_path = os.path.join(output_dir, "fashion_reviews.json")

    print("=" * 70)
    print("📥 Descargando Amazon Fashion Reviews Dataset desde Hugging Face...")
    print("=" * 70)
    print(f"Destino: {output_dir}")
    print(f"Dataset: McAuley-Lab/Amazon-Reviews-2023")
    print(f"Categoría: Clothing, Shoes and Jewelry")

    # Usar Hugging Face datasets
    try:
        from datasets import load_dataset
    except ImportError:
        print("\n❌ Error: 'datasets' no está instalado")
        print("Instala con: pip install datasets")
        print("\nAlternativas:")
        print("1. Descarga manual desde: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023")
        print("2. O desde: https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/")
        return False

    try:
        print("\n⏳ Cargando dataset desde Hugging Face...")
        print("   (Primera vez tardará un poco, se cachea localmente)")

        # Cargar dataset - filtrando solo ropa
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_meta_Clothing_Shoes_and_Jewelry",
            trust_remote_code=True,
            split="full"
        )

        print(f"\n✅ Dataset cargado: {len(dataset)} registros")
        print(f"   Columnas: {dataset.column_names}")

        # Convertir a JSONL
        print(f"\n💾 Guardando en formato JSONL...")
        count = 0
        with open(json_path, 'w', encoding='utf-8') as f:
            for row in dataset:
                # Convertir a formato compatible
                review = {
                    'reviewerID': row.get('reviewer_id', ''),
                    'asin': row.get('asin', ''),
                    'overall': float(row.get('rating', 0)),
                    'summary': row.get('title', ''),
                    'reviewText': row.get('text', ''),
                    'unixReviewTime': row.get('timestamp', 0)
                }
                f.write(json.dumps(review) + '\n')
                count += 1

                if (count + 1) % 10000 == 0:
                    print(f"   ✓ {count + 1:,} reviews guardados...")

        print(f"\n✅ Dataset guardado: {json_path}")
        print(f"   Total de reviews: {count:,}")

        # Mostrar información
        file_size = os.path.getsize(json_path) / (1024 * 1024)
        print(f"   Tamaño del archivo: {file_size:.2f} MB")

        print("\n" + "=" * 70)
        print("🎉 ¡Dataset descargado exitosamente!")
        print("=" * 70)
        print("\nEstructura de los datos:")
        print("  Cada línea es un JSON con:")
        print("    - reviewerID: ID del usuario")
        print("    - asin: ID del producto")
        print("    - overall: Rating (1-5 estrellas)")
        print("    - reviewText: Texto de la review")
        print("    - summary: Resumen de la review")
        print("    - unixReviewTime: Timestamp")

        print("\n✅ Próximos pasos:")
        print("   python main.py  # Ejecutar pipeline de análisis")

        return True

    except Exception as e:
        print(f"\n❌ Error descargando el dataset: {e}")
        print("\n📥 Alternativas para obtener el dataset:")
        print("1. Descargar desde Hugging Face:")
        print("   https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023")
        print("\n2. Descargar desde UCSD Datarepo:")
        print("   https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/")
        print("\n3. Usar datos de prueba:")
        print("   python generate_test_dataset.py")
        raise

if __name__ == "__main__":
    download_fashion_dataset()
