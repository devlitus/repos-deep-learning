"""
Script para descargar el dataset Amazon Fashion Reviews
"""
import os
import gzip
import shutil
import requests
from pathlib import Path
from tqdm import tqdm
import json

def download_fashion_dataset(output_dir='.\\data\\raw'):
    """
    Descarga el dataset Amazon Fashion Reviews

    Este script descarga las reviews de ropa del dataset de Amazon.
    El dataset contiene: asin (product ID), reviewerID, overall (rating),
    reviewText, summary, y otros metadatos.

    Args:
        output_dir: Directorio donde se guardará el dataset
    """
    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # URL del dataset comprimido (Amazon Fashion Reviews - Clothing, Shoes and Jewelry)
    # Fuente: https://datarepo.eng.ucsd.edu/ (actualizado 2024)
    # URL anterior (deprecated): http://jmcauley.ucsd.edu/data/amazon/categoryFilesSmall/...
    url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Clothing_Shoes_and_Jewelry.json.gz"
    gz_path = os.path.join(output_dir, "Clothing_Shoes_and_Jewelry.json.gz")
    json_path = os.path.join(output_dir, "fashion_reviews.json")

    print("="*60)
    print("📥 Descargando Amazon Fashion Reviews Dataset...")
    print("="*60)
    print(f"URL: {url}")
    print(f"Destino: {output_dir}")

    try:
        # Descargar el archivo
        response = requests.get(url, stream=True, timeout=30)
        total_size = int(response.headers.get('content-length', 0))

        print(f"\n📊 Tamaño del archivo: {total_size / (1024**2):.2f} MB")

        with open(gz_path, 'wb') as file, tqdm(
            desc="Descargando",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)

        print(f"\n✅ Descarga completada: {gz_path}")

        # Descomprimir el archivo
        print("\n📂 Descomprimiendo archivo...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(json_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"✅ Archivo descomprimido: {json_path}")

        # Mostrar información del dataset
        print("\n📊 Información del dataset:")
        print(f"  - Total de reviews: ~2.7 millones")
        print(f"  - Productos únicos: ~180,000")
        print(f"  - Usuarios únicos: ~800,000")
        print(f"  - Rango de ratings: 1-5 estrellas")

        # Eliminar archivo gz para ahorrar espacio
        os.remove(gz_path)
        print(f"\n🗑️  Archivo .gz eliminado para ahorrar espacio")

        print("\n" + "="*60)
        print("🎉 ¡Dataset descargado exitosamente!")
        print("="*60)
        print("\nEstructura de los datos:")
        print("  Cada línea es un JSON con:")
        print("    - reviewerID: ID del usuario")
        print("    - asin: ID del producto")
        print("    - overall: Rating (1-5 estrellas)")
        print("    - reviewText: Texto de la review")
        print("    - summary: Resumen de la review")
        print("    - unixReviewTime: Timestamp")

        print("\nPróximos pasos:")
        print("1. Procesar los datos JSON")
        print("2. Crear matriz de ratings usuario-producto")
        print("3. Entrenar sistemas de recomendación")
        print(f"4. Los datos están en: {json_path}")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error descargando el archivo: {e}")
        print("\nAlternativas para obtener el dataset:")
        print("1. Descargar manualmente de: http://jmcauley.ucsd.edu/data/amazon/")
        print("2. Buscar en Kaggle: 'Amazon Fashion Reviews'")
        print("3. Usar un subset más pequeño para pruebas")
        raise

if __name__ == "__main__":
    download_fashion_dataset()
