#!/usr/bin/env python3
"""
Descargador de Dataset - Amazon Fashion Reviews
Descarga el dataset de reviews de moda desde Hugging Face o genera datos de prueba como fallback
"""

import sys
import io
import json
import gzip
from pathlib import Path
import requests
from tqdm import tqdm
import random
from datetime import datetime

# Configurar codificación UTF-8 para Windows
if sys.platform == 'win32' and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# Importar configuración centralizada
from config import (
    DATASET_FILE,
    DATASET_GZIP_FILE,
    AMAZON_REVIEWS_URL,
    HUGGING_FACE_DATASET,
    RANDOM_STATE
)

# URLs de descarga
HUGGING_FACE_URL = f"https://huggingface.co/datasets/{HUGGING_FACE_DATASET}"

def print_header(text):
    """Imprime un encabezado decorativo"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_step(text):
    """Imprime un paso del proceso"""
    print(f"\n📍 {text}")

def print_success(text):
    """Imprime mensaje de éxito"""
    print(f"✅ {text}")

def print_error(text):
    """Imprime mensaje de error"""
    print(f"❌ {text}")

def print_warning(text):
    """Imprime mensaje de advertencia"""
    print(f"⚠️  {text}")

def check_existing_dataset():
    """Verifica si el dataset ya existe"""
    if DATASET_FILE.exists():
        # Contar líneas del archivo
        try:
            with open(DATASET_FILE, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            file_size_mb = DATASET_FILE.stat().st_size / (1024 * 1024)
            print_success(f"Dataset ya existe: {DATASET_FILE.name}")
            print(f"  - Líneas: {line_count:,}")
            print(f"  - Tamaño: {file_size_mb:.2f} MB")
            return True
        except Exception as e:
            print_warning(f"No se pudo leer el dataset existente: {e}")
            return False
    return False

def download_from_huggingface():
    """Intenta descargar desde Hugging Face usando la librería datasets"""
    print_step("Intentando descargar desde Hugging Face...")

    try:
        from datasets import load_dataset
        print("  Cargando dataset desde Hugging Face...")

        # Intentar descargar el dataset
        dataset = load_dataset(HUGGING_FACE_DATASET, split="train", streaming=False)

        print(f"  Dataset cargado: {len(dataset)} reviews")

        # Guardar como JSON línea por línea
        with open(DATASET_FILE, 'w', encoding='utf-8') as f:
            for i, item in enumerate(dataset):
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                if (i + 1) % 100000 == 0:
                    print(f"    Guardadas {i + 1:,} líneas...")

        print_success(f"Dataset descargado y guardado en {DATASET_FILE.name}")
        return True

    except ImportError:
        print_warning("La librería 'datasets' no está instalada")
        print("  Instálala con: pip install datasets")
        return False
    except Exception as e:
        print_error(f"Error descargando desde Hugging Face: {e}")
        return False

def download_from_amazon_reviews():
    """Intenta descargar desde Amazon Review Data (UCSD)"""
    print_step("Intentando descargar desde Amazon Review Data...")

    try:
        print(f"  Descargando desde: {AMAZON_REVIEWS_URL}")
        print("  (Este archivo es grande, puede tomar tiempo)")

        response = requests.get(AMAZON_REVIEWS_URL, stream=True, timeout=60)
        response.raise_for_status()

        # Obtener tamaño total
        total_size = int(response.headers.get('content-length', 0))

        # Descargar el archivo .gz
        with open(DATASET_GZIP_FILE, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Descargando") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print_success(f"Archivo descargado: {DATASET_GZIP_FILE.name}")

        # Descomprimir
        print_step("Descomprimiendo archivo...")
        with gzip.open(DATASET_GZIP_FILE, 'rt', encoding='utf-8') as f_in:
            with open(DATASET_FILE, 'w', encoding='utf-8') as f_out:
                for i, line in enumerate(f_in):
                    f_out.write(line)
                    if (i + 1) % 100000 == 0:
                        print(f"  Procesadas {i + 1:,} líneas...")

        print_success(f"Dataset descomprimido: {DATASET_FILE.name}")

        # Eliminar archivo comprimido para ahorrar espacio
        DATASET_GZIP_FILE.unlink()
        print_success("Archivo .gz eliminado")

        return True

    except requests.RequestException as e:
        print_error(f"Error de red descargando desde Amazon: {e}")
        return False
    except Exception as e:
        print_error(f"Error descomprimiendo dataset: {e}")
        return False

def generate_test_dataset(num_reviews=5000):
    """Genera un dataset de prueba sintético para testing rápido"""
    print_step(f"Generando dataset de prueba con {num_reviews:,} reviews...")

    try:
        random.seed(RANDOM_STATE)

        # Generar datos sintéticos
        num_users = max(200, num_reviews // 25)  # Aproximadamente 25 reviews por usuario
        num_products = max(500, num_reviews // 10)  # Aproximadamente 10 reviews por producto

        user_ids = [f"user_{i:05d}" for i in range(num_users)]
        product_ids = [f"B{random.randint(100000000, 999999999):09d}" for _ in range(num_products)]

        reviews = []
        base_time = int(datetime(2020, 1, 1).timestamp())

        for i in range(num_reviews):
            review = {
                "reviewerID": random.choice(user_ids),
                "asin": random.choice(product_ids),
                "overall": round(random.uniform(1, 5), 1),
                "reviewText": f"Review #{i+1} - Sample review text",
                "summary": f"Sample summary #{i+1}",
                "unixReviewTime": base_time + random.randint(0, 365*24*3600),
                "verified": random.choice([True, False])
            }
            reviews.append(review)

        # Guardar como JSON línea por línea
        with open(DATASET_FILE, 'w', encoding='utf-8') as f:
            for review in reviews:
                json.dump(review, f, ensure_ascii=False)
                f.write('\n')

        file_size_mb = DATASET_FILE.stat().st_size / (1024 * 1024)
        print_success(f"Dataset de prueba generado: {DATASET_FILE.name}")
        print(f"  - Reviews: {num_reviews:,}")
        print(f"  - Usuarios: {num_users:,}")
        print(f"  - Productos: {num_products:,}")
        print(f"  - Tamaño: {file_size_mb:.2f} MB")

        return True

    except Exception as e:
        print_error(f"Error generando dataset de prueba: {e}")
        return False

def validate_dataset():
    """Valida el dataset generado"""
    print_step("Validando dataset...")

    try:
        if not DATASET_FILE.exists():
            print_error("El archivo del dataset no existe")
            return False

        # Leer primeras líneas
        reviews = []
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 5:  # Leer solo primeras 5 líneas
                    reviews.append(json.loads(line))
                else:
                    break

        if not reviews:
            print_error("El dataset está vacío")
            return False

        # Verificar campos requeridos
        required_fields = {'reviewerID', 'asin', 'overall'}
        first_review = reviews[0]

        if not required_fields.issubset(first_review.keys()):
            print_error(f"Campos faltantes. Se esperaban: {required_fields}")
            return False

        print_success("Dataset validado correctamente")

        # Mostrar estadísticas rápidas
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)

        file_size_mb = DATASET_FILE.stat().st_size / (1024 * 1024)
        print(f"\n📊 Estadísticas del dataset:")
        print(f"  - Líneas (reviews): {line_count:,}")
        print(f"  - Tamaño: {file_size_mb:.2f} MB")
        print(f"  - Campos: {', '.join(first_review.keys())}")

        return True

    except json.JSONDecodeError as e:
        print_error(f"Error decodificando JSON: {e}")
        return False
    except Exception as e:
        print_error(f"Error validando dataset: {e}")
        return False

def main():
    """Ejecuta el descargador con estrategia de fallback"""

    print_header("📥 DESCARGADOR DE DATASET - AMAZON FASHION REVIEWS")

    # Verificar si ya existe
    if check_existing_dataset():
        response = input("\n¿Descargar de nuevo? (s/n): ").strip().lower()
        if response != 's':
            print("\nUsando dataset existente.")
            return validate_dataset()

    # Estrategia de descarga con fallbacks
    strategies = [
        ("Hugging Face", download_from_huggingface),
        ("Amazon Review Data", download_from_amazon_reviews),
        ("Dataset de Prueba", lambda: generate_test_dataset(5000))
    ]

    for name, download_func in strategies:
        print(f"\n{'='*70}")
        print(f"Estrategia: {name}")
        print(f"{'='*70}")

        if download_func():
            print("\n" + "=" * 70)
            if validate_dataset():
                print_success("¡Descarga completada exitosamente!")
                print("\n📋 Próximos pasos:")
                print("  1. Ejecuta: python main.py")
                print("  2. O ejecuta: streamlit run web/app.py")
                print("\n" + "=" * 70)
                return True

    print_error("\n⚠️  No se pudo descargar el dataset con ninguna estrategia")
    print("Por favor, intenta:")
    print("  - Verificar tu conexión a internet")
    print("  - Instalar la librería: pip install datasets")
    print("  - Instalar: pip install -r requirements.txt")

    return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
