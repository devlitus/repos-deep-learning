"""
Script para descargar el dataset MovieLens 100K
"""
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_movielens(output_dir='.\\data\\raw'):
    """
    Descarga y descomprime el dataset MovieLens 100K
    
    Args:
        output_dir: Directorio donde se guardará el dataset
    """
    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # URL del dataset MovieLens 100K
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = os.path.join(output_dir, "ml-100k.zip")
    
    print("📥 Descargando MovieLens 100K...")
    print(f"URL: {url}")
    
    # Descargar el archivo
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as file, tqdm(
        desc="Descargando",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    print(f"✅ Descarga completada: {zip_path}")
    
    # Descomprimir el archivo
    print("\n📂 Descomprimiendo archivos...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"✅ Archivos descomprimidos en: {output_dir}/ml-100k/")
    
    # Mostrar archivos descargados
    print("\n📄 Archivos principales:")
    ml_dir = os.path.join(output_dir, "ml-100k")
    important_files = ['u.data', 'u.item', 'u.user', 'README']
    
    for file in important_files:
        file_path = os.path.join(ml_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {file}: {size:.2f} KB")
    
    # Eliminar el archivo zip
    os.remove(zip_path)
    print(f"\n🗑️  Archivo zip eliminado para ahorrar espacio")
    
    print("\n" + "="*50)
    print("🎉 ¡Dataset descargado exitosamente!")
    print("="*50)
    print("\nPróximos pasos:")
    print("1. Explorar los datos con pandas")
    print("2. Hacer análisis exploratorio")
    print(f"3. Los datos están en: {ml_dir}")

if __name__ == "__main__":
    download_movielens()