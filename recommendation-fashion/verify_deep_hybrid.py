"""
Script de verificación rápida para el Deep Hybrid System
Verifica instalación de dependencias y configuración básica
"""

import sys
import io
from pathlib import Path

# Configurar UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

def check_dependency(name, import_name=None):
    """Verifica si un paquete está instalado"""
    if import_name is None:
        import_name = name

    try:
        __import__(import_name)
        return True, "✅"
    except ImportError:
        return False, "❌"

def check_pytorch_cuda():
    """Verifica PyTorch y disponibilidad de CUDA"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return True, f"✅ v{version} (CUDA: {gpu_name}, {gpu_memory:.1f}GB)"
        else:
            return True, f"⚠️  v{version} (solo CPU - será lento)"
    except ImportError:
        return False, "❌ No instalado"

def main():
    print("\n" + "=" * 80)
    print("  🔍 VERIFICACIÓN DE DEPENDENCIAS - DEEP HYBRID SYSTEM")
    print("=" * 80)

    # Dependencias críticas
    print("\n📦 Dependencias Críticas:")
    print("-" * 80)

    deps_critical = [
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
    ]

    all_critical_ok = True
    for name, import_name in deps_critical:
        ok, status = check_dependency(name, import_name)
        print(f"  {status} {name:20s}", end="")

        if ok:
            try:
                module = __import__(import_name)
                if hasattr(module, '__version__'):
                    print(f" (v{module.__version__})")
                else:
                    print()
            except:
                print()
        else:
            print(f" → pip install {import_name}")
            all_critical_ok = False

    # PyTorch (Deep Learning)
    print("\n🧠 Deep Learning:")
    print("-" * 80)

    pytorch_ok, pytorch_status = check_pytorch_cuda()
    print(f"  {pytorch_status} PyTorch")

    if not pytorch_ok:
        print("    → Instalar:")
        print("      CPU: pip install torch torchvision")
        print("      GPU (CUDA 11.8): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("      GPU (CUDA 12.1): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    # Surprise (Collaborative Filtering)
    print("\n🤝 Collaborative Filtering:")
    print("-" * 80)

    surprise_ok, surprise_status = check_dependency("Surprise", "surprise")
    print(f"  {surprise_status} Surprise")

    if not surprise_ok:
        print("    → pip install scikit-surprise")

    # Verificar archivos del proyecto
    print("\n📁 Archivos del Proyecto:")
    print("-" * 80)

    files_to_check = [
        ("src/deep_hybrid_recommender.py", "Módulo Deep Hybrid"),
        ("config.py", "Configuración"),
        ("train_deep_hybrid.py", "Script de entrenamiento"),
        ("DEEP_HYBRID_GUIDE.md", "Guía de uso"),
    ]

    all_files_ok = True
    for filepath, description in files_to_check:
        full_path = PROJECT_DIR / filepath
        if full_path.exists():
            print(f"  ✅ {description:30s} → {filepath}")
        else:
            print(f"  ❌ {description:30s} → {filepath} (no encontrado)")
            all_files_ok = False

    # Dataset
    print("\n💾 Dataset:")
    print("-" * 80)

    import config

    if config.DATASET_FILE.exists():
        size_mb = config.DATASET_FILE.stat().st_size / 1024 / 1024
        print(f"  ✅ Amazon Fashion Reviews ({size_mb:.1f} MB)")
        dataset_ok = True
    else:
        print(f"  ❌ Dataset no encontrado")
        print(f"    → python download_fashion.py")
        dataset_ok = False

    # Verificar configuración
    print("\n⚙️  Configuración:")
    print("-" * 80)

    print(f"  Embedding size: {config.DL_EMBEDDING_SIZE}")
    print(f"  Hidden layers: {config.DL_HIDDEN_LAYERS}")
    print(f"  Batch size: {config.DL_BATCH_SIZE}")
    print(f"  Epochs: {config.DL_EPOCHS}")
    print(f"  Learning rate: {config.DL_LEARNING_RATE}")
    print(f"  Dropout: {config.DL_DROPOUT_RATE}")

    # Resumen
    print("\n" + "=" * 80)
    print("  📊 RESUMEN")
    print("=" * 80)

    all_ok = all_critical_ok and pytorch_ok and surprise_ok and all_files_ok and dataset_ok

    if all_ok:
        print("\n  ✅ TODO LISTO - Puedes entrenar el Deep Hybrid System")
        print("\n  🚀 Ejecuta:")
        print("     python train_deep_hybrid.py")
        print("     o")
        print("     python main.py")
    else:
        print("\n  ⚠️  FALTAN DEPENDENCIAS O ARCHIVOS")
        print("\n  📋 Pasos a seguir:")

        if not all_critical_ok:
            print("     1. pip install -r requirements.txt")

        if not pytorch_ok:
            print("     2. pip install torch torchvision")

        if not surprise_ok:
            print("     3. pip install scikit-surprise")

        if not dataset_ok:
            print("     4. python download_fashion.py")

        if not all_files_ok:
            print("     5. Verificar integridad del proyecto")

    # Verificar GPU (bonus)
    if pytorch_ok:
        try:
            import torch
            if torch.cuda.is_available():
                print("\n  💡 TIP: GPU detectada - El entrenamiento será 5-10x más rápido")
            else:
                print("\n  💡 TIP: Sin GPU - Considera reducir DL_EPOCHS en config.py para pruebas rápidas")
        except:
            pass

    print("\n" + "=" * 80)
    print("  📖 Documentación completa: DEEP_HYBRID_GUIDE.md")
    print("=" * 80 + "\n")

    return all_ok

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
