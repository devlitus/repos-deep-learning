"""
Script de Instalación de Recursos NLP
======================================

Este script descarga automáticamente todos los recursos necesarios
para el proyecto de análisis de sentimientos.

Ejecutar UNA VEZ después de instalar requirements.txt:
    python install_resources.py

O será ejecutado automáticamente por main.py si detecta recursos faltantes.
"""

import sys
import subprocess


def download_nltk_resources():
    """Descarga recursos de NLTK necesarios para el proyecto."""
    print("\n" + "=" * 70)
    print("📥 DESCARGANDO RECURSOS DE NLTK")
    print("=" * 70)

    import nltk

    # Lista de recursos necesarios
    resources = [
        ('punkt_tab', 'Tokenizador de palabras (nuevo formato)'),
        ('punkt', 'Tokenizador de palabras (formato legacy)'),
        ('stopwords', 'Palabras vacías (stop words)'),
        ('wordnet', 'Base de datos léxica WordNet'),
        ('averaged_perceptron_tagger', 'Etiquetador gramatical (POS tagger)'),
        ('omw-1.4', 'Open Multilingual Wordnet'),
    ]

    downloaded = []
    failed = []

    for resource, description in resources:
        try:
            print(f"\n📦 Descargando '{resource}'...")
            print(f"   ({description})")
            nltk.download(resource, quiet=False)
            downloaded.append(resource)
            print(f"   ✅ Completado")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed.append(resource)

    print("\n" + "-" * 70)
    print(f"✅ NLTK: {len(downloaded)}/{len(resources)} recursos descargados")

    if failed:
        print(f"⚠️  Fallos: {', '.join(failed)}")
        return False

    return True


def download_spacy_model():
    """Descarga modelo de spaCy en inglés."""
    print("\n" + "=" * 70)
    print("📥 DESCARGANDO MODELO DE SPACY")
    print("=" * 70)

    model_name = 'en_core_web_sm'

    try:
        # Verificar si ya está instalado
        import spacy
        try:
            nlp = spacy.load(model_name)
            print(f"\n✅ Modelo '{model_name}' ya está instalado")
            print(f"   Versión: {nlp.meta['version']}")
            return True
        except OSError:
            # No está instalado, descargar
            print(f"\n📦 Descargando '{model_name}'...")
            print(f"   (Modelo de procesamiento de lenguaje natural en inglés)")
            print(f"   (Tamaño: ~12 MB)")

            # Ejecutar comando de descarga
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model_name],
                stdout=sys.stdout,
                stderr=sys.stderr
            )

            print(f"\n✅ Modelo '{model_name}' descargado exitosamente")
            return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error al descargar modelo de spaCy: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return False


def verify_installation():
    """Verifica que todos los recursos estén instalados correctamente."""
    print("\n" + "=" * 70)
    print("🔍 VERIFICANDO INSTALACIÓN")
    print("=" * 70)

    all_good = True

    # Verificar NLTK
    print("\n📚 Verificando recursos de NLTK...")
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords, wordnet

        # Test tokenización
        test_text = "This is a test sentence."
        tokens = word_tokenize(test_text)
        print(f"   ✅ Tokenización: {tokens}")

        # Test stopwords
        stop_words = set(stopwords.words('english'))
        print(f"   ✅ Stopwords: {len(stop_words)} palabras cargadas")

        # Test wordnet
        syns = wordnet.synsets("good")
        print(f"   ✅ WordNet: {len(syns)} synsets para 'good'")

    except Exception as e:
        print(f"   ❌ Error en NLTK: {e}")
        all_good = False

    # Verificar spaCy
    print("\n🧠 Verificando modelo de spaCy...")
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp("This is a test.")
        tokens = [token.text for token in doc]
        print(f"   ✅ spaCy: {tokens}")
        print(f"   ✅ Modelo cargado: en_core_web_sm v{nlp.meta['version']}")
    except Exception as e:
        print(f"   ❌ Error en spaCy: {e}")
        all_good = False

    return all_good


def main():
    """Ejecuta la instalación completa de recursos."""
    print("\n" + "🎬" * 35)
    print(" INSTALACIÓN DE RECURSOS NLP")
    print("🎬" * 35)

    print("\nEste script descargará:")
    print("  1. Recursos de NLTK (~30-50 MB)")
    print("  2. Modelo de spaCy en inglés (~12 MB)")
    print("\nTiempo estimado: 2-5 minutos (depende de tu conexión)")

    # Descargar recursos
    nltk_ok = download_nltk_resources()
    spacy_ok = download_spacy_model()

    # Verificar instalación
    if nltk_ok and spacy_ok:
        print("\n" + "=" * 70)
        verification_ok = verify_installation()

        if verification_ok:
            print("\n" + "=" * 70)
            print("🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
            print("=" * 70)
            print("\n✅ Todos los recursos están listos para usar")
            print("\n💡 Próximos pasos:")
            print("   1. Ejecuta el pipeline completo: python main.py")
            print("   2. O abre los notebooks: jupyter notebook")
            return True
        else:
            print("\n⚠️  Instalación completada con advertencias")
            print("   Algunos recursos pueden no funcionar correctamente")
            return False
    else:
        print("\n" + "=" * 70)
        print("❌ ERROR EN LA INSTALACIÓN")
        print("=" * 70)
        print("\n⚠️  Algunos recursos no se descargaron correctamente")
        print("\n💡 Intenta:")
        print("   1. Verificar tu conexión a internet")
        print("   2. Ejecutar nuevamente: python install_resources.py")
        print("   3. O descargar manualmente:")
        print("      python -m nltk.downloader punkt_tab stopwords wordnet")
        print("      python -m spacy download en_core_web_sm")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
