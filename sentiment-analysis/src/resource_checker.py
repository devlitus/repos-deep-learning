"""
Resource Checker - Verificación automática de recursos NLP
===========================================================

Este módulo verifica que todos los recursos necesarios (NLTK, spaCy)
estén instalados y los descarga automáticamente si faltan.
"""

import sys
import subprocess


def check_nltk_resources():
    """
    Verifica si los recursos de NLTK están instalados.
    Retorna True si todos están disponibles, False si falta alguno.
    """
    import nltk

    required_resources = [
        'tokenizers/punkt_tab',
        'tokenizers/punkt',
        'corpora/stopwords',
        'corpora/wordnet',
        'taggers/averaged_perceptron_tagger',
        'corpora/omw-1.4',
    ]

    missing = []
    for resource in required_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(resource)

    return len(missing) == 0, missing


def check_spacy_model(model_name='en_core_web_sm'):
    """
    Verifica si el modelo de spaCy está instalado.
    Retorna True si está disponible, False si falta.
    """
    try:
        import spacy
        spacy.load(model_name)
        return True, None
    except OSError:
        return False, model_name
    except Exception as e:
        return False, str(e)


def auto_download_resources(quiet=False):
    """
    Descarga automáticamente recursos faltantes.

    Args:
        quiet (bool): Si True, minimiza la salida de texto

    Returns:
        bool: True si todos los recursos se descargaron exitosamente
    """
    if not quiet:
        print("\n🔍 Verificando recursos NLP...")

    # Verificar NLTK
    nltk_ok, missing_nltk = check_nltk_resources()

    if not nltk_ok:
        if not quiet:
            print(f"\n⚠️  Faltan {len(missing_nltk)} recursos de NLTK")
            print("📥 Descargando automáticamente...")

        try:
            import nltk

            resources_to_download = [
                'punkt_tab',
                'punkt',
                'stopwords',
                'wordnet',
                'averaged_perceptron_tagger',
                'omw-1.4',
            ]

            for resource in resources_to_download:
                if not quiet:
                    print(f"   Descargando {resource}...")
                nltk.download(resource, quiet=quiet)

            if not quiet:
                print("   ✅ Recursos de NLTK descargados")

        except Exception as e:
            print(f"\n❌ Error al descargar recursos de NLTK: {e}")
            print("\n💡 Ejecuta manualmente: python install_resources.py")
            return False
    else:
        if not quiet:
            print("   ✅ Recursos de NLTK: OK")

    # Verificar spaCy
    spacy_ok, missing_spacy = check_spacy_model()

    if not spacy_ok:
        if not quiet:
            print(f"\n⚠️  Falta modelo de spaCy: {missing_spacy}")
            print("📥 Descargando automáticamente...")
            print("   (Esto puede tomar 1-2 minutos...)")

        try:
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                stdout=subprocess.DEVNULL if quiet else None,
                stderr=subprocess.DEVNULL if quiet else None
            )

            if not quiet:
                print("   ✅ Modelo de spaCy descargado")

        except Exception as e:
            print(f"\n❌ Error al descargar modelo de spaCy: {e}")
            print("\n💡 Ejecuta manualmente: python -m spacy download en_core_web_sm")
            return False
    else:
        if not quiet:
            print("   ✅ Modelo de spaCy: OK")

    if not quiet:
        print("\n✅ Todos los recursos están listos\n")

    return True


def ensure_resources_available():
    """
    Asegura que todos los recursos estén disponibles.
    Si faltan, los descarga automáticamente.
    Si la descarga falla, muestra instrucciones y termina el programa.
    """
    print("\n" + "=" * 70)
    print("🔧 VERIFICANDO DEPENDENCIAS NLP")
    print("=" * 70)

    # Verificar recursos
    nltk_ok, _ = check_nltk_resources()
    spacy_ok, _ = check_spacy_model()

    if nltk_ok and spacy_ok:
        print("\n✅ Todos los recursos NLP están instalados")
        return True

    # Intentar descargar automáticamente
    print("\n⚠️  Algunos recursos faltan. Intentando descarga automática...")
    print("   (Primera ejecución: esto puede tomar 2-5 minutos)")

    success = auto_download_resources(quiet=False)

    if not success:
        print("\n" + "=" * 70)
        print("❌ ERROR: No se pudieron descargar los recursos")
        print("=" * 70)
        print("\n💡 SOLUCIÓN:")
        print("   Ejecuta manualmente el script de instalación:")
        print("\n   python install_resources.py")
        print("\n   O descarga los recursos uno por uno:")
        print("   python -m nltk.downloader punkt_tab stopwords wordnet")
        print("   python -m spacy download en_core_web_sm")
        print("\n" + "=" * 70)

        sys.exit(1)

    return True
