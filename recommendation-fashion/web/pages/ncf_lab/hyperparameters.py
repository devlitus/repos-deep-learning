"""
Módulo de configuración de hiperparámetros NCF
UI para selección de arquitectura y parámetros de entrenamiento
"""
import streamlit as st


def render_hyperparameters(ncf_data):
    """Renderizar UI de configuración de hiperparámetros y retornar dict de configuración"""
    st.subheader("⚙️ Configuración de Hiperparámetros")
    st.caption(
        "Ajusta estos controles para cambiar cómo aprende la red neuronal. "
        "Pasa el ratón sobre el **?** de cada control para ver una explicación."
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        embedding_dim = st.select_slider(
            "📐 Tamaño de Representación",
            options=[16, 32, 64, 128, 256, 512, 1024],
            value=64,
            help="¿Con cuántos números describimos a cada usuario y producto? "
                 "Más números = más detalle, pero necesita más datos para aprender bien. "
                 "Piensa en ello como la resolución de una foto: más píxeles = más detalle."
        )

        architecture = st.selectbox(
            "🏗️ Tamaño del Cerebro",
            options=[
                "Pequeña [64, 32]",
                "Mediana [128, 64, 32]",
                "Grande [256, 128, 64]",
                "Muy Grande [512, 256, 128, 64]",
                "Personalizada"
            ],
            index=1,
            help="El 'cerebro' de la red tiene varias capas de neuronas. "
                 "Más capas y más neuronas = puede aprender patrones más complejos, "
                 "pero también tarda más y puede memorizar en vez de aprender."
        )

    with col_b:
        dropout = st.slider(
            "🎲 Olvido Aleatorio (Dropout)",
            min_value=0.0,
            max_value=0.7,
            value=0.2,
            step=0.05,
            help="En cada ronda, este porcentaje de neuronas se 'apaga' al azar. "
                 "Esto obliga al modelo a no depender demasiado de neuronas específicas "
                 "y aprende patrones más generales. Como estudiar tapando partes del libro."
        )

        use_batch_norm = st.toggle(
            "📊 Normalización (Batch Norm)",
            value=True,
            help="Mantiene los números dentro de la red en un rango controlado. "
                 "Esto hace que el aprendizaje sea más estable y rápido. "
                 "Casi siempre es buena idea tenerlo activado."
        )

    with col_c:
        learning_rate = st.select_slider(
            "🚶 Velocidad de Aprendizaje",
            options=[0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01],
            value=0.001,
            help="¿Qué tan grandes son los 'pasos' que da el modelo para aprender? "
                 "Muy rápido (0.01) = puede pasarse de largo y no encontrar la solución. "
                 "Muy lento (0.0001) = aprende bien pero tarda mucho. "
                 "0.001 suele ser un buen punto de partida."
        )

        optimizer_choice = st.selectbox(
            "🔧 Estrategia de Aprendizaje",
            options=["AdamW (Recomendado)", "Adam", "SGD + Momentum"],
            index=0,
            help="El algoritmo que decide cómo ajustar la red en cada paso. "
                 "AdamW: inteligente y con protección contra memorización. "
                 "Adam: versión clásica sin protección extra. "
                 "SGD: el más simple, pero puede funcionar bien con paciencia."
        )

    # Capas personalizadas
    arch_map = {
        "Pequeña [64, 32]": [64, 32],
        "Mediana [128, 64, 32]": [128, 64, 32],
        "Grande [256, 128, 64]": [256, 128, 64],
        "Muy Grande [512, 256, 128, 64]": [512, 256, 128, 64],
    }

    if architecture == "Personalizada":
        custom_layers_str = st.text_input(
            "Capas personalizadas (separadas por coma):",
            value="128, 64, 32",
            help="Ejemplo: 256, 128, 64"
        )
        try:
            hidden_layers = [int(x.strip()) for x in custom_layers_str.split(',')]
        except ValueError:
            st.warning("Formato inválido. Usando [128, 64, 32]")
            hidden_layers = [128, 64, 32]
    else:
        hidden_layers = arch_map[architecture]

    st.divider()

    # Segunda fila de hiperparámetros
    col_d, col_e, col_f, col_g = st.columns(4)

    with col_d:
        weight_decay = st.select_slider(
            "⚖️ Penalización por Complejidad",
            options=[0.0, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1],
            value=1e-5,
            format_func=lambda x: f"{x:.0e}" if x > 0 else "0 (sin penalización)",
            help="Penaliza al modelo si sus pesos internos crecen demasiado. "
                 "Esto evita que el modelo se vuelva demasiado 'seguro' de sí mismo. "
                 "Valores bajos (1e-5) son un buen comienzo. Valores altos (>1e-2) para fuerte regularización."
        )

    with col_e:
        batch_size = st.select_slider(
            "📦 Datos por Paso",
            options=[64, 128, 256, 512, 1024],
            value=256,
            help="¿Cuántos ejemplos ve el modelo antes de actualizar lo aprendido? "
                 "Más ejemplos = aprendizaje más estable pero más lento. "
                 "Menos ejemplos = más rápido pero más ruidoso."
        )

    with col_f:
        epochs = st.slider(
            "🔄 Rondas Máximas",
            min_value=3,
            max_value=50,
            value=15,
            help="¿Cuántas veces puede el modelo repasar todos los datos? "
                 "Más rondas = más oportunidades de aprender, pero puede memorizar. "
                 "El sistema puede parar antes si detecta que ya no mejora."
        )

    with col_g:
        patience = st.slider(
            "⏳ Paciencia",
            min_value=2,
            max_value=10,
            value=3,
            help="¿Cuántas rondas sin mejorar esperamos antes de parar? "
                 "Poca paciencia (2) = para rápido, puede perder mejoras lentas. "
                 "Mucha paciencia (8+) = da más tiempo, pero puede memorizar."
        )

    st.sidebar.markdown("### 🔧 Ajustes Avanzados")

    gradient_clip = st.sidebar.slider(
        "✂️ Límite de Gradiente",
        min_value=0.5,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Limita los 'saltos' que da el modelo al aprender. "
             "Evita que el entrenamiento se desestabilice por correcciones demasiado grandes."
    )

    use_scheduler = st.sidebar.toggle(
        "📉 Reducción Automática de Velocidad",
        value=True,
        help="Si el modelo se estanca, reduce automáticamente la velocidad de aprendizaje "
             "para hacer ajustes más finos. Como cuando afinas un instrumento: primero giras "
             "la clavija rápido, luego ajustas despacio."
    )

    # Calcular parámetros del modelo
    total_emb_params = (ncf_data['n_users'] + ncf_data['n_products']) * embedding_dim
    mlp_params = 0
    prev_dim = embedding_dim * 2
    for h in hidden_layers:
        mlp_params += prev_dim * h + h  # weights + biases
        if use_batch_norm:
            mlp_params += h * 2  # gamma + beta
        prev_dim = h
    mlp_params += prev_dim * 1 + 1  # output layer
    total_params = total_emb_params + mlp_params

    with st.expander("📋 Resumen de Configuración", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Arquitectura:**
            - Embeddings: {embedding_dim} dims
            - Capas: {hidden_layers}
            - Dropout: {dropout}
            - Batch Norm: {'✅' if use_batch_norm else '❌'}
            - Total parámetros: **{total_params:,}**
            """)
        with col2:
            st.markdown(f"""
            **Entrenamiento:**
            - LR: {learning_rate}
            - Optimizador: {optimizer_choice.split(' ')[0]}
            - Weight Decay: {weight_decay:.0e}
            - Batch Size: {batch_size}
            - Epochs: {epochs} (patience: {patience})
            - Gradient Clip: {gradient_clip}
            - LR Scheduler: {'✅' if use_scheduler else '❌'}
            """)

        st.markdown(f"""
        **Datos:**
        - Usuarios: {ncf_data['n_users']:,} | Productos: {ncf_data['n_products']:,}
        - Train: {len(ncf_data['train_df']):,} ({len(ncf_data['train_df'])/ncf_data['total_interactions']*100:.1f}%)
        - Validation: {len(ncf_data['val_df']):,} ({len(ncf_data['val_df'])/ncf_data['total_interactions']*100:.1f}%)
        - Test: {len(ncf_data['test_df']):,} ({len(ncf_data['test_df'])/ncf_data['total_interactions']*100:.1f}%)
        """)

        with st.expander("ℹ️ ¿Para qué sirve cada conjunto de datos?"):
            st.markdown("""
            **🎓 Metodología Train/Validation/Test:**

            - **Train (70%)**: El modelo aprende SOLO de estos datos. Ajusta sus pesos aquí.
            - **Validation (15%)**: Durante el entrenamiento, evaluamos aquí para:
                - Ajustar hiperparámetros (learning rate, momentum, etc.)
                - Decidir cuándo parar (early stopping)
                - Activar el LR Scheduler
            - **Test (15%)**: Se usa UNA SOLA VEZ al FINAL para medir el rendimiento real.

            **⚠️ Regla de oro**: El test set NUNCA debe influir en decisiones de entrenamiento.
            Si el modelo "viera" el test durante el entrenamiento, estaríamos **haciendo trampa** y
            reportaríamos un error artificialmente bajo.

            **✅ En esta implementación:**
            - Durante el entrenamiento solo verás Train RMSE y Validation RMSE
            - El Test RMSE se calcula UNA vez al terminar el entrenamiento
            - Así garantizamos una evaluación justa y sin sesgos
            """)

    # Retornar diccionario de hiperparámetros
    return {
        'embedding_dim': embedding_dim,
        'hidden_layers': hidden_layers,
        'dropout': dropout,
        'use_batch_norm': use_batch_norm,
        'learning_rate': learning_rate,
        'optimizer_choice': optimizer_choice,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'epochs': epochs,
        'patience': patience,
        'gradient_clip': gradient_clip,
        'use_scheduler': use_scheduler,
        'total_params': total_params
    }
