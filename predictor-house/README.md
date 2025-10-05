predictor-casas/
│
├── data/
│ ├── raw/ # Datos originales (nunca modificar)
│ │ └── casas.csv
│ └── processed/ # Datos procesados/limpios
│ └── casas_clean.csv
│
├── notebooks/ # Jupyter notebooks para exploración
│ └── 01_exploracion.ipynb
│
├── src/ # Código fuente
│ ├── **init**.py
│ ├── data_loader.py # Cargar y limpiar datos
│ ├── feature_engineering.py # Crear/transformar características
│ ├── model.py # Entrenar y evaluar modelos
│ └── predictor.py # Hacer predicciones
│
├── models/ # Modelos entrenados guardados
│ └── modelo_casas.pkl
│
├── reports/ # Gráficas y reportes
│ └── figures/
│ └── analisis.png
│
├── tests/ # Tests unitarios (opcional)
│ └── test_model.py
│
├── main.py # Script principal
├── requirements.txt # Dependencias
├── config.py # Configuraciones
└── README.md # Documentación
