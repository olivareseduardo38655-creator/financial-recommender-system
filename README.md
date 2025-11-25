Sistema de Recomendación Híbrido FinancieroDescripción del ProyectoEste proyecto implementa un motor de inferencia analítica diseñado específicamente para el sector de Banca Minorista (Retail Banking). El sistema resuelve el problema de la asignación ineficiente de productos de crédito mediante una arquitectura híbrida que combina Filtrado Colaborativo Neuronal (NCF) con Factorización de Matrices (SVD).El diferencial crítico de esta solución es la integración de una Capa de Gobierno de Riesgos (Risk Governance Layer) post-inferencia. Esta capa intercepta las recomendaciones generadas por la inteligencia artificial y aplica filtros determinísticos de solvencia y capacidad de pago, asegurando que cualquier oferta presentada al cliente cumpla con los lineamientos de riesgo crediticio (alineado a principios de Basilea III).Objetivos TécnicosOrquestación Híbrida: Desarrollo de un ensamble ponderado que mitiga el problema de arranque en frío (Cold Start) mediante algoritmos lineales y captura patrones complejos mediante Deep Learning.Ingeniería de Características Vectorial: Transformación de variables demográficas y transaccionales en espacios latentes densos (Embeddings) para su ingesta eficiente.Inferencia Restringida: Diseño de una lógica de optimización que prioriza la salud financiera del cliente sobre la conversión pura, reduciendo la exposición al riesgo de impago (Default Risk).Despliegue Agnóstico: Infraestructura contenerizada mediante Docker para garantizar la reproducibilidad total del entorno de ejecución.Arquitectura del SistemaEl flujo de datos sigue una arquitectura modular controlada por una tubería de procesamiento estricta:graph LR
    A[Data Lake Simulado] -->|ETL Vectorial| B(Feature Engineering)
    B --> C{Entrenamiento Modelos}
    C -->|Matriz Dispersa| D[SVD Factorización]
    C -->|Tensores Densos| E[NCF Red Neuronal]
    D --> F[Motor Híbrido]
    E --> F
    F -->|Scores Brutos| G{Capa de Gobierno de Riesgos}
    G -->|Validación Normativa| H[Recomendaciones Finales]
    H --> I[Dashboard Ejecutivo]
Estructura del RepositorioFINANCIAL_RECOMMENDER_SYSTEM/
├── src/
│   ├── hybrid_engine.py       # Orquestador de lógica de negocio y fusión de modelos
│   ├── train_ncf.py           # Script de entrenamiento de Red Neuronal (PyTorch)
│   ├── train_collaborative.py # Script de entrenamiento SVD (Scikit-Learn)
│   ├── dashboard.py           # Interfaz gráfica para auditoría y visualización
│   ├── features.py            # Pipelines de transformación de datos
│   └── data_generation.py     # Generador estocástico de datos sintéticos
├── data/                      # Almacenamiento temporal (Data Lake simulado)
├── models_store/              # Persistencia de artefactos binarios (.pkl, .pth)
├── Dockerfile                 # Definición de infraestructura
└── requirements.txt           # Dependencias del entorno
Guía de Instalación y DespliegueEste proyecto ha sido diseñado para ser agnóstico al entorno mediante el uso de contenedores Docker.Opción A: Despliegue en Contenedor (Producción)Construcción de la Imagen:docker build -t fintech-recsys:v1 .
Ejecución del Servicio:docker run -p 8501:8501 fintech-recsys:v1
Acceso:Navegar a http://localhost:8501 en el navegador web para acceder al Dashboard.Opción B: Ejecución Local (Desarrollo)Se requiere Python 3.10+ y un entorno virtual configurado.pip install -r requirements.txt
python src/data_generation.py
python src/features.py
python src/train_ncf.py
streamlit run src/dashboard.py
Lógica de Negocio / Gobierno de DatosEl sistema aplica las siguientes reglas de validación financiera sobre los resultados del modelo predictivo:Variable de ControlCondición LógicaAcción del SistemaJustificación FinancieraCredit Score (FICO)Menor a 720 puntosBloqueo de productos Tier 'Platinum'Mitigación de exposición al riesgo en líneas de crédito no garantizadas.Capacidad de PagoAnualidad > 30% del IngresoEliminación del candidatoPrevención de sobreendeudamiento según normativa de crédito responsable.HistorialSin interacción previaFallback a SVD (Cold Start)Asegurar recomendación relevante basada en clústeres demográficos similares.Autor: Lead Data Scientist & Financial Engineer
