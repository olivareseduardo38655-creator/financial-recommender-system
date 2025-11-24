SISTEMA DE RECOMENDACIÓN HÍBRIDO FINANCIERO | MOTOR RISK-AWARE

DESCRIPCIÓN EJECUTIVA

Este repositorio contiene la implementación de un motor de inferencia analítica diseñado para el sector de Banca Minorista (Retail Banking). El sistema resuelve el problema de asignación ineficiente de productos de crédito mediante una arquitectura híbrida que combina Filtrado Colaborativo Neuronal (NCF) con Factorización de Matrices (SVD).

A diferencia de los sistemas de recomendación convencionales orientados puramente a la conversión, este motor integra una Capa de Gobierno de Riesgos (Risk Governance Layer). Esta capa asegura que todas las ofertas presentadas al cliente cumplan estrictamente con las normativas de solvencia y capacidad de pago (alineado a principios de Basilea III), mitigando la exposición al incumplimiento (Default Risk) antes de la etapa de originación.

OBJETIVOS TÉCNICOS

El desarrollo de este sistema abordó y resolvió los siguientes retos de ingeniería de datos:

Modelado Híbrido: Implementación de un ensamble ponderado que mitiga el problema de arranque en frío (Cold Start) mediante SVD y captura no-linealidades complejas mediante Deep Learning.

Ingeniería de Características Vectorial: Transformación de variables demográficas y transaccionales en espacios latentes densos para su ingesta en redes neuronales.

Contenerización y Reproducibilidad: Empaquetado total del entorno de ejecución (librerías, drivers y código) mediante Docker, eliminando dependencias del sistema operativo anfitrión.

Inferencia con Restricciones (Constrained Optimization): Aplicación de lógica determinística post-predicción para filtrar candidatos de alto riesgo financiero.

ARQUITECTURA DEL SISTEMA

El flujo de datos sigue una tubería ETL estricta, culminando en una inferencia en tiempo real orquestada por Docker.

graph LR
    A[Raw Data Lake] -->|Limpieza y Vectorización| B(Feature Engineering Pipeline)
    B --> C{Entrenamiento de Modelos}
    C -->|Relaciones Lineales| D[SVD - Matrix Factorization]
    C -->|Relaciones No-Lineales| E[NCF - Deep Neural Network]
    D --> F[Hybrid Inference Engine]
    E --> F
    F -->|Scores Brutos| G{Capa de Gobierno de Riesgos}
    G -->|Filtrado Normativo| H[Recomendaciones Finales]
    H --> I[Dashboard Ejecutivo Streamlit]


ESTRUCTURA DEL REPOSITORIO

FINANCIAL_RECOMMENDER_SYSTEM/
├── src/
│   ├── hybrid_engine.py       # Orquestador de lógica de negocio y fusión de modelos
│   ├── train_ncf.py           # Script de entrenamiento de Red Neuronal (PyTorch)
│   ├── train_collaborative.py # Script de entrenamiento SVD (Scikit-Learn)
│   ├── dashboard.py           # Interfaz gráfica para análisis ejecutivo
│   ├── features.py            # Pipelines de transformación de datos
│   └── data_generation.py     # Generador estocástico de datos sintéticos
├── data/                      # Almacenamiento temporal (Data Lake simulado)
├── models_store/              # Persistencia de artefactos binarios (.pkl, .pth)
├── Dockerfile                 # Definición de infraestructura como código
└── requirements.txt           # Dependencias del entorno Python


GUÍA DE INSTALACIÓN Y DESPLIEGUE

Este proyecto ha sido diseñado para ser agnóstico al sistema operativo mediante Docker.

Opción A: Despliegue en Contenedor (Recomendado)

Construcción de la Imagen:
Compilar el entorno con las dependencias optimizadas para CPU.

docker build -t fintech-recsys:v1 .


Ejecución del Servicio:
Desplegar el dashboard en el puerto 8501.

docker run -p 8501:8501 fintech-recsys:v1


Acceso:
Navegar a http://localhost:8501 en su explorador web.

Opción B: Ejecución Local

Requiere Python 3.10+ y entorno virtual configurado.

pip install -r requirements.txt
python src/data_generation.py
python src/features.py
python src/train_collaborative.py
python src/train_ncf.py
streamlit run src/dashboard.py


LÓGICA DE NEGOCIO Y GOBIERNO DE DATOS

El sistema aplica las siguientes reglas determinísticas sobre los scores de propensión generados por la IA:

Variable de Control

Condición Lógica

Acción del Sistema

Justificación Financiera

Credit Score (FICO)

Menor a 720

Bloqueo de productos Tier 'Platinum'

Mitigación de riesgo en productos de alta línea de crédito sin garantía.

Capacidad de Pago

Anualidad mayor al 30% del Ingreso

Eliminación del candidato

Prevención de sobreendeudamiento según normativa de riesgo crediticio.

Historial

Sin interacción previa

Fallback a SVD (Cold Start)

Asegurar recomendación relevante basada en clústeres demográficos.

Autor: Lead Data Scientist & Financial Engineer
