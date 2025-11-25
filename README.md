# Financial Hybrid Recommender System | Risk-Aware Engine

![Status](https://img.shields.io/badge/Status-Production-success)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## Resumen Ejecutivo

Este sistema es un motor de recomendación de productos financieros de **arquitectura híbrida** diseñado para el sector bancario (Retail Banking). A diferencia de los sistemas tradicionales que solo optimizan el *engagement* (clics), este motor integra una capa de **Gobierno de Riesgos** (Risk Governance Layer) que alinea las ofertas comerciales con la normativa de solvencia del cliente (Basel III compliance).

El sistema resuelve el problema del "Over-Lending" (sobreendeudamiento) al filtrar recomendaciones mediante un análisis vectorial del perfil crediticio del usuario antes de mostrar la oferta.

---

##  Arquitectura del Sistema

El flujo de datos sigue una tubería ETL estricta, culminando en una inferencia en tiempo real orquestada por Docker.

```mermaid
graph LR
    A[Raw Data / Lake] -->|ETL Pipeline| B(Feature Engineering)
    B --> C{Model Training}
    C -->|Matrix Factorization| D[SVD Model]
    C -->|Deep Learning| E[NCF Neural Network]
    D --> F[Hybrid Engine]
    E --> F
    F -->|Raw Scores| G{Risk Logic Layer}
    G -->|Business Rules| H[Final Recommendations]
    H --> I[Streamlit Dashboard]
