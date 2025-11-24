FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema básicas
RUN apt-get update && apt-get install -y build-essential

# ESTRATEGIA SENIOR: Instalar PyTorch versión CPU (Ultraligero)
# Esto evita descargar los gigas de drivers Nvidia
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Instalar el resto de librerías (separado para aprovechar caché)
RUN pip install pandas numpy scikit-learn matplotlib seaborn plotly faker fastapi uvicorn streamlit

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
