import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Asegurar acceso al backend
sys.path.append('src')
try:
    from hybrid_engine import HybridEngine
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from hybrid_engine import HybridEngine

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Financial Intelligence Unit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ACADÉMICO ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Merriweather', serif; }
    .metric-box { background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 15px; text-align: center; border-radius: 5px; }
    .observation-box { background-color: #fff; border-left: 4px solid #2c3e50; padding: 20px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); font-style: italic; }
    .prescription-box { background-color: #f0f7fb; border-left: 4px solid #007bff; padding: 20px; margin: 20px 0; }
    .risk-alert { background-color: #fff5f5; border-left: 4px solid #dc3545; padding: 15px; color: #721c24; }
    </style>
""", unsafe_allow_html=True)

# --- CARGA DEL MOTOR ---
@st.cache_resource
def load_engine():
    return HybridEngine()

try:
    engine = load_engine()
    # Usamos try/except por si la lista de usuarios no carga al inicio
    if hasattr(engine, 'users_raw'):
        users_list = engine.users_raw['user_id'].unique()[:50]
    else:
        users_list = ["U000001"] # Fallback
except Exception as e:
    st.error(f"Error crítico: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.markdown("## Configuración Experimental")
st.sidebar.markdown("---")
selected_user = st.sidebar.selectbox("Sujeto de Análisis", users_list)
top_k = st.sidebar.slider("Umbral (K)", 3, 10, 5)

# --- LÓGICA ---
user_data = engine.users_raw[engine.users_raw['user_id'] == selected_user].iloc[0]
recommendations = engine.predict_hybrid(selected_user, top_k=top_k)

def get_radar_data(u_data):
    return pd.DataFrame({
        'Métrica': ['Solvencia', 'Fiabilidad', 'Experiencia', 'Estabilidad'],
        'Valor': [
            min(u_data['monthly_income'] / 50000, 1.0),
            min(u_data['credit_score'] / 850, 1.0),
            min(u_data['age'] / 80, 1.0),
            1.0 if u_data['employment_status'] in ['Salaried', 'Self-Employed'] else 0.5
        ]
    })

# --- REPORTE VISUAL ---
st.title("Informe Analítico de Propensión Financiera")
st.markdown(f"**Sujeto:** {selected_user} | **Segmento:** Retail Banking")

tab1, tab2, tab3, tab4 = st.tabs(["I. Perfilamiento", "II. Metodología", "III. Resultados", "IV. Estrategia"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f'''
        <div class="metric-box"><h2>{int(user_data['credit_score'])}</h2><small>FICO Score</small></div>
        <br>
        <div class="metric-box"><h2>${user_data['monthly_income']:,.2f}</h2><small>Ingreso</small></div>
        ''', unsafe_allow_html=True)
    with col2:
        radar_df = get_radar_data(user_data)
        fig = px.line_polar(radar_df, r='Valor', theta='Métrica', line_close=True, range_r=[0,1])
        fig.update_traces(fill='toself', line_color='#2c3e50')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Modelo Híbrido (NCF + SVD)")
    st.latex(r'\hat{y}_{u,i} = f(P^T v_u^U + Q^T v_i^I)')
    st.markdown("Se aplican reglas de negocio determinísticas (Basel III compliance).")

with tab3:
    if recommendations:
        rec_df = pd.DataFrame(recommendations, columns=['Producto', 'Score'])
        fig = px.bar(rec_df, x='Score', y='Producto', orientation='h', title="Ranking de Propensión")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        fig.update_traces(marker_color='#2c3e50')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="risk-alert">Bloqueo Total de Riesgo activado.</div>', unsafe_allow_html=True)

with tab4:
    score = user_data['credit_score']
    if score < 650:
        title = "ESTRATEGIA DE RECUPERACIÓN"
        plan = "Suspensión de crédito. Ofrecer producto garantizado."
    else:
        title = "ESTRATEGIA DE FIDELIZACIÓN"
        plan = "Upselling a productos Platinum. Asignar ejecutivo."
    
    st.markdown(f'''
    <div class="prescription-box">
        <h4>{title}</h4>
        <p>{plan}</p>
    </div>
    ''', unsafe_allow_html=True)
