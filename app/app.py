# =============================================================================
# app/app.py
# DataScope Solutions — Bank Marketing Prediction Demo
# Interactive interface to test the trained model
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to path so we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="DataScope Solutions — Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# Professional styling matching the presentation palette
# =============================================================================

st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1A2744 0%, #2C5F8A 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        margin: 0;
    }
    .main-header p {
        color: #CADCFC;
        margin: 0.5rem 0 0 0;
    }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-top: 4px solid #E8734A;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1A2744;
    }
    .kpi-label {
        color: #8C9AB1;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* Result boxes */
    .result-positive {
        background: #E8F5EE;
        border-left: 5px solid #4CAF7D;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-negative {
        background: #FDE8E6;
        border-left: 5px solid #E8734A;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Sidebar */
    .sidebar-section {
        background: #F4F6FA;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL
# =============================================================================

@st.cache_resource
def load_model():
    """
    Load the trained model pipeline from disk.
    @st.cache_resource ensures the model is loaded only once
    across all user sessions — critical for performance.
    
    Returns:
        Fitted sklearn Pipeline (preprocessor + classifier)
    """
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "best_model_random_forest.pkl"
    )
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.info("Please run the training notebook first to generate the model.")
        st.stop()
    return joblib.load(model_path)


model = load_model()

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>🏦 DataScope Solutions</h1>
    <p>Predictor de Suscripción a Depósitos a Plazo — Demo Interactivo</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — Client profile inputs
# =============================================================================

st.sidebar.markdown("## 👤 Perfil del Cliente")
st.sidebar.markdown("---")

# --- Personal data ---
st.sidebar.markdown("### 📋 Datos personales")

age = st.sidebar.slider(
    "Edad", 
    min_value=18, max_value=95, value=40,
    help="Edad del cliente en años"
)

job = st.sidebar.selectbox(
    "Tipo de trabajo",
    options=["admin.", "blue-collar", "entrepreneur", "housemaid",
             "management", "retired", "self-employed", "services",
             "student", "technician", "unemployed", "unknown"],
    index=0,
    help="Categoría profesional del cliente"
)

marital = st.sidebar.selectbox(
    "Estado civil",
    options=["divorced", "married", "single", "unknown"],
    index=1
)

education = st.sidebar.selectbox(
    "Nivel educativo",
    options=["illiterate", "basic.4y", "basic.6y", "basic.9y",
             "high.school", "professional.course", "university.degree", "unknown"],
    index=6
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💳 Situación financiera")

default = st.sidebar.selectbox(
    "¿Tiene créditos en mora?",
    options=["no", "yes", "unknown"],
    index=0
)

housing = st.sidebar.selectbox(
    "¿Tiene hipoteca?",
    options=["no", "yes", "unknown"],
    index=1
)

loan = st.sidebar.selectbox(
    "¿Tiene préstamo personal?",
    options=["no", "yes", "unknown"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 Datos del contacto")

contact = st.sidebar.selectbox(
    "Tipo de contacto",
    options=["cellular", "telephone"],
    index=0
)

month = st.sidebar.selectbox(
    "Mes del último contacto",
    options=["jan", "feb", "mar", "apr", "may", "jun",
             "jul", "aug", "sep", "oct", "nov", "dec"],
    index=4
)

day_of_week = st.sidebar.selectbox(
    "Día de la semana",
    options=["mon", "tue", "wed", "thu", "fri"],
    index=0
)

campaign = st.sidebar.slider(
    "Nº de contactos esta campaña",
    min_value=1, max_value=20, value=2,
    help="Número de veces contactado en la campaña actual"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Historial de campañas")

previously_contacted = st.sidebar.checkbox(
    "¿Fue contactado en campaña anterior?",
    value=False
)

if previously_contacted:
    pdays = st.sidebar.slider(
        "Días desde último contacto anterior",
        min_value=1, max_value=30, value=7
    )
    previous = st.sidebar.slider(
        "Nº contactos en campañas previas",
        min_value=1, max_value=7, value=1
    )
    poutcome = st.sidebar.selectbox(
        "Resultado de campaña anterior",
        options=["failure", "success"],
        index=0
    )
else:
    pdays   = 999
    previous = 0
    poutcome = "nonexistent"

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 Contexto macroeconómico")

emp_var_rate = st.sidebar.slider(
    "Tasa de variación del empleo",
    min_value=-3.4, max_value=1.4, value=1.1, step=0.1
)

cons_price_idx = st.sidebar.slider(
    "Índice de precios al consumo",
    min_value=92.0, max_value=95.0, value=93.8, step=0.1
)

cons_conf_idx = st.sidebar.slider(
    "Índice de confianza del consumidor",
    min_value=-51.0, max_value=-26.0, value=-36.4, step=0.1
)

euribor3m = st.sidebar.slider(
    "Euribor 3 meses (%)",
    min_value=0.6, max_value=5.1, value=4.9, step=0.1,
    help="Tipo de interés interbancario europeo a 3 meses"
)

nr_employed = st.sidebar.slider(
    "Empleados en la economía (miles)",
    min_value=4963.0, max_value=5228.0, value=5191.0, step=1.0
)

# =============================================================================
# FEATURE ENGINEERING (same as training pipeline)
# =============================================================================

# Apply the same feature engineering from the training phase
was_previously_contacted = 1 if previously_contacted else 0
pdays_clean              = pdays if previously_contacted else 0
campaign_capped          = min(campaign, 14)  # Same cap as training (p99)

# =============================================================================
# BUILD INPUT DATAFRAME
# Must match EXACTLY the columns and types expected by the pipeline
# =============================================================================

input_data = pd.DataFrame([{
    "age"              : age,
    "job"              : job,
    "marital"          : marital,
    "education"        : education,
    "default"          : default,
    "housing"          : housing,
    "loan"             : loan,
    "contact"          : contact,
    "month"            : month,
    "day_of_week"      : day_of_week,
    "campaign_capped"  : campaign_capped,
    "pdays_clean"      : pdays_clean,
    "previous"         : previous,
    "poutcome"         : poutcome,
    "emp.var.rate"     : emp_var_rate,
    "cons.price.idx"   : cons_price_idx,
    "cons.conf.idx"    : cons_conf_idx,
    "euribor3m"        : euribor3m,
    "nr.employed"      : nr_employed,
    "was_previously_contacted": was_previously_contacted,
}])

# =============================================================================
# PREDICTION
# =============================================================================

# --- Predict button ---
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button(
        "🔮 Predecir probabilidad de suscripción",
        use_container_width=True,
        type="primary"
    )

st.markdown("---")

# =============================================================================
# RESULTS SECTION
# =============================================================================

if predict_btn:
    try:
        # Get probability and class prediction
        probability = model.predict_proba(input_data)[0][1]  # P(yes)
        prediction  = model.predict(input_data)[0]            # 0 or 1

        # --- KPI row ---
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{probability*100:.1f}%</div>
                <div class="kpi-label">Probabilidad de suscripción</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            pred_label = "✅ Suscribirá" if prediction == 1 else "❌ No suscribirá"
            pred_color = "#4CAF7D" if prediction == 1 else "#E8734A"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{pred_color};font-size:1.6rem">
                    {pred_label}
                </div>
                <div class="kpi-label">Predicción del modelo</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            if probability >= 0.6:
                priority = "🔴 ALTA"
                p_color  = "#E8734A"
            elif probability >= 0.3:
                priority = "🟡 MEDIA"
                p_color  = "#F39C12"
            else:
                priority = "🟢 BAJA"
                p_color  = "#4CAF7D"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{p_color};font-size:1.8rem">
                    {priority}
                </div>
                <div class="kpi-label">Prioridad de contacto</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")  # spacing

        # --- Result interpretation ---
        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-title">✅ Cliente de alta probabilidad</div>
                <p>El modelo predice que este cliente <strong>suscribirá el depósito a plazo</strong> 
                con una probabilidad del <strong>{probability*100:.1f}%</strong>.</p>
                <p><strong>Recomendación:</strong> Incluir en la lista prioritaria de contacto 
                de la próxima campaña. Contactar preferentemente por teléfono móvil.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-title">❌ Cliente de baja probabilidad</div>
                <p>El modelo predice que este cliente <strong>no suscribirá</strong> 
                con una probabilidad de suscripción del <strong>{probability*100:.1f}%</strong>.</p>
                <p><strong>Recomendación:</strong> No incluir en la lista prioritaria. 
                Considerar para campañas de otros productos o próximas temporadas.</p>
            </div>
            """, unsafe_allow_html=True)

        # --- Probability gauge ---
        st.markdown("### 📊 Probabilidad de suscripción")
        st.progress(float(probability))
        st.caption(f"El modelo asigna una probabilidad de suscripción del {probability*100:.1f}% a este cliente.")

        # --- Key factors (simplified explanation) ---
        st.markdown("### 🔍 Factores clave del perfil")
        col_a, col_b = st.columns(2)

        with col_a:
            st.info(f"""
            **Euribor actual**: {euribor3m:.1f}%
            {'↓ Tipos bajos → mayor receptividad a depósitos' if euribor3m < 3.0 else '↑ Tipos altos → menor ventaja comparativa del depósito'}
            """)
            st.info(f"""
            **Canal de contacto**: {contact}
            {'✅ Móvil: 3x más conversión que teléfono fijo' if contact == 'cellular' else '⚠️ Teléfono fijo: menor tasa de conversión histórica'}
            """)

        with col_b:
            st.info(f"""
            **Historial de campaña**: {poutcome}
            {'🏆 Éxito previo: 5x más propenso a suscribir' if poutcome == 'success' else '📊 Sin historial previo o fracaso anterior'}
            """)
            st.info(f"""
            **Contactos esta campaña**: {campaign}
            {'✅ Pocas llamadas: buen candidato' if campaign <= 3 else '⚠️ Muchos contactos: señal de resistencia — reconsiderar'}
            """)

        # --- Input summary ---
        with st.expander("📋 Ver resumen completo del perfil del cliente"):
            col_x, col_y = st.columns(2)
            with col_x:
                st.dataframe(
                    input_data[["age","job","marital","education",
                                "default","housing","loan","contact"]].T.rename(
                        columns={0: "Valor"}),
                    use_container_width=True
                )
            with col_y:
                st.dataframe(
                    input_data[["month","day_of_week","campaign_capped",
                                "poutcome","euribor3m","cons.price.idx",
                                "cons.conf.idx","nr.employed"]].T.rename(
                        columns={0: "Valor"}),
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"Error al realizar la predicción: {str(e)}")
        st.exception(e)

else:
    # Default state — instructions
    st.markdown("""
    ### 👈 Cómo usar este demo
    
    1. **Configura el perfil** del cliente usando los controles de la barra lateral izquierda
    2. **Pulsa el botón** "Predecir probabilidad de suscripción"
    3. **Interpreta el resultado**: probabilidad, predicción y recomendación de contacto
    
    ---
    
    ### 📊 Sobre el modelo
    
    | Característica | Detalle |
    |---|---|
    | **Algoritmo** | Random Forest (200 árboles) |
    | **Dataset** | 41.188 clientes — Bank Marketing UCI |
    | **ROC-AUC** | 0.811 |
    | **Recall** | 62.5% |
    | **F1-Score** | 52.2% |
    | **Metodología** | CRISP-DM + Pipeline anti-leakage |
    
    ---
    
    ### ⚠️ Aviso importante
    
    Este es un modelo de demostración entrenado con datos históricos de un banco portugués (2008-2013).
    Para uso en producción, el modelo debería re-entrenarse con datos actualizados del banco cliente.
    """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8C9AB1; font-size:0.85rem; padding: 1rem 0;">
    DataScope Solutions · María Isabel Durango · 
    <a href="https://github.com/MarialsaDurango/datascope-bank-marketing" 
       style="color:#2C5F8A;">GitHub del proyecto</a>
</div>
""", unsafe_allow_html=True)