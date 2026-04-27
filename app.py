import streamlit as st
import joblib
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sheet Pile Service Life Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* Root variables */
:root {
    --steel:   #1a2332;
    --rust:    #c0392b;
    --amber:   #f39c12;
    --concrete:#8395a7;
    --light:   #ecf0f1;
    --card:    #1e2d3d;
    --border:  #2c3e50;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: var(--steel) !important;
    color: var(--light) !important;
}

/* Hide default header */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0d1b2a 100%);
    min-height: 100vh;
}

/* Main title */
.main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 0.12em;
    color: var(--light);
    line-height: 1;
    margin-bottom: 0;
}
.main-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--concrete);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 4px;
}
.accent-bar {
    height: 4px;
    background: linear-gradient(90deg, var(--rust), var(--amber), transparent);
    border-radius: 2px;
    margin: 14px 0 28px 0;
}

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    padding: 6px 12px;
    border-radius: 3px;
    margin-bottom: 14px;
    font-weight: 600;
}
.header-corrosion {
    background: rgba(192, 57, 43, 0.18);
    border-left: 3px solid var(--rust);
    color: #e74c3c;
}
.header-lateral {
    background: rgba(52, 152, 219, 0.15);
    border-left: 3px solid #3498db;
    color: #5dade2;
}
.header-soil {
    background: rgba(39, 174, 96, 0.15);
    border-left: 3px solid #27ae60;
    color: #58d68d;
}

/* Card containers */
.param-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 18px;
    margin-bottom: 16px;
}

/* Input labels */
.stNumberInput label, .stSelectbox label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--concrete) !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* Input fields */
.stNumberInput input {
    background: #0f1923 !important;
    border: 1px solid var(--border) !important;
    color: var(--light) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
    border-radius: 4px !important;
}
.stNumberInput input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(243,156,18,0.2) !important;
}

/* Select box */
.stSelectbox > div > div {
    background: #0f1923 !important;
    border: 1px solid var(--border) !important;
    color: var(--light) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--rust), #96281b) !important;
    color: white !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.4rem !important;
    letter-spacing: 0.2em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 14px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 10px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e74c3c, var(--rust)) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(192,57,43,0.4) !important;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #0d2137, #1a2d45);
    border: 1px solid var(--amber);
    border-radius: 10px;
    padding: 28px 32px;
    text-align: center;
    margin-top: 20px;
    position: relative;
    overflow: hidden;
}
.result-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--amber), transparent);
}
.result-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    color: var(--amber);
    line-height: 1;
    letter-spacing: 0.05em;
}
.result-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--concrete);
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-top: 4px;
}
.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--concrete);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.interval-box {
    background: rgba(52,152,219,0.1);
    border: 1px solid rgba(52,152,219,0.3);
    border-radius: 6px;
    padding: 14px 20px;
    text-align: center;
    margin-top: 14px;
}
.interval-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    color: #5dade2;
    font-weight: 600;
}
.interval-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--concrete);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--rust), var(--amber)) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Importance badge */
.imp-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 10px;
    background: rgba(243,156,18,0.15);
    color: var(--amber);
    border: 1px solid rgba(243,156,18,0.3);
    margin-left: 6px;
    vertical-align: middle;
}

/* Model info bar */
.model-info {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--concrete);
    letter-spacing: 0.1em;
    padding: 8px 14px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 24px;
}
.model-info span { color: var(--amber); }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("sheet_pile_rfr_model__2_.pkl")

data  = load_model()
model = data['model']


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Sheet Pile Service Life</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Random Forest Regression Predictor · Structural Assessment Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="model-info">
  Model: <span>sheet_pile_rfr_model__2_.pkl</span> &nbsp;·&nbsp;
  Trees: <span>400</span> &nbsp;·&nbsp;
  Max Depth: <span>6</span> &nbsp;·&nbsp;
  Features: <span>18</span> &nbsp;·&nbsp;
  Bootstrap Sample: <span>80%</span> &nbsp;·&nbsp;
  Target: <span>Service Life (years)</span>
</div>
""", unsafe_allow_html=True)


# ── Input columns ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1], gap="large")

# ── CORROSION ─────────────────────────────────────────────────────────────────
with col1:
    st.markdown('<div class="section-header header-corrosion">🔴 Corrosion Parameters — 46.65% importance</div>', unsafe_allow_html=True)

    flange_thick = st.number_input(
        "Flange Initial Thick. (mm)  ▸ 24.00%",
        min_value=1.0, max_value=50.0, value=10.0, step=0.5,
        help="Steel reserve before corrosion penetration — most important feature"
    )
    corrosion_rate = st.number_input(
        "Factored Corrosion Rate (mm/yr)  ▸ 14.64%",
        min_value=0.001, max_value=10.0, value=0.05, step=0.001, format="%.3f",
        help="Annual steel loss rate. Service Life ≈ Flange Thick ÷ Corrosion Rate"
    )
    chloride = st.number_input(
        "Mean Chloride Deposition  ▸ 8.01%",
        min_value=0.0, max_value=1000.0, value=50.0, step=1.0,
        help="Primary electrochemical corrosion accelerant"
    )
    sulfate = st.number_input(
        "Mean Sulfate Deposition  ▸ 4.12%",
        min_value=0.0, max_value=1000.0, value=30.0, step=1.0,
        help="Secondary chemical degradation driver"
    )
    humidity = st.number_input(
        "Mean Humidity (%)  ▸ 2.40%",
        min_value=0.0, max_value=100.0, value=75.0, step=1.0
    )
    annual_temp = st.number_input(
        "Mean Annual Temp (°C)  ▸ 4.35%",
        min_value=-20.0, max_value=55.0, value=27.0, step=0.5
    )

# ── LATERAL EARTH PRESSURE ────────────────────────────────────────────────────
with col2:
    st.markdown('<div class="section-header header-lateral">🔵 Lateral Earth Pressure — 24.03% importance</div>', unsafe_allow_html=True)

    surcharge = st.number_input(
        "Surcharge (kPa)  ▸ 7.22%",
        min_value=0.0, max_value=500.0, value=20.0, step=1.0,
        help="Imposed surface load increasing lateral pressure"
    )
    embedment = st.number_input(
        "Embedment Depth (m)  ▸ 6.89%",
        min_value=0.0, max_value=40.0, value=5.0, step=0.1,
        help="Passive resistance depth — structural stability"
    )
    gwt = st.number_input(
        "GWT Elev. from Surface (m)  ▸ 5.92%",
        min_value=0.0, max_value=30.0, value=2.0, step=0.1,
        help="Groundwater table elevation — hydrostatic pressure component"
    )
    lat_stress = st.number_input(
        "Lateral Effective Stress (kPa)  ▸ 5.00%",
        min_value=0.0, max_value=500.0, value=80.0, step=1.0
    )
    vert_stress = st.number_input(
        "Vertical Effective Stress (kPa)  ▸ 4.13%",
        min_value=0.0, max_value=600.0, value=100.0, step=1.0
    )

# ── SOIL PROPERTIES ───────────────────────────────────────────────────────────
with col3:
    st.markdown('<div class="section-header header-soil">🟢 Soil Properties — 16.54% importance</div>', unsafe_allow_html=True)

    porosity = st.number_input(
        "Porosity  ▸ 2.60%",
        min_value=0.0, max_value=1.0, value=0.40, step=0.01, format="%.3f"
    )
    cohesion = st.number_input(
        "Effective Cohesion (kPa)  ▸ 2.43%",
        min_value=0.0, max_value=200.0, value=10.0, step=0.5
    )
    void_ratio = st.number_input(
        "Void Ratio  ▸ 2.17%",
        min_value=0.0, max_value=5.0, value=0.67, step=0.01, format="%.3f"
    )
    spt = st.number_input(
        "SPT Corrected N-Values  ▸ 1.68%",
        min_value=0.0, max_value=100.0, value=15.0, step=1.0
    )
    friction_ang = st.number_input(
        "Internal Friction Angle (deg)  ▸ 1.57%",
        min_value=0.0, max_value=50.0, value=30.0, step=0.5
    )
    sat_unit_wt = st.number_input(
        "Saturated Unit Weight (kN/m³)  ▸ 1.44%",
        min_value=10.0, max_value=30.0, value=19.0, step=0.1
    )
    soil_type = st.selectbox(
        "Soil Type  ▸ 1.42%",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "0 — Clay",
            1: "1 — Sand",
            2: "2 — Silt",
            3: "3 — Gravel",
            4: "4 — Rock"
        }[x],
        help="Label-encoded (same encoding used during training)"
    )


# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("---")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("⟶  PREDICT SERVICE LIFE", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict:
    # Exact feature order from model.feature_names_in_:
    # [0]  Lateral Effective Stress (kPa)
    # [1]  Vertical Effective Stress (kPa)
    # [2]  Surcharge (kPa)
    # [3]  Embedment Depth (m)
    # [4]  Groundwater Table Elev. from Surface (m)
    # [5]  SPT Corrected N-Values
    # [6]  Internal Friction Angle (deg)
    # [7]  Effective Cohesion (kPa)
    # [8]  Saturated Unit Weight (kN/m³)
    # [9]  Factored Corrosion Rate (mm/yr)
    # [10] Flange Initial Thick. (mm)
    # [11] Mean Chloride Deposition
    # [12] Mean Sulfate Deposition
    # [13] Mean Humidity (%)
    # [14] Mean Annual Temp (°C)
    # [15] Void Ratio
    # [16] Porosity
    # [17] Soil_Type
    X = np.array([[
        lat_stress, vert_stress, surcharge, embedment, gwt,
        spt, friction_ang, cohesion, sat_unit_wt,
        corrosion_rate, flange_thick, chloride, sulfate,
        humidity, annual_temp, void_ratio, porosity, soil_type
    ]])

    pred       = model.predict(X)[0]
    tree_preds = [t.predict(X)[0] for t in model.estimators_]
    lo         = np.percentile(tree_preds, 5)
    hi         = np.percentile(tree_preds, 95)
    std_dev    = np.std(tree_preds)

    # ── Result display ────────────────────────────────────────
    _, res_col, _ = st.columns([1, 2, 1])
    with res_col:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Service Life</div>
            <div class="result-value">{pred:.1f}</div>
            <div class="result-unit">Years</div>
            <div class="interval-box" style="margin-top:18px;">
                <div class="interval-value">{lo:.1f} – {hi:.1f} yrs</div>
                <div class="interval-label">90% Prediction Interval · ±{std_dev:.1f} std dev</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar vs 100-year reference
        pct = min(pred / 100.0, 1.0)
        st.markdown(f"<br><small style='font-family:IBM Plex Mono;font-size:0.65rem;color:#8395a7;letter-spacing:0.2em;'>SERVICE LIFE vs 100-YEAR REFERENCE</small>", unsafe_allow_html=True)
        st.progress(pct)
        st.markdown(f"<p style='font-family:IBM Plex Mono;font-size:0.75rem;color:#f39c12;text-align:right;'>{pred:.1f} / 100 yrs ({pct*100:.0f}%)</p>", unsafe_allow_html=True)

    # ── Corrosion check ───────────────────────────────────────
    theoretical = flange_thick / corrosion_rate if corrosion_rate > 0 else float('inf')
    st.markdown("---")
    check_col1, check_col2, check_col3 = st.columns(3)
    with check_col1:
        st.metric(
            "Theoretical Max (T ÷ CR)",
            f"{theoretical:.1f} yrs",
            help="Flange Thickness ÷ Corrosion Rate — physical upper bound"
        )
    with check_col2:
        st.metric(
            "RFR Prediction",
            f"{pred:.1f} yrs",
            delta=f"{pred - theoretical:.1f} vs theoretical"
        )
    with check_col3:
        st.metric(
            "Interval Width",
            f"{hi - lo:.1f} yrs",
            help="Narrower = higher model confidence"
        )
