# app.py — Premium Accident Severity Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoadSense AI — Accident Severity Predictor",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --surface2:  #181c24;
    --border:    #252a35;
    --accent:    #f97316;
    --accent2:   #3b82f6;
    --accent3:   #10b981;
    --fatal:     #ef4444;
    --serious:   #f59e0b;
    --slight:    #22c55e;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --font-head: 'Bebas Neue', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* ── Base ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

.stApp { background: var(--bg); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
.sidebar-logo {
    font-family: var(--font-head);
    font-size: 2.4rem;
    letter-spacing: 3px;
    color: var(--accent);
    line-height: 1;
    padding: 0.5rem 0 0.2rem;
}
.sidebar-sub {
    font-size: 0.72rem;
    letter-spacing: 4px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.sidebar-stat {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
}
.sidebar-stat span { color: var(--accent); font-weight: 600; }

/* ── Page title ── */
.page-header {
    border-left: 5px solid var(--accent);
    padding: 0.3rem 0 0.3rem 1.2rem;
    margin-bottom: 2rem;
}
.page-header h1 {
    font-family: var(--font-head) !important;
    font-size: 3rem !important;
    letter-spacing: 4px !important;
    color: var(--text) !important;
    margin: 0 !important;
    line-height: 1 !important;
}
.page-header p {
    color: var(--muted) !important;
    font-size: 0.85rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    margin: 0.3rem 0 0 !important;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent);
}
.metric-card.blue::before  { background: var(--accent2); }
.metric-card.green::before { background: var(--accent3); }
.metric-card.red::before   { background: var(--fatal); }
.metric-label { font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
.metric-value { font-family: var(--font-head); font-size: 2.2rem; color: var(--text); line-height: 1; }
.metric-sub   { font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; }

/* ── Section headers ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
}

/* ── Prediction result ── */
.pred-box {
    border-radius: 12px;
    padding: 2rem 2.5rem;
    text-align: center;
    border: 2px solid;
    margin: 1.5rem 0;
}
.pred-box.fatal   { background: rgba(239,68,68,0.08);  border-color: var(--fatal);   }
.pred-box.serious { background: rgba(245,158,11,0.08); border-color: var(--serious); }
.pred-box.slight  { background: rgba(34,197,94,0.08);  border-color: var(--slight);  }
.pred-label { font-family: var(--font-head); font-size: 3.5rem; letter-spacing: 5px; }
.pred-label.fatal   { color: var(--fatal);   }
.pred-label.serious { color: var(--serious); }
.pred-label.slight  { color: var(--slight);  }
.pred-desc { font-size: 0.9rem; color: var(--muted); margin-top: 0.5rem; }

/* ── Probability bars ── */
.prob-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.6rem; }
.prob-name { font-family: var(--font-mono); font-size: 0.78rem; width: 70px; color: var(--muted); }
.prob-bar-bg { flex: 1; height: 10px; background: var(--surface2); border-radius: 5px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 5px; transition: width 0.8s ease; }
.prob-pct { font-family: var(--font-mono); font-size: 0.78rem; width: 44px; text-align: right; }

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.8rem;
}
.risk-high    { background: rgba(239,68,68,0.15);  color: var(--fatal);   border: 1px solid var(--fatal);   }
.risk-medium  { background: rgba(245,158,11,0.15); color: var(--serious); border: 1px solid var(--serious); }
.risk-low     { background: rgba(34,197,94,0.15);  color: var(--slight);  border: 1px solid var(--slight);  }

/* ── Input panel ── */
.input-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Info cards (home page) ── */
.info-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin: 1.5rem 0; }
.info-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem;
}
.info-card-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.info-card-title { font-family: var(--font-head); font-size: 1.3rem; letter-spacing: 2px; color: var(--text); margin-bottom: 0.4rem; }
.info-card-body  { font-size: 0.82rem; color: var(--muted); line-height: 1.6; }

/* ── Severity chip ── */
.chip-fatal   { color: var(--fatal);   background: rgba(239,68,68,0.1);  border: 1px solid var(--fatal);   padding: 2px 10px; border-radius: 20px; font-size:0.75rem; font-family:var(--font-mono); }
.chip-serious { color: var(--serious); background: rgba(245,158,11,0.1); border: 1px solid var(--serious); padding: 2px 10px; border-radius: 20px; font-size:0.75rem; font-family:var(--font-mono); }
.chip-slight  { color: var(--slight);  background: rgba(34,197,94,0.1);  border: 1px solid var(--slight);  padding: 2px 10px; border-radius: 20px; font-size:0.75rem; font-family:var(--font-mono); }

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stSlider > div,
.stNumberInput > div { background: var(--surface2) !important; border-color: var(--border) !important; }
label { color: var(--muted) !important; font-size: 0.8rem !important; letter-spacing: 1px !important; }
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--font-head) !important;
    font-size: 1.1rem !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }
div[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#111318',
    'axes.facecolor':    '#111318',
    'axes.edgecolor':    '#252a35',
    'axes.labelcolor':   '#6b7280',
    'xtick.color':       '#6b7280',
    'ytick.color':       '#6b7280',
    'text.color':        '#e8eaf0',
    'grid.color':        '#1e2330',
    'grid.alpha':        0.5,
    'font.family':       'monospace',
})
COLORS = {'fatal': '#ef4444', 'serious': '#f59e0b', 'slight': '#22c55e'}
PAL3   = [COLORS['fatal'], COLORS['serious'], COLORS['slight']]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_models():
    m  = joblib.load('best_model.pkl')
    sc = joblib.load('scaler.pkl')
    lw = joblib.load('le_weather.pkl')
    lr = joblib.load('le_road.pkl')
    ll = joblib.load('le_light.pkl')
    lv = joblib.load('le_vehicle.pkl')
    fc = joblib.load('feature_cols.pkl')
    return m, sc, lw, lr, ll, lv, fc

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv('unified_accident_data.csv')
    df['severity'] = pd.to_numeric(df.get('severity', df.get('Accident_Severity')), errors='coerce')
    df = df.dropna(subset=['severity'])
    df['severity'] = df['severity'].astype(int)
    df = df[df['severity'].isin([1,2,3])]
    return df

model, scaler, le_weather, le_road, le_light, le_vehicle, feature_cols = load_models()
df = load_data()

SEV_MAP   = {1:'Fatal', 2:'Serious', 3:'Slight'}
SEV_CLASS = {1:'fatal', 2:'serious', 3:'slight'}

NUM_FEATURES = [
    'Speed_limit','Number_of_Vehicles','Number_of_Casualties',
    'hour','avg_driver_age','casualty_per_vehicle'
]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">ROAD<br>SENSE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">AI Safety Analytics</div>', unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Home",
        "🔮  Predict Severity",
        "📊  EDA Dashboard",
        "🤖  Model Intelligence"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">Live Stats</div>', unsafe_allow_html=True)

    total  = len(df)
    fatal  = (df['severity']==1).sum()
    serious= (df['severity']==2).sum()
    slight = (df['severity']==3).sum()

    st.markdown(f'<div class="sidebar-stat">Total Records<br><span>{total:,}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat">Fatal Cases<br><span style="color:#ef4444">{fatal:,} ({fatal/total*100:.1f}%)</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat">Serious Cases<br><span style="color:#f59e0b">{serious:,} ({serious/total*100:.1f}%)</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat">Slight Cases<br><span style="color:#22c55e">{slight:,} ({slight/total*100:.1f}%)</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat">Model Accuracy<br><span>69.9%</span></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE 1: HOME ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if "Home" in page:
    st.markdown("""
    <div class="page-header">
        <h1>ROADSENSE AI</h1>
        <p>Road Accident Severity Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">59,998</div>
            <div class="metric-sub">UK accident dataset</div>
        </div>
        <div class="metric-card blue">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">69.9%</div>
            <div class="metric-sub">Random Forest + SMOTE</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">Features Used</div>
            <div class="metric-value">12</div>
            <div class="metric-sub">Engineered & encoded</div>
        </div>
        <div class="metric-card red">
            <div class="metric-label">Fatal Cases</div>
            <div class="metric-value">646</div>
            <div class="metric-sub">1.08% of dataset</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-grid">
        <div class="info-card">
            <div class="info-card-icon">🔮</div>
            <div class="info-card-title">PREDICT</div>
            <div class="info-card-body">Input real-time accident parameters — time, weather, road type, vehicle — and get instant severity prediction with confidence probabilities.</div>
        </div>
        <div class="info-card">
            <div class="info-card-icon">📊</div>
            <div class="info-card-title">EXPLORE</div>
            <div class="info-card-body">12+ interactive visualisations covering severity distribution, weather impact, hourly patterns, speed limits, driver age, and more.</div>
        </div>
        <div class="info-card">
            <div class="info-card-icon">🤖</div>
            <div class="info-card-title">UNDERSTAND</div>
            <div class="info-card-body">Deep-dive into model internals — feature importance, confusion matrices, class-level metrics, and SMOTE balancing analysis.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Severity Class Guide</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="input-panel" style="border-top:3px solid #ef4444;">
            <div style="font-size:2rem">🔴</div>
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;letter-spacing:3px;color:#ef4444;">FATAL (1)</div>
            <div style="font-size:0.82rem;color:#6b7280;margin-top:0.4rem;">At least one death resulted. Highest priority intervention needed. Represents 1.08% of all accidents.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="input-panel" style="border-top:3px solid #f59e0b;">
            <div style="font-size:2rem">🟡</div>
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;letter-spacing:3px;color:#f59e0b;">SERIOUS (2)</div>
            <div style="font-size:0.82rem;color:#6b7280;margin-top:0.4rem;">Hospital admission required. Significant injury but survivable. Represents 12.3% of all accidents.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="input-panel" style="border-top:3px solid #22c55e;">
            <div style="font-size:2rem">🟢</div>
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;letter-spacing:3px;color:#22c55e;">SLIGHT (3)</div>
            <div style="font-size:0.82rem;color:#6b7280;margin-top:0.4rem;">Minor injuries, no hospitalisation. Most common outcome. Represents 86.6% of all accidents.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:2rem;">Dataset Overview</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5,3))
        counts = df['severity'].value_counts().sort_index()
        bars = ax.bar(['Fatal','Serious','Slight'], counts.values, color=PAL3,
                      edgecolor='#252a35', linewidth=0.8, width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                    f'{val:,}', ha='center', va='bottom', fontsize=9, color='#e8eaf0')
        ax.set_title('Severity Distribution', fontsize=11, pad=10)
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5,3))
        sizes  = counts.values
        explode= (0.08, 0.04, 0)
        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=['Fatal','Serious','Slight'],
            autopct='%1.1f%%', colors=PAL3,
            textprops={'color':'#e8eaf0','fontsize':9},
            wedgeprops={'edgecolor':'#111318','linewidth':2}
        )
        for at in autotexts: at.set_fontsize(8)
        ax.set_title('Class Proportions', fontsize=11, pad=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE 2: PREDICT ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif "Predict" in page:
    st.markdown("""
    <div class="page-header">
        <h1>PREDICT SEVERITY</h1>
        <p>Enter accident parameters to get instant AI prediction</p>
    </div>
    """, unsafe_allow_html=True)

    col_inp, col_out = st.columns([1.1, 0.9], gap="large")

    with col_inp:
        st.markdown('<div class="section-label">🚦 Road & Environment</div>', unsafe_allow_html=True)
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                weather = st.selectbox("Weather Condition", list(le_weather.classes_))
                road    = st.selectbox("Road Type",         list(le_road.classes_))
            with c2:
                light   = st.selectbox("Light Condition",   list(le_light.classes_))
                speed   = st.slider("Speed Limit (mph)", 10, 100, 30, step=5)

        st.markdown('<div class="section-label" style="margin-top:1rem;">🚗 Vehicle & Driver</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            vehicle     = st.selectbox("Vehicle Type", list(le_vehicle.classes_))
            num_vehicles= st.number_input("No. of Vehicles", 1, 10, 1)
        with c2:
            driver_age  = st.slider("Driver Age", 16, 90, 35)
            num_casualties = st.number_input("No. of Casualties", 0, 20, 1)

        st.markdown('<div class="section-label" style="margin-top:1rem;">🕐 Time & Context</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            hour = st.slider("Hour of Day (0–23)", 0, 23, 8)
        with c2:
            urban_rural = st.selectbox("Area Type", ["Urban", "Rural"])

        cas_per_veh = num_casualties / (num_vehicles + 1)
        is_night    = 1 if (hour < 6 or hour > 20) else 0
        is_rush     = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0

        # Context badges
        badges = []
        if is_night:    badges.append("🌙 Night Drive")
        if is_rush:     badges.append("⚡ Rush Hour")
        if speed >= 70: badges.append("💨 High Speed")
        if driver_age < 25: badges.append("🧑 Young Driver")
        if driver_age > 65: badges.append("👴 Senior Driver")
        if weather in ['Rain','Fog','Snow']: badges.append("🌧️ Poor Weather")

        if badges:
            st.markdown("**Risk Flags:** " + " &nbsp; ".join(
                [f'<span style="background:#1e2330;border:1px solid #252a35;border-radius:12px;padding:2px 10px;font-size:0.78rem;">{b}</span>'
                 for b in badges]), unsafe_allow_html=True)

        predict_btn = st.button("⚡  RUN PREDICTION")

    with col_out:
        st.markdown('<div class="section-label">🎯 Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            # Build feature vector
            weather_enc = le_weather.transform([weather])[0]
            road_enc    = le_road.transform([road])[0]
            light_enc   = le_light.transform([light])[0]
            vehicle_enc = le_vehicle.transform([vehicle])[0]

            raw = {
                'weather_enc': weather_enc, 'road_enc': road_enc,
                'light_enc': light_enc,     'vehicle_enc': vehicle_enc,
                'Speed_limit': speed,       'Number_of_Vehicles': num_vehicles,
                'Number_of_Casualties': num_casualties, 'hour': hour,
                'avg_driver_age': driver_age, 'casualty_per_vehicle': cas_per_veh,
                'is_night': is_night,         'is_rush_hour': is_rush
            }
            X_input = pd.DataFrame([raw])[feature_cols]

            # Scale numeric
            X_scaled = X_input.copy()
            X_scaled[NUM_FEATURES] = scaler.transform(X_scaled[NUM_FEATURES])

            proba = model.predict_proba(X_scaled)[0]
            # model classes: 0=Fatal,1=Serious,2=Slight → severity 1,2,3
            pred_class = np.argmax(proba)  # 0,1,2
            pred_sev   = pred_class + 1    # 1,2,3

            sev_name  = SEV_MAP[pred_sev]
            sev_class = SEV_CLASS[pred_sev]
            confidence= proba[pred_class] * 100

            st.markdown(f"""
            <div class="pred-box {sev_class}">
                <div class="pred-label {sev_class}">{sev_name.upper()}</div>
                <div class="pred-desc">Predicted Accident Severity · Severity Class {pred_sev}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;margin-top:0.8rem;">
                    Confidence: <strong>{confidence:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="section-label">Class Probabilities</div>', unsafe_allow_html=True)
            prob_labels = ['Fatal','Serious','Slight']
            prob_colors = ['#ef4444','#f59e0b','#22c55e']
            for lbl, pct, color in zip(prob_labels, proba*100, prob_colors):
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-name">{lbl}</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
                    </div>
                    <div class="prob-pct" style="color:{color};">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Risk level badge
            if pred_sev == 1:
                badge_class, badge_text = "risk-high",   "🔴 HIGH RISK"
            elif pred_sev == 2:
                badge_class, badge_text = "risk-medium", "🟡 MEDIUM RISK"
            else:
                badge_class, badge_text = "risk-low",    "🟢 LOW RISK"
            st.markdown(f'<div class="risk-badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)

            # Input summary
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Input Summary</div>', unsafe_allow_html=True)
            summary = {
                "Weather": weather, "Road": road, "Light": light,
                "Speed (mph)": speed, "Vehicle": vehicle,
                "Vehicles": num_vehicles, "Casualties": num_casualties,
                "Hour": f"{hour:02d}:00", "Driver Age": driver_age,
                "Night": "Yes" if is_night else "No",
                "Rush Hour": "Yes" if is_rush else "No"
            }
            rows = "".join([
                f'<tr><td style="color:#6b7280;padding:3px 12px 3px 0;font-size:0.8rem;">{k}</td>'
                f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;">{v}</td></tr>'
                for k, v in summary.items()
            ])
            st.markdown(f'<table style="width:100%">{rows}</table>', unsafe_allow_html=True)

            # Recommendation
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Recommendation</div>', unsafe_allow_html=True)
            recs = {
                1: "⚠️ **Immediate emergency response required.** Deploy all emergency services. Scene preservation critical for investigation.",
                2: "🚑 **Ambulance dispatch recommended.** Ensure hospital notification. Traffic management needed.",
                3: "🚔 **Standard police attendance.** Exchange details, clear scene safely. Monitor for delayed injury reports."
            }
            st.info(recs[pred_sev])

        else:
            st.markdown("""
            <div style="background:#111318;border:1px dashed #252a35;border-radius:12px;
                        padding:3rem;text-align:center;margin-top:1rem;">
                <div style="font-size:3rem;margin-bottom:1rem;">🔮</div>
                <div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;
                            letter-spacing:4px;color:#252a35;">AWAITING INPUT</div>
                <div style="color:#374151;font-size:0.82rem;margin-top:0.5rem;">
                    Fill in parameters and click RUN PREDICTION
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE 3: EDA DASHBOARD ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif "EDA" in page:
    st.markdown("""
    <div class="page-header">
        <h1>EDA DASHBOARD</h1>
        <p>Exploratory data analysis across all accident dimensions</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1 ──
    st.markdown('<div class="section-label">Distribution & Time Analysis</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = plt.subplots(figsize=(4.5,3.2))
        counts = df['severity'].value_counts().sort_index()
        bars = ax.bar(['Fatal','Serious','Slight'], counts.values, color=PAL3,
                      edgecolor='#1e2330', linewidth=0.8, width=0.55)
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+100,
                    f'{v:,}', ha='center', va='bottom', fontsize=8)
        ax.set_title('Severity Distribution', fontsize=10, pad=8)
        ax.set_ylabel('Count', fontsize=8); ax.grid(axis='y',alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        if 'hour' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            for sev, color, label in zip([1,2,3], PAL3, ['Fatal','Serious','Slight']):
                sub = df[df['severity']==sev].groupby('hour').size()
                ax.plot(sub.index, sub.values, color=color, label=label,
                        linewidth=1.8, marker='o', markersize=3)
            ax.set_title('Accidents by Hour of Day', fontsize=10, pad=8)
            ax.set_xlabel('Hour', fontsize=8); ax.set_ylabel('Count', fontsize=8)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    with c3:
        if 'Day_of_Week' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            day_map = {1:'Sun',2:'Mon',3:'Tue',4:'Wed',5:'Thu',6:'Fri',7:'Sat'}
            df2 = df.copy(); df2['day'] = df2['Day_of_Week'].map(day_map)
            order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            day_c = df2['day'].value_counts().reindex(order)
            ax.bar(order, day_c.values, color='#3b82f6', edgecolor='#1e2330',
                   linewidth=0.8, width=0.6)
            ax.set_title('Accidents by Day of Week', fontsize=10, pad=8)
            ax.set_ylabel('Count', fontsize=8); ax.grid(axis='y',alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Row 2 ──
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Weather, Road & Light Analysis</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        if 'weather' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            wdata = pd.crosstab(df['weather'], df['severity'], normalize='index')
            wdata.columns = ['Fatal','Serious','Slight']
            wdata.plot(kind='bar', stacked=True, ax=ax, color=PAL3, edgecolor='#1e2330', linewidth=0.5)
            ax.set_title('Weather vs Severity (normalised)', fontsize=10, pad=8)
            ax.set_xlabel(''); ax.set_ylabel('Proportion', fontsize=8)
            ax.tick_params(axis='x', rotation=30, labelsize=7)
            ax.legend(fontsize=7, loc='upper right')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        if 'road_type' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            rdata = pd.crosstab(df['road_type'], df['severity'], normalize='index')
            rdata.columns = ['Fatal','Serious','Slight']
            rdata.plot(kind='bar', stacked=True, ax=ax, color=PAL3, edgecolor='#1e2330', linewidth=0.5)
            ax.set_title('Road Type vs Severity', fontsize=10, pad=8)
            ax.set_xlabel(''); ax.set_ylabel('Proportion', fontsize=8)
            ax.tick_params(axis='x', rotation=30, labelsize=7)
            ax.legend(fontsize=7)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    with c3:
        if 'light' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            ldata = pd.crosstab(df['light'], df['severity'], normalize='index')
            ldata.columns = ['Fatal','Serious','Slight']
            ldata.plot(kind='bar', stacked=True, ax=ax, color=PAL3, edgecolor='#1e2330', linewidth=0.5)
            ax.set_title('Light Condition vs Severity', fontsize=10, pad=8)
            ax.set_xlabel(''); ax.set_ylabel('Proportion', fontsize=8)
            ax.tick_params(axis='x', rotation=0, labelsize=8)
            ax.legend(fontsize=7)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Row 3 ──
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Speed, Age & Vehicle Analysis</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        if 'Speed_limit' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            for sev, color, label in zip([1,2,3], PAL3, ['Fatal','Serious','Slight']):
                vals = df[df['severity']==sev]['Speed_limit'].dropna()
                ax.hist(vals, bins=15, alpha=0.6, color=color, label=label, density=True)
            ax.set_title('Speed Limit Distribution by Severity', fontsize=10, pad=8)
            ax.set_xlabel('Speed Limit (mph)', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        if 'avg_driver_age' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            for sev, color, label in zip([1,2,3], PAL3, ['Fatal','Serious','Slight']):
                vals = df[df['severity']==sev]['avg_driver_age'].dropna()
                ax.hist(vals, bins=20, alpha=0.6, color=color, label=label, density=True)
            ax.set_title('Driver Age Distribution by Severity', fontsize=10, pad=8)
            ax.set_xlabel('Average Driver Age', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    with c3:
        if 'primary_vehicle_type' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            vdata = pd.crosstab(df['primary_vehicle_type'], df['severity'], normalize='index')
            vdata.columns = ['Fatal','Serious','Slight']
            vdata.plot(kind='bar', stacked=True, ax=ax, color=PAL3, edgecolor='#1e2330', linewidth=0.5)
            ax.set_title('Vehicle Type vs Severity', fontsize=10, pad=8)
            ax.set_xlabel(''); ax.set_ylabel('Proportion', fontsize=8)
            ax.tick_params(axis='x', rotation=30, labelsize=7)
            ax.legend(fontsize=7)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Row 4 ──
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Correlations & Advanced Patterns</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        corr_cols = [c for c in ['severity','Speed_limit','Number_of_Vehicles',
                     'Number_of_Casualties','hour','avg_driver_age','casualty_per_vehicle']
                     if c in df.columns]
        if len(corr_cols) >= 3:
            fig, ax = plt.subplots(figsize=(4.5,3.5))
            corr = df[corr_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
                        ax=ax, annot_kws={'size':7}, linewidths=0.5,
                        linecolor='#1e2330', cbar_kws={'shrink':0.7})
            ax.set_title('Correlation Heatmap', fontsize=10, pad=8)
            ax.tick_params(labelsize=7)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        if 'hour' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            hour_sev = df.groupby('hour')['severity'].mean()
            ax.fill_between(hour_sev.index, hour_sev.values, alpha=0.3, color='#f97316')
            ax.plot(hour_sev.index, hour_sev.values, color='#f97316', linewidth=2, marker='o', markersize=4)
            ax.axvspan(7,9,   alpha=0.12, color='#3b82f6', label='Morning rush')
            ax.axvspan(17,19, alpha=0.12, color='#8b5cf6', label='Evening rush')
            ax.set_title('Mean Severity by Hour', fontsize=10, pad=8)
            ax.set_xlabel('Hour', fontsize=8); ax.set_ylabel('Mean Severity', fontsize=8)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    with c3:
        if 'casualty_per_vehicle' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            for sev, color, label in zip([1,2,3], PAL3, ['Fatal','Serious','Slight']):
                vals = df[df['severity']==sev]['casualty_per_vehicle'].clip(0,3)
                ax.hist(vals, bins=20, alpha=0.6, color=color, label=label, density=True)
            ax.set_title('Casualties per Vehicle by Severity', fontsize=10, pad=8)
            ax.set_xlabel('Casualties / Vehicle', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Row 5: Bonus charts ──
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Night / Rush Hour & Urban vs Rural</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        if 'is_night' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            night_sev = pd.crosstab(df['is_night'], df['severity'], normalize='index')
            night_sev.index = ['Day','Night']
            night_sev.columns = ['Fatal','Serious','Slight']
            night_sev.plot(kind='bar', stacked=True, ax=ax, color=PAL3, edgecolor='#1e2330', linewidth=0.5)
            ax.set_title('Day vs Night Severity', fontsize=10, pad=8)
            ax.set_xlabel(''); ax.set_ylabel('Proportion', fontsize=8)
            ax.tick_params(axis='x', rotation=0, labelsize=8)
            ax.legend(fontsize=7)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        if 'is_rush_hour' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            rush_sev = pd.crosstab(df['is_rush_hour'], df['severity'], normalize='index')
            rush_sev.index = ['Non-Rush','Rush Hour']
            rush_sev.columns = ['Fatal','Serious','Slight']
            rush_sev.plot(kind='bar', stacked=True, ax=ax, color=PAL3, edgecolor='#1e2330', linewidth=0.5)
            ax.set_title('Rush Hour vs Severity', fontsize=10, pad=8)
            ax.set_xlabel(''); ax.set_ylabel('Proportion', fontsize=8)
            ax.tick_params(axis='x', rotation=0, labelsize=8)
            ax.legend(fontsize=7)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    with c3:
        if 'Urban_or_Rural_Area' in df.columns:
            fig, ax = plt.subplots(figsize=(4.5,3.2))
            ur = df.copy()
            ur['area'] = ur['Urban_or_Rural_Area'].map({1:'Urban',2:'Rural'}).fillna('Other')
            area_sev = pd.crosstab(ur['area'], ur['severity'], normalize='index')
            area_sev.columns = ['Fatal','Serious','Slight']
            area_sev.plot(kind='bar', stacked=True, ax=ax, color=PAL3, edgecolor='#1e2330', linewidth=0.5)
            ax.set_title('Urban vs Rural Severity', fontsize=10, pad=8)
            ax.set_xlabel(''); ax.set_ylabel('Proportion', fontsize=8)
            ax.tick_params(axis='x', rotation=0, labelsize=8)
            ax.legend(fontsize=7)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE 4: MODEL INTELLIGENCE ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif "Model" in page:
    st.markdown("""
    <div class="page-header">
        <h1>MODEL INTELLIGENCE</h1>
        <p>Deep dive into model performance and internals</p>
    </div>
    """, unsafe_allow_html=True)

    # Top metrics
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Overall Accuracy</div>
            <div class="metric-value">69.9%</div>
            <div class="metric-sub">Random Forest</div>
        </div>
        <div class="metric-card blue">
            <div class="metric-label">Estimators</div>
            <div class="metric-value">200</div>
            <div class="metric-sub">Decision trees</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">Max Depth</div>
            <div class="metric-value">15</div>
            <div class="metric-sub">Tree depth</div>
        </div>
        <div class="metric-card red">
            <div class="metric-label">Balancing</div>
            <div class="metric-value">SMOTE</div>
            <div class="metric-sub">3× oversampled</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
        importances = model.feature_importances_
        indices     = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(indices)))
        bars = ax.barh(range(len(indices)),
                       importances[indices],
                       color=colors_fi, edgecolor='#1e2330', linewidth=0.5)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=8)
        ax.set_xlabel('Importance Score', fontsize=8)
        ax.set_title('Feature Importance — Random Forest', fontsize=10, pad=8)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, importances[indices]):
            ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.markdown('<div class="section-label">Class-Level Metrics</div>', unsafe_allow_html=True)
        metrics = {
            'Class':     ['Fatal (1)',   'Serious (2)', 'Slight (3)'],
            'Precision': [0.04,          0.18,          0.89],
            'Recall':    [0.18,          0.31,          0.76],
            'F1-Score':  [0.06,          0.23,          0.82],
            'Support':   [129,           1476,          10395],
        }
        metrics_df = pd.DataFrame(metrics)
        fig, axes = plt.subplots(1, 3, figsize=(5.5,3.5))
        for idx, metric in enumerate(['Precision','Recall','F1-Score']):
            vals   = metrics_df[metric].values
            colors = [PAL3[i] for i in range(3)]
            axes[idx].bar(['F','Se','Sl'], vals, color=colors, edgecolor='#1e2330', linewidth=0.5)
            axes[idx].set_title(metric, fontsize=9)
            axes[idx].set_ylim(0,1)
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].spines['top'].set_visible(False)
            axes[idx].spines['right'].set_visible(False)
            for j, v in enumerate(vals):
                axes[idx].text(j, v+0.02, f'{v:.2f}', ha='center', fontsize=8)
        fig.suptitle('Class-Level Performance', fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Confusion matrices
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Confusion Matrices</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # Computed from known results
    cm_data = np.array([[23,  56,  50],
                         [245, 457, 774],
                         [620,1872,7903]])

    with c1:
        fig, ax = plt.subplots(figsize=(4.5,3.5))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Fatal','Serious','Slight'],
                    yticklabels=['Fatal','Serious','Slight'],
                    linewidths=0.5, linecolor='#1e2330', annot_kws={'size':9})
        ax.set_title('Confusion Matrix (Counts)', fontsize=10, pad=8)
        ax.set_xlabel('Predicted', fontsize=8); ax.set_ylabel('Actual', fontsize=8)
        ax.tick_params(labelsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(4.5,3.5))
        cm_norm = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                    xticklabels=['Fatal','Serious','Slight'],
                    yticklabels=['Fatal','Serious','Slight'],
                    linewidths=0.5, linecolor='#1e2330', annot_kws={'size':9})
        ax.set_title('Confusion Matrix (Normalised)', fontsize=10, pad=8)
        ax.set_xlabel('Predicted', fontsize=8); ax.set_ylabel('Actual', fontsize=8)
        ax.tick_params(labelsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # SMOTE analysis
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">SMOTE Class Balancing</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(4.5,3.2))
        before = [517, 5901, 41580]
        ax.bar(['Fatal','Serious','Slight'], before, color=PAL3,
               edgecolor='#1e2330', linewidth=0.5, width=0.5, alpha=0.7)
        ax.set_title('Before SMOTE (Training Set)', fontsize=10, pad=8)
        ax.set_ylabel('Count', fontsize=8)
        for i, v in enumerate(before):
            ax.text(i, v+300, f'{v:,}', ha='center', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(4.5,3.2))
        after = [41580, 41580, 41580]
        ax.bar(['Fatal','Serious','Slight'], after, color=PAL3,
               edgecolor='#1e2330', linewidth=0.5, width=0.5)
        ax.set_title('After SMOTE (Balanced)', fontsize=10, pad=8)
        ax.set_ylabel('Count', fontsize=8)
        for i, v in enumerate(after):
            ax.text(i, v+300, f'{v:,}', ha='center', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Model pipeline summary
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Pipeline Summary</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Data Loading",       "3 CSV files merged → 59,998 records × 35 columns"),
        ("02", "Preprocessing",      "Null handling · Hour extraction · Type casting"),
        ("03", "Feature Engineering","Weather/Road/Light mapping · is_night · is_rush_hour · casualty_per_vehicle"),
        ("04", "Encoding",           "LabelEncoder for 4 categorical features"),
        ("05", "Scaling",            "StandardScaler on 6 numerical features"),
        ("06", "SMOTE",              "Oversampled Fatal 80× · Serious 7× → 124,740 training samples"),
        ("07", "Model Training",     "RandomForestClassifier · 200 trees · max_depth=15 · n_jobs=-1"),
        ("08", "Evaluation",         "69.9% accuracy · Strong Slight recall (0.76) · Weak Fatal recall (0.18)"),
    ]
    for num, title, detail in steps:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:1rem;margin-bottom:0.7rem;
                    background:#111318;border:1px solid #252a35;border-radius:8px;padding:0.8rem 1rem;">
            <div style="font-family:'JetBrains Mono',monospace;color:#f97316;font-size:0.9rem;
                        font-weight:600;min-width:28px;">{num}</div>
            <div>
                <div style="font-weight:600;font-size:0.85rem;margin-bottom:0.2rem;">{title}</div>
                <div style="font-size:0.78rem;color:#6b7280;">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
