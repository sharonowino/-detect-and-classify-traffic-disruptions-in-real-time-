"""
Real-Time Early Warning Dashboard
==================================
A comprehensive Streamlit dashboard for GTFS Disruption Detection with real-time predictions.

This dashboard integrates:
- ML models from gtfs_disruption_detection pipeline
- Real-time predictions using trained models
- Geographic visualization of disruptions
- Alert management and tracking
- Performance metrics and lead-time analysis

Run with: streamlit run realtime_early_warning_dashboard.py
"""

import os
import sys
import json
import pickle
import numpy as np
import requests
import streamlit as st
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Try to import optional dependencies
try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model paths
MODEL_DIR = "my_api/models"
SCALER_PATH = "my_api/models/scaler_20260305_061409.pkl"
MODEL_PATH = "disruption_model.pkl"

# API endpoints (if using separate model server)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Severity mapping
SEVERITY_MAP = {
    0: {"label": "NORMAL", "color": "#2ed573", "icon": "✅"},
    1: {"label": "MINOR", "color": "#1e90ff", "icon": "⚠️"},
    2: {"label": "MODERATE", "color": "#ffa502", "icon": "🔶"},
    3: {"label": "SEVERE", "color": "#ff4757", "icon": "🚨"}
}

# Available models on the backend
BINARY_MODELS = ["XGBoost", "RandomForest", "NeuralNet", "BEST"]
MULTI_MODELS  = ["XGBoost", "RandomForest", "NeuralNet", "BEST"]


def check_api_health():
    """Check if FastAPI backend is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
    except:
        pass
    return False, {}

DEFAULT_FEATURE_NAMES = [
    "speed_mean", "speed_std", "delay_mean_5m", "delay_mean_15m", "delay_mean_30m",
    "bunching_index", "on_time_pct", "headway_variance", "alert_nlp_score",
    "alert_count", "fleet_utilization", "speed_drop_ratio"
]

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="🚨 Real-Time Early Warning Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0a0e1a; color: #e2e8f0; }

[data-testid="stSidebar"] {
    background-color: #0f1628;
    border-right: 1px solid #1e2d4a;
}

.metric-card {
    background: linear-gradient(135deg, #0f1628 0%, #162035 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.normal::before { background: #2ed573; }
.metric-card.minor::before { background: #1e90ff; }
.metric-card.moderate::before { background: #ffa502; }
.metric-card.severe::before { background: #ff4757; }
.metric-card.total::before { background: #1e90ff; }

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
}
.normal .metric-value { color: #2ed573; }
.minor .metric-value { color: #1e90ff; }
.moderate .metric-value { color: #ffa502; }
.severe .metric-value { color: #ff4757; }
.total .metric-value { color: #1e90ff; }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.badge-normal { background: rgba(46,213,115,0.15); color: #2ed573; border: 1px solid #2ed573; }
.badge-minor { background: rgba(30,144,255,0.15); color: #1e90ff; border: 1px solid #1e90ff; }
.badge-moderate { background: rgba(255,165,2,0.15); color: #ffa502; border: 1px solid #ffa502; }
.badge-severe { background: rgba(255,71,87,0.15); color: #ff4757; border: 1px solid #ff4757; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #1e90ff;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2d4a;
}

.api-status-ok  { color: #2ed573; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
.api-status-err { color: #ff4757; font-family: 'Space Mono', monospace; font-size: 0.75rem; }

[data-testid="stDataFrame"] { border: 1px solid #1e2d4a; border-radius: 8px; }

.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #1e90ff);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e90ff, #1e3a5f);
    transform: translateY(-1px);
}

.stTabs [data-baseweb="tab-list"] { background: #0f1628; border-bottom: 1px solid #1e2d4a; }
.stTabs [data-baseweb="tab"]      { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #64748b; }
.stTabs [aria-selected="true"]    { color: #1e90ff !important; }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

/* Widget styling */
.stRadio > label,
.stSelect > label,
.stSlider > label,
.stTextInput > label,
.stTextArea > label,
.stNumberInput > label,
.stTimeInput > label,
.stDateInput > label,
.stCheckbox > label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #94a3b8;
}

.stRadio [data-baseweb="radio"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
}

.stSelect [data-baseweb="select"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
}

.stSlider [data-baseweb="slider"] {
    font-family: 'DM Sans', sans-serif;
}

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
}

.stCheckbox [data-testid="stCheckbox"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
}

/* Sidebar widget container */
section[data-testid="stSidebar"] {
    font-family: 'DM Sans', sans-serif;
}

section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stSelect,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] .stCheckbox {
    font-family: 'DM Sans', sans-serif;
}

.severity-normal { border-left: 4px solid #2ed573; }
.severity-minor { border-left: 4px solid #1e90ff; }
.severity-moderate { border-left: 4px solid #ffa502; }
.severity-severe { border-left: 4px solid #ff4757; }

.alert-item {
    background: #162035;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    border-left: 3px solid;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_models():
    """Load trained models and scaler."""
    models = {}
    
    # Try to load from model directory
    model_files = list(Path(MODEL_DIR).glob("*.pkl")) if os.path.exists(MODEL_DIR) else []
    
    for mf in model_files:
        try:
            with open(mf, 'rb') as f:
                models[mf.stem] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load {mf.name}: {e}")
    
    # Try to load scaler
    scaler = None
    for scaler_path in [SCALER_PATH, os.path.join(MODEL_DIR, "scaler_latest.pkl")]:
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                break
            except:
                pass
    
    return models, scaler

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def make_prediction(features: List[float], model, scaler) -> Dict:
    """Make a single prediction using the model."""
    if model is None:
        # Return simulated prediction
        return _simulated_prediction(features)
    
    try:
        X = np.array(features).reshape(1, -1)
        
        if scaler is not None:
            X = scaler.transform(X)
        
        prediction = model.predict(X)[0]
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
        
        severity = SEVERITY_MAP.get(prediction, SEVERITY_MAP[0])
        
        return {
            "severity": severity["label"],
            "severity_class": int(prediction),
            "confidence": float(max(probabilities)) * 100 if probabilities is not None else 80.0,
            "probabilities": probabilities.tolist() if probabilities is not None else [0.25]*4,
            "status": "ok"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _simulated_prediction(features: List[float]) -> Dict:
    """Generate simulated prediction when model not available."""
    # Use features to determine severity
    bunching = features[5] if len(features) > 5 else 0.2
    delay_mean = features[3] if len(features) > 3 else 50.0
    
    if bunching > 0.7 or delay_mean > 200:
        severity_class = 3
    elif bunching > 0.4 or delay_mean > 100:
        severity_class = 2
    elif bunching > 0.2 or delay_mean > 30:
        severity_class = 1
    else:
        severity_class = 0
    
    severity = SEVERITY_MAP[severity_class]
    
    return {
        "severity": severity["label"],
        "severity_class": severity_class,
        "confidence": 85.0 - severity_class * 10,
        "probabilities": [0.76 - severity_class*0.2, 0.14, 0.07, 0.03],
        "status": "simulated"
    }

# =============================================================================
# DATA GENERATION (FOR DEMO)
# =============================================================================

@st.cache_data(ttl=900)
def generate_realtime_data(n_routes: int = 20) -> pd.DataFrame:
    """Generate simulated real-time transit data."""
    np.random.seed(42)
    
    routes = [f"Route_{i:03d}" for i in range(1, n_routes + 1)]
    
    data = {
        "route_id": routes,
        "timestamp": [datetime.now()] * n_routes,
        "speed_mean": np.random.uniform(20, 50, n_routes),
        "speed_std": np.random.uniform(2, 15, n_routes),
        "delay_mean_5m": np.random.uniform(0, 120, n_routes),
        "delay_mean_15m": np.random.uniform(10, 180, n_routes),
        "delay_mean_30m": np.random.uniform(20, 240, n_routes),
        "bunching_index": np.random.uniform(0, 1, n_routes),
        "on_time_pct": np.random.uniform(0.6, 1.0, n_routes),
        "headway_variance": np.random.uniform(5, 60, n_routes),
        "alert_nlp_score": np.random.uniform(0, 0.5, n_routes),
        "alert_count": np.random.poisson(1, n_routes),
        "fleet_utilization": np.random.uniform(0.7, 1.0, n_routes),
        "speed_drop_ratio": np.random.uniform(0, 0.3, n_routes),
    }
    
    df = pd.DataFrame(data)
    
    # Add location data
    df['lat'] = np.random.uniform(51.9, 52.1, n_routes)  # Netherlands area
    df['lon'] = np.random.uniform(4.3, 5.1, n_routes)
    
    return df


@st.cache_data(ttl=900)
def load_live_feed_data() -> pd.DataFrame:
    """
    Load real-time data from GTFS-RT feeds.
    
    Uses the GTFSDataCollector and GTFSFeatureEngineer to fetch
    all three feeds and transform to dashboard schema.
    """
    try:
        from utils.gtfs_collector import collect_and_process
        
        # Get feed URLs from secrets or use defaults
        try:
            from data_connectors import get_secrets
            secrets = get_secrets()
            api_keys = secrets.get('api_keys', {})
            vehicle_url = api_keys.get('GTFS_VEHICLE_URL', 'http://gtfs.ovapi.nl/nl/vehiclePositions.pb')
            trip_url = api_keys.get('GTFS_TRIP_UPDATE_URL', 'http://gtfs.ovapi.nl/nl/tripUpdates.pb')
            alerts_url = api_keys.get('GTFS_ALERTS_URL', 'http://gtfs.ovapi.nl/nl/alerts.pb')
        except:
            # Use default OVAPI URLs
            vehicle_url = 'http://gtfs.ovapi.nl/nl/vehiclePositions.pb'
            trip_url = 'http://gtfs.ovapi.nl/nl/tripUpdates.pb'
            alerts_url = 'http://gtfs.ovapi.nl/nl/alerts.pb'
        
        # Collect and process data
        df = collect_and_process(
            vehicle_url=vehicle_url,
            trip_url=trip_url,
            alerts_url=alerts_url
        )
        
        if df.empty:
            st.warning("No data in feed, using demo data")
            return generate_realtime_data(n_routes=20)
        
        return df
        
    except ImportError as e:
        st.warning(f"gtfs_collector module not available: {e}. Using demo data.")
        return generate_realtime_data(n_routes=20)
    except Exception as e:
        st.warning(f"Error loading live feed: {e}. Using demo data.")
        return generate_realtime_data(n_routes=20)

# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

def display_header():
    """Display dashboard header."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.image("https://img.icons8.com/fluency/96/bus.png", width=64)
    
    with col2:
        st.markdown("""
        # 🚨 Real-Time Early Warning Dashboard
        **GTFS Transit Disruption Detection System**
        
        Monitoring transit routes for potential disruptions
        """)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: right; color: #888;">
            Last Updated:<br>
            <strong style="color: #fff; font-family: 'Space Mono';">
                {datetime.now().strftime('%H:%M:%S')}
            </strong>
        </div>
        """, unsafe_allow_html=True)

def display_severity_summary(df: pd.DataFrame, predictions: List[Dict]):
    """Display severity summary metrics."""
    if not predictions:
        return
    
    # Count by severity
    severity_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for p in predictions:
        severity_counts[p['severity_class']] = severity_counts.get(p['severity_class'], 0) + 1
    
    total = len(predictions)
    
    # Display metrics
    cols = st.columns(5)
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card severity-total">
            <div style="color: #888; font-size: 14px;">TOTAL ROUTES</div>
            <div style="color: #fff; font-size: 32px; font-family: 'Space Mono'">{total}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card severity-normal">
            <div style="color: #2ed573; font-size: 14px;">NORMAL</div>
            <div style="color: #2ed573; font-size: 32px; font-family: 'Space Mono'">{severity_counts[0]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card severity-minor">
            <div style="color: #1e90ff; font-size: 14px;">MINOR</div>
            <div style="color: #1e90ff; font-size: 32px; font-family: 'Space Mono'">{severity_counts[1]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(f"""
        <div class="metric-card severity-moderate">
            <div style="color: #ffa502; font-size: 14px;">MODERATE</div>
            <div style="color: #ffa502; font-size: 32px; font-family: 'Space Mono'">{severity_counts[2]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[4]:
        st.markdown(f"""
        <div class="metric-card severity-severe">
            <div style="color: #ff4757; font-size: 14px;">SEVERE</div>
            <div style="color: #ff4757; font-size: 32px; font-family: 'Space Mono'">{severity_counts[3]}</div>
        </div>
        """, unsafe_allow_html=True)

def display_route_table(df: pd.DataFrame, predictions: List[Dict]):
    """Display detailed route predictions."""
    # Data already has predictions attached from main()
    result_df = df.copy()
    
    # Sort by severity (most severe first)
    result_df = result_df.sort_values('severity_class', ascending=False)
    
    # Display
    st.subheader("📊 Route Status")
    
    # Display table (data is already filtered from sidebar)
    display_cols = ['route_id', 'prediction', 'confidence', 'speed_mean', 'delay_mean_15m', 'bunching_index']
    
    st.dataframe(
        result_df[display_cols].style.background_gradient(
            subset=['confidence'],
            cmap='RdYlGn'
        ),
        use_container_width=True,
        height=400
    )

def display_alert_feed(predictions: List[Dict], df: pd.DataFrame):
    """Display real-time alert feed."""
    st.subheader("🚨 Active Alerts")
    
    # Filter to only severe/moderate
    critical_alerts = [
        (p, df.iloc[i]) 
        for i, p in enumerate(predictions) 
        if p['severity_class'] >= 2
    ]
    
    if not critical_alerts:
        st.info("No active alerts at this time.")
        return
    
    for prediction, route_data in critical_alerts:
        severity = SEVERITY_MAP[prediction['severity_class']]
        
        st.markdown(f"""
        <div class="alert-item" style="border-left-color: {severity['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: {severity['color']}; font-size: 18px;">
                        {severity['icon']} {route_data['route_id']}
                    </strong>
                    <div style="color: #888; margin-top: 4px;">
                        {severity['label']} Disruption • Confidence: {prediction['confidence']:.1f}%
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #fff;">Delay: {route_data['delay_mean_15m']:.0f}s</div>
                    <div style="color: #888;">Bunching: {route_data['bunching_index']:.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_metrics_charts(df: pd.DataFrame, predictions: List[Dict]):
    """Display metrics visualizations."""
    result_df = df.copy()
    result_df['severity_class'] = [p['severity_class'] for p in predictions]
    
    # Create tabs for different charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Delay Trends", "📊 Distribution", "🎯 Feature Analysis", "🚨 Alert Queue"])
    
    with tab1:
        # Line chart of delays over time (simulated)
        fig = px.line(
            result_df, 
            x='route_id', 
            y='delay_mean_15m',
            color='severity_class',
            title='15-Minute Delay by Route',
            color_discrete_map={k: v['color'] for k, v in SEVERITY_MAP.items()},
            template="plotly_dark"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font=dict(family="DM Sans"),
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Distribution charts
        st.markdown('<p class="section-title">📊 DISTRIBUTION & ANALYTICS</p>', unsafe_allow_html=True)
        
        analytics_days = st.slider("Historical Days", 7, 90, 30)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bunching Index Distribution
            fig = px.histogram(
                result_df, 
                x='bunching_index',
                color='severity_class',
                title='Bunching Index Distribution',
                color_discrete_map={k: v['color'] for k, v in SEVERITY_MAP.items()},
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,22,40,0.8)",
                font=dict(family="DM Sans"),
                xaxis=dict(gridcolor="#1e2d4a"),
                yaxis=dict(gridcolor="#1e2d4a"),
                margin=dict(l=0,r=0,t=10,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Severity distribution bar chart
            severity_counts = result_df['severity_class'].value_counts().reset_index()
            severity_counts.columns = ['Severity', 'Count']
            severity_counts['Severity'] = severity_counts['Severity'].map(
                {0: 'Normal', 1: 'Minor', 2: 'Moderate', 3: 'Severe'}
            )
            fig_bar = px.bar(severity_counts, x='Severity', y='Count',
                           title="Disruptions by Severity",
                           color='Severity',
                           color_discrete_map={'Normal': '#2ed573', 'Minor': '#1e90ff', 
                                             'Moderate': '#ffa502', 'Severe': '#ff4757'},
                           template="plotly_dark")
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,22,40,0.8)")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Routes by Severity pie
            fig = px.pie(
                result_df, 
                names='severity_class',
                title='Routes by Severity',
                color='severity_class',
                color_discrete_map={k: v['color'] for k, v in SEVERITY_MAP.items()},
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,22,40,0.8)",
                font=dict(family="DM Sans"),
                margin=dict(l=0,r=0,t=10,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Route severity counts
            route_counts = result_df.groupby(['route_id', 'severity_class']).size().reset_index(name='Count')
            route_counts['severity_class'] = route_counts['severity_class'].map(
                {0: 'Normal', 1: 'Minor', 2: 'Moderate', 3: 'Severe'}
            )
            fig_line = px.bar(route_counts, x='route_id', y='Count',
                            title="Routes by Alert Count",
                            color='severity_class',
                            template="plotly_dark")
            fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,22,40,0.8)")
            st.plotly_chart(fig_line, use_container_width=True)
    
    with tab3:
        # Feature correlation heatmap
        feature_cols = ['speed_mean', 'delay_mean_15m', 'bunching_index', 'on_time_pct', 'headway_variance']
        corr = result_df[feature_cols].corr()
        
        fig = px.imshow(
            corr,
            text_auto=True,
            title='Feature Correlations',
            color_continuous_scale='RdBu_r',
            template="plotly_dark"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font=dict(family="DM Sans"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Alert Queue from transit_sentinel
        st.markdown('<p class="section-title">ALERT PRIORITIZATION QUEUE</p>', unsafe_allow_html=True)
        
        # Filters for interactivity
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            selected_severity = st.multiselect(
                "Filter by Severity",
                options=["SEVERE", "MODERATE", "MINOR", "NORMAL"],
                default=["SEVERE", "MODERATE", "MINOR", "NORMAL"]
            )
        with col_filter2:
            sort_by = st.selectbox(
                "Sort by",
                options=["Severity", "Delay", "Bunching", "Confidence"]
            )
        
        # KPIs with vibrant colors
        c1, c2, c3, c4 = st.columns(4)
        active_alerts = [p for p in predictions if p['severity_class'] >= 2]
        critical_n = sum(1 for p in predictions if p['severity_class'] == 3)
        moderate_n = sum(1 for p in predictions if p['severity_class'] == 2)
        total_pax = len(active_alerts) * 150  # Simulated
        
        # Custom styled metrics
        c1.markdown(f'''
        <div class="metric-card">
            <div class="metric-label" style="color:#64748b;">ACTIVE ALERTS</div>
            <div class="metric-value" style="color:#1e90ff;">{len(active_alerts)}</div>
        </div>
        ''', unsafe_allow_html=True)
        c2.markdown(f'''
        <div class="metric-card">
            <div class="metric-label" style="color:#64748b;">CRITICAL</div>
            <div class="metric-value" style="color:#ff4757;">{critical_n}</div>
        </div>
        ''', unsafe_allow_html=True)
        c3.markdown(f'''
        <div class="metric-card">
            <div class="metric-label" style="color:#64748b;">MODERATE</div>
            <div class="metric-value" style="color:#ffa502;">{moderate_n}</div>
        </div>
        ''', unsafe_allow_html=True)
        c4.markdown(f'''
        <div class="metric-card">
            <div class="metric-label" style="color:#64748b;">AVG RESPONSE</div>
            <div class="metric-value" style="color:#2ed573;">15m</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Build alert data with filters
        severity_map_reverse = {"SEVERE": 3, "MODERATE": 2, "MINOR": 1, "NORMAL": 0}
        
        alert_data = []
        for i, p in enumerate(predictions):
            if p['severity'] in selected_severity:
                alert_data.append({
                    "Route": df.iloc[i]['route_id'] if i < len(df) else "N/A",
                    "Severity": p['severity'],
                    "severity_class": p['severity_class'],
                    "Delay (15m)": df.iloc[i]['delay_mean_15m'] if i < len(df) else 0,
                    "Bunching": df.iloc[i]['bunching_index'] if i < len(df) else 0,
                    "Confidence": p['confidence'],
                })
        
        # Sort data
        if sort_by == "Severity":
            alert_data.sort(key=lambda x: x['severity_class'], reverse=True)
        elif sort_by == "Delay":
            alert_data.sort(key=lambda x: x['Delay (15m)'], reverse=True)
        elif sort_by == "Bunching":
            alert_data.sort(key=lambda x: x['Bunching'], reverse=True)
        elif sort_by == "Confidence":
            alert_data.sort(key=lambda x: x['Confidence'], reverse=True)
        
        # Display alert table
        if alert_data:
            # Format for display
            display_data = []
            for a in alert_data:
                display_data.append({
                    "Route": a['Route'],
                    "Severity": a['Severity'],
                    "Delay (15m)": f"{a['Delay (15m)']:.0f}s",
                    "Bunching": f"{a['Bunching']:.2f}",
                    "Confidence": f"{a['Confidence']:.1f}%",
                })
            
            alert_df = pd.DataFrame(display_data)
            
            # Row coloring based on severity
            def _row_color(row):
                colors = {"SEVERE": "#FEE2E2", "MODERATE": "#FEF3C7", "MINOR": "#EFF6FF", "NORMAL": "#D1FAE5"}
                return [f"background-color:{colors.get(row['Severity'], 'white')}"] * len(row)
            
            st.dataframe(
                alert_df.style.apply(_row_color, axis=1),
                use_container_width=True, height=300,
                column_config={
                    "Route": st.column_config.TextColumn("Route", width="medium"),
                    "Severity": st.column_config.TextColumn("Severity", width="small"),
                    "Delay (15m)": st.column_config.TextColumn("Delay", width="small"),
                    "Bunching": st.column_config.TextColumn("Bunching", width="small"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                }
            )
            
            # Detail view with interactive selection
            st.markdown("### 📋 Alert Details")
            
            col_detail1, col_detail2 = st.columns([2, 1])
            with col_detail1:
                sel_route = st.selectbox("Select Route to View Details", 
                                        options=[a['Route'] for a in alert_data],
                                        key="alert_detail_select")
            
            sel_data = next((a for a in alert_data if a['Route'] == sel_route), None)
            
            if sel_data:
                # Detail view from transit_sentinel format
                st.markdown("### 📋 Selected Alert Details")
                d1, d2 = st.columns(2)
                with d1:
                    for k, v in [
                        ("Route", sel_data['Route']),
                        ("Severity", sel_data['Severity']),
                        ("Delay", f"{sel_data['Delay (15m)']:.0f}s"),
                        ("Bunching", f"{sel_data['Bunching']:.2f}")
                    ]:
                        st.markdown(f"**{k}:** {v}")
                with d2:
                    for k, v in [
                        ("Confidence", f"{sel_data['Confidence']:.1f}%"),
                        ("Impact Score", f"{sel_data['severity_class'] * 25}%"),
                        ("Status", "Active"),
                        ("Detected", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    ]:
                        st.markdown(f"**{k}:** {v}")
        else:
            st.info("No alerts match the selected filters.")

def display_map(df: pd.DataFrame, predictions: List[Dict]):
    """Display geographic map with routes."""
    if not FOLIUM_AVAILABLE:
        st.warning("Map visualization requires folium. Install with: pip install folium streamlit-folium")
        return
    
    # Add prediction data to dataframe
    map_df = df.copy()
    map_df['severity'] = [p['severity'] for p in predictions]
    map_df['severity_class'] = [p['severity_class'] for p in predictions]
    map_df['color'] = map_df['severity_class'].map(lambda x: SEVERITY_MAP[x]['color'])
    
    # Create map centered on Netherlands
    m = folium.Map(location=[52.0, 4.7], zoom_start=10, tiles='CartoDB dark_matter')
    
    # Add markers for each route
    for _, row in map_df.iterrows():
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,
                color=row['color'],
                fill=True,
                fillColor=row['color'],
                fillOpacity=0.7,
                popup=f"{row['route_id']}: {row['severity']}"
            ).add_to(m)
    
    st_folium(m, use_container_width=True, height=500)

def display_lead_time_analysis():
    """Display lead-time analysis for early warning."""
    st.subheader("⏱️ Early Warning Lead Time Analysis")
    
    # Simulated lead time data
    lead_times = np.random.exponential(15, 1000)  # minutes
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            x=lead_times,
            nbins=30,
            title='Distribution of Warning Lead Times',
            labels={'x': 'Lead Time (minutes)', 'y': 'Count'},
            template="plotly_dark",
            color_discrete_sequence=["#1e90ff"]
        )
        fig.add_vline(x=15, line_dash="dash", line_color="#ff4757", annotation_text="15 min threshold")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font=dict(family="DM Sans"),
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Lead Time Metrics
        
        | Metric | Value |
        |--------|-------|
        | Mean Lead Time | 15.2 min |
        | Median Lead Time | 12.8 min |
        | Std Deviation | 8.4 min |
        | Alerts with >15min lead | 42% |
        
        **Goal**: Achieve >80% of alerts with ≥15 minute lead time before disruption.
        """)

def display_model_performance():
    """Display model performance metrics."""
    st.subheader("📊 Model Performance")
    
    # Sample metrics (would come from actual model evaluation)
    metrics = {
        "Model": ["XGBoost", "Random Forest", "LightGBM", "Neural Net", "Ensemble"],
        "F1 Score": [0.85, 0.82, 0.87, 0.79, 0.89],
        "Precision": [0.88, 0.84, 0.89, 0.81, 0.91],
        "Recall": [0.82, 0.80, 0.85, 0.77, 0.87],
        "ROC-AUC": [0.92, 0.89, 0.94, 0.86, 0.95]
    }
    
    metrics_df = pd.DataFrame(metrics)
    
    # Display with styling
    st.dataframe(
        metrics_df.style.background_gradient(
            subset=['F1 Score', 'Precision', 'Recall', 'ROC-AUC'],
            cmap='Greens'
        ),
        use_container_width=True
    )

# =============================================================================
# SIDEBAR
# =============================================================================

def display_sidebar():
    """Display sidebar controls."""
    with st.sidebar:
        st.title("⚙️ Dashboard Controls")
        
        st.markdown('<p class="section-title">Data Source</p>', unsafe_allow_html=True)
        data_source = st.radio(
            "Select Data Source",
            ["Live Feed", "Demo Data", "Upload CSV"],
            index=1
        )
        
        st.markdown("---")
        
        # ── Global filters ───────────────────────
        st.markdown('<p class="section-title">Global Filters</p>', unsafe_allow_html=True)
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=["NORMAL", "MINOR", "MODERATE", "SEVERE"],
            default=["NORMAL", "MINOR", "MODERATE", "SEVERE"]
        )
        route_search = st.text_input("Search Route", "")
        
        st.markdown("---")
        
        # ── Model selection ───────────────────────
        st.markdown('<p class="section-title">Model Selection</p>', unsafe_allow_html=True)
        task = st.radio("Task type", ["binary", "multi"], horizontal=True)
        model_pool = BINARY_MODELS if task == "binary" else MULTI_MODELS
        selected_model = st.selectbox("Model", model_pool, index=model_pool.index("BEST") if "BEST" in model_pool else 0)
        
        st.markdown("---")
        
        # ── API status indicator ──────────────────
        st.markdown('<p class="section-title">FastAPI Backend</p>', unsafe_allow_html=True)
        api_ok, api_info = check_api_health()
        if api_ok:
            binary_avail = api_info.get("binary_models", BINARY_MODELS)
            multi_avail  = api_info.get("multi_models",  MULTI_MODELS)
            st.markdown(
                f'<p class="api-status-ok">● ONLINE — {len(binary_avail)+len(multi_avail)} models loaded</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="api-status-err">● OFFLINE — using demo predictions</p>',
                unsafe_allow_html=True,
            )
            st.caption("Start backend: `uvicorn main:app --reload`")
        
        st.markdown("---")
        
        st.markdown('<p class="section-title">Refresh Settings</p>', unsafe_allow_html=True)
        
        if AUTOREFRESH_AVAILABLE:
            refresh_interval = st.slider("Auto-refresh interval (seconds)", 5, 60, 30)
            st_autorefresh(refresh_interval * 1000, key="data_refresh")
        else:
            st.info("Install streamlit-autorefresh for auto-refresh")
            if st.button("🔄 Refresh Data"):
                st.rerun()
        
        st.markdown("---")
        
        st.markdown('<p class="section-title">Display Options</p>', unsafe_allow_html=True)
        
        show_map = st.checkbox("Show Geographic Map", value=True)
        show_alerts = st.checkbox("Show Alert Feed", value=True)
        show_charts = st.checkbox("Show Analysis Charts", value=True)
        
        confidence_threshold = st.slider(
            "Alert Confidence Threshold",
            min_value=0,
            max_value=100,
            value=70,
            step=5
        )
        
        st.markdown("---")
        st.markdown('<p class="section-title">About</p>', unsafe_allow_html=True)
        st.markdown("""
        **GTFS Real-Time Early Warning System**
        
        Uses machine learning to predict transit disruptions 15-30 minutes before they occur.
        
        Models: XGBoost, Random Forest, LightGBM, Neural Networks, GNN
        """)
        
        return {
            "data_source": data_source,
            "task": task,
            "selected_model": selected_model,
            "show_map": show_map,
            "show_alerts": show_alerts,
            "show_charts": show_charts,
            "confidence_threshold": confidence_threshold,
            "severity_filter": severity_filter,
            "route_search": route_search
        }

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Load models
    models, scaler = load_models()
    
    # Display header
    display_header()
    
    # Get sidebar settings
    settings = display_sidebar()
    
    st.markdown("---")
    
    # Generate or load data
    if settings["data_source"] == "Demo Data":
        df = generate_realtime_data(n_routes=20)
    elif settings["data_source"] == "Live Feed":
        with st.spinner("Connecting to GTFS-RT feed..."):
            df = load_live_feed_data()
    else:
        # Default to demo data
        df = generate_realtime_data(n_routes=20)
    
    # Make predictions
    features = df[DEFAULT_FEATURE_NAMES].values.tolist()
    predictions = []
    
    for feature_row in features:
        pred = make_prediction(feature_row, models.get('BEST_binary_model'), scaler)
        predictions.append(pred)
    
    # Combine data with predictions for filtering
    result_df = df.copy()
    result_df['prediction'] = [p['severity'] for p in predictions]
    result_df['confidence'] = [p['confidence'] for p in predictions]
    result_df['severity_class'] = [p['severity_class'] for p in predictions]
    
    # Apply global filters from sidebar
    filtered_df = result_df[result_df['prediction'].isin(settings["severity_filter"])]
    if settings["route_search"]:
        filtered_df = filtered_df[filtered_df['route_id'].str.contains(settings["route_search"], case=False)]
    
    # Extract filtered predictions (maintain alignment with filtered_df)
    filtered_predictions = []
    for idx in filtered_df.index:
        orig_idx = result_df.index.get_loc(idx)
        filtered_predictions.append(predictions[orig_idx])
    
    # Display severity summary (use filtered data)
    display_severity_summary(filtered_df, filtered_predictions)
    
    st.markdown("---")
    
    # Display main content (use filtered data)
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Route table (use filtered_df directly)
        display_route_table(filtered_df, filtered_predictions)
        
        # Charts (use filtered data)
        if settings["show_charts"]:
            display_metrics_charts(filtered_df, filtered_predictions)
    
    with col_right:
        # Alert feed (use filtered data)
        if settings["show_alerts"]:
            display_alert_feed(filtered_predictions, filtered_df)
        
        # Lead time analysis
        display_lead_time_analysis()
    
    # Map (full width) - use filtered data
    if settings["show_map"]:
        st.markdown('<p class="section-title">🗺️ Geographic View</p>', unsafe_allow_html=True)
        display_map(filtered_df, filtered_predictions)
    with st.expander("📊 Model Performance Details"):
        display_model_performance()

if __name__ == "__main__":
    main()

response = requests.get("http://localhost:8000/predictions")

data = pd.DataFrame(response.json())

st.dataframe(data)

