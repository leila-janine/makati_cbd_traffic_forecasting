import streamlit as st
import pandas as pd
import numpy as np
import json
import torch
import os
import requests
import pydeck as pdk
import time  
import torch.nn as nn
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
from torch_geometric_temporal.nn.recurrent import A3TGCN

# ==============================================================================
# CONFIGURATION & PAGE SETUP
# ==============================================================================
st.set_page_config(page_title="Makati CBD Traffic Forecast", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

# --- UP NOAH STYLING (Dark Theme UI) ---
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #4A90E2;}
    .stAlert {background-color: #1e2530; border: 1px solid #4A90E2;}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🔑 INSERT LIVE API KEYS HERE FOR THE DEFENSE
# ==============================================================================
with st.sidebar:
    st.title("System Controls")
    st.markdown("API Integrations")
    TOMTOM_API_KEY = st.text_input("TomTom API Key", value="s6lUv0XMtFGh1cDXSeJJKrE0oLWb95nK", type="password")
    OWM_API_KEY = st.text_input("OpenWeatherMap API Key", value="c0d396ba6de2119f4b396f73e5d64273", type="password")
    st.markdown("---")
    
    # Auto-Refresh Toggle for Defense
    auto_refresh = st.checkbox("Enable 1-Minute Auto-Refresh", value=False)
    
    st.markdown("---")
    st.markdown("**Powered by A3T-GCN**\n\nAnalyzes Spatial (Graph) and Temporal (Time-Series) dependencies simultaneously.")

MAKATI_LAT, MAKATI_LON = 14.5570, 121.0204

# Hardcoded hyperparameters matching training script
SEQ_LEN = 12
HIDDEN_DIM = 64
NUM_FEATURES = 6

# Smart Pathing: Uses CSV locally, but uses the compressed Parquet online
CSV_PATH = "makati_aug2024_hybrid_master.csv"
PARQUET_PATH = "makati_aug2024_hybrid_master.parquet"
JSON_PATH = "Makati_CBD_Traffic_Data.json"

MODEL_WEIGHTS = {
    15: "checkpoints/a3tgcn_h1.pt",
    30: "checkpoints/a3tgcn_h2.pt",
    60: "checkpoints/a3tgcn_h4.pt"
}

# ==============================================================================
# 1. MODEL ARCHITECTURE DEFINITION
# ==============================================================================
class A3TGCN_Makati(torch.nn.Module):
    def __init__(self, node_features, hidden_dim):
        super(A3TGCN_Makati, self).__init__()
        self.a3tgcn = A3TGCN(in_channels=node_features, out_channels=hidden_dim, periods=SEQ_LEN)
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        B, S, N, F_ = x.shape
        x_batched = x.permute(0, 2, 3, 1).reshape(B * N, F_, S) 
        offset = (torch.arange(B, device=x.device) * N).view(B, 1, 1)
        edge_index_batched = (edge_index.unsqueeze(0) + offset).transpose(0, 1).reshape(2, -1)
        h = self.a3tgcn(x_batched, edge_index_batched).reshape(B, N, -1)
        return self.linear(h).squeeze(-1)

# ==============================================================================
# 2. CACHED DATA, GRAPH LOADERS & LIVE API FETCHERS
# ==============================================================================
@st.cache_resource
def load_spatial_graph():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    segments = data['network']['segmentResults']
    seg_ids = sorted([seg['segmentId'] for seg in segments])
    seg_map = {sid: idx for idx, sid in enumerate(seg_ids)}
    idx_to_seg = {idx: sid for sid, idx in seg_map.items()}
    n_nodes = len(seg_ids)

    centroids = np.zeros((n_nodes, 2))
    idx_to_name = {}
    for seg in segments:
        idx = seg_map[seg['segmentId']]
        coords = seg['shape']
        lat = np.mean([c['latitude'] for c in coords])
        lon = np.mean([c['longitude'] for c in coords])
        centroids[idx] = [lat, lon]
        
        street_name = seg.get('streetName', seg.get('roadName', ''))
        if not street_name:
            street_name = f"Unnamed Segment (Lat: {lat:.4f}, Lon: {lon:.4f})"
        idx_to_name[idx] = street_name
        
    dist_matrix = cdist(centroids, centroids) * 111000 
    edge_list = [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j and dist_matrix[i, j] < 50]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return edge_index, n_nodes, idx_to_seg, idx_to_name, centroids

def fetch_live_apis():
    """Pings TomTom and OpenWeatherMap for the absolute latest real-time data."""
    current_time = datetime.utcnow() + timedelta(hours=8)
    
    try:
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={MAKATI_LAT}&lon={MAKATI_LON}&appid={OWM_API_KEY}&units=metric"
        res = requests.get(weather_url, timeout=5).json()
        live_rain = res.get('rain', {}).get('1h', 0.0)
    except:
        live_rain = 0.0 
        
    try:
        traffic_url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={MAKATI_LAT}%2C{MAKATI_LON}&key={TOMTOM_API_KEY}"
        res = requests.get(traffic_url, timeout=5).json()
        live_speed = res['flowSegmentData']['currentSpeed']
    except:
        live_speed = 20.0 

    hour_sin = np.sin(2 * np.pi * current_time.hour / 23.0)
    hour_cos = np.cos(2 * np.pi * current_time.hour / 23.0)
    day_sin = np.sin(2 * np.pi * current_time.weekday() / 6.0)
    day_cos = np.cos(2 * np.pi * current_time.weekday() / 6.0)
    
    return live_speed, live_rain, hour_sin, hour_cos, day_sin, day_cos, current_time

@st.cache_data(ttl=60)
def construct_live_tensor(n_nodes, seg_ids):
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
    else:
        df = pd.read_csv(CSV_PATH)
        
    min_speed = df['current_speed'].min()
    max_speed = df['current_speed'].max()
    
    history_df = df.tail(n_nodes * 15).copy()
    history_df['hour_sin'] = np.sin(2 * np.pi * pd.to_datetime(history_df['timestamp']).dt.hour / 23.0)
    history_df['hour_cos'] = np.cos(2 * np.pi * pd.to_datetime(history_df['timestamp']).dt.hour / 23.0)
    history_df['day_sin'] = np.sin(2 * np.pi * pd.to_datetime(history_df['timestamp']).dt.dayofweek / 6.0)
    history_df['day_cos'] = np.cos(2 * np.pi * pd.to_datetime(history_df['timestamp']).dt.dayofweek / 6.0)
    history_df['scaled_speed'] = (history_df['current_speed'] - min_speed) / (max_speed - min_speed)
    
    features = ["scaled_speed", "rainfall_mm", "hour_sin", "hour_cos", "day_sin", "day_cos"]
    pivot_list = []
    
    for feat in features:
        p = history_df.pivot(index='timestamp', columns='segment_id', values=feat)
        p = p.reindex(columns=seg_ids)
        p = p.ffill().bfill()
        p = p.ffill(axis=1).bfill(axis=1).fillna(0)
        p = p.tail(11)
        pivot_list.append(p.values)
        
    history_matrix = np.stack(pivot_list, axis=-1)
    
    if history_matrix.shape[0] < 11:
        pad_len = 11 - history_matrix.shape[0]
        padding = np.zeros((pad_len, n_nodes, NUM_FEATURES))
        history_matrix = np.concatenate([padding, history_matrix], axis=0)
    
    live_speed, live_rain, h_sin, h_cos, d_sin, d_cos, current_time = fetch_live_apis()
    
    live_slice = np.zeros((1, n_nodes, NUM_FEATURES))
    live_slice[0, :, 0] = history_matrix[-1, :, 0] 
    live_slice[0, :, 1] = live_rain
    live_slice[0, :, 2] = h_sin
    live_slice[0, :, 3] = h_cos
    live_slice[0, :, 4] = d_sin
    live_slice[0, :, 5] = d_cos
    
    live_X = np.concatenate([history_matrix, live_slice], axis=0)
    
    base_avg_kmh = np.mean(live_X[-1, :, 0]) * (max_speed - min_speed) + min_speed
    shift_kmh = live_speed - base_avg_kmh
    
    for i in range(12):
        seq_time = current_time - timedelta(minutes=15 * (11 - i))
        live_X[i, :, 2] = np.sin(2 * np.pi * seq_time.hour / 23.0)
        live_X[i, :, 3] = np.cos(2 * np.pi * seq_time.hour / 23.0)
        live_X[i, :, 4] = np.sin(2 * np.pi * seq_time.weekday() / 6.0)
        live_X[i, :, 5] = np.cos(2 * np.pi * seq_time.weekday() / 6.0)
        
        step_speeds_kmh = live_X[i, :, 0] * (max_speed - min_speed) + min_speed
        noise = np.random.normal(0, 1.0, n_nodes)
        shifted_kmh = np.clip(step_speeds_kmh + shift_kmh + noise, 0.0, max_speed)
        
        live_X[i, :, 0] = (shifted_kmh - min_speed) / (max_speed - min_speed)
        live_X[i, :, 1] = live_rain

    final_live_speeds = live_X[-1, :, 0] * (max_speed - min_speed) + min_speed
    
    return live_X, current_time, min_speed, max_speed, final_live_speeds

@st.cache_resource
def load_all_models():
    device = torch.device('cpu') 
    models = {}
    for mins, path in MODEL_WEIGHTS.items():
        actual_path = path
        if not os.path.exists(actual_path) and os.path.exists(os.path.basename(path)):
            actual_path = os.path.basename(path)
            
        try:
            model = A3TGCN_Makati(node_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
            model.load_state_dict(torch.load(actual_path, map_location=device))
            model.eval()
            models[mins] = model
        except Exception as e:
            st.warning(f"Could not load model for {mins}m horizon. Ensure '{actual_path}' exists. Error: {e}")
    return models, device

def get_congestion_color(speed):
    if speed < 10: 
        return [139, 0, 0, 220]      # Dark Red 
    elif speed < 20: 
        return [255, 0, 0, 220]      # Red 
    elif speed < 35: 
        return [255, 165, 0, 220]    # Orange 
    else: 
        return [0, 255, 0, 220]      # Green 

# ==============================================================================
# 3. RUN MULTI-HORIZON INFERENCE
# ==============================================================================
st.title("Makati CBD Traffic Forecast")
st.markdown("Live spatial-temporal forecasting of the Makati Central Business District.")

with st.spinner("Establishing Live Feed & Generating Spatial Tensors..."):
    try:
        edge_index, n_nodes, idx_to_seg, idx_to_name, centroids = load_spatial_graph()
        seg_ids_list = [idx_to_seg[i] for i in range(n_nodes)]
        live_X, latest_timestamp, min_speed, max_speed, diverse_live_speeds = construct_live_tensor(n_nodes, seg_ids_list)
        models, device = load_all_models()
        
        live_X_tensor = torch.tensor(live_X.tolist(), dtype=torch.float32).unsqueeze(0).to(device)
        edge_index = edge_index.to(device)
        predictions_kmh = {0: diverse_live_speeds} 
        
        with torch.no_grad():
            for mins, model in models.items():
                prediction_scaled = model(live_X_tensor, edge_index)
                pred_list = prediction_scaled.detach().cpu().flatten().tolist()
                pred_np = np.clip(np.array(pred_list), 0.0, 1.0)
                predictions_kmh[mins] = pred_np * (max_speed - min_speed) + min_speed
        
        system_ready = True
    except Exception as e:
        st.error(f"System Error: {e}")
        system_ready = False

# ==============================================================================
# 4. STREAMLIT UI DASHBOARD (MAP CENTRIC)
# ==============================================================================
if system_ready:
    st.markdown(f"**Latest data timestamp:** {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')} (Philippine Standard Time)")
    
    st.subheader("Live Congestion Map")
    
    map_horizon = st.select_slider(
        "Select Time Horizon to Visualize on Map:",
        options=[0, 15, 30, 60],
        format_func=lambda x: "Right Now (Live)" if x == 0 else f"+{x} Minutes Forecast"
    )
    
    map_df = pd.DataFrame(centroids, columns=['lat', 'lon'])
    map_df['Segment ID'] = [idx_to_seg[i] for i in range(n_nodes)]
    map_df['Location'] = [idx_to_name[i] for i in range(n_nodes)]
    map_df['Speed (km/h)'] = np.round(predictions_kmh.get(map_horizon, np.zeros(n_nodes)), 1)
    map_df['Color'] = map_df['Speed (km/h)'].apply(get_congestion_color)
    
    view_state = pdk.ViewState(latitude=MAKATI_LAT, longitude=MAKATI_LON, zoom=14.5, pitch=45)
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_color='Color',
        get_radius=25, 
        pickable=True,
        opacity=0.8,
        filled=True,
    )
    
    # NEW FIX: Custom HTML Legend placed right above the map (Clean UI)
    st.markdown("""
        <div style="display: flex; justify-content: center; gap: 25px; padding: 5px 0; margin-bottom: 15px;">
            <div style="display: flex; align-items: center;"><div style="width: 14px; height: 14px; background-color: rgb(139, 0, 0); border-radius: 50%; margin-right: 8px;"></div><span style="font-size: 14px; color: white;"><b>Heavy Traffic</b> (< 10 km/h)</span></div>
            <div style="display: flex; align-items: center;"><div style="width: 14px; height: 14px; background-color: rgb(255, 0, 0); border-radius: 50%; margin-right: 8px;"></div><span style="font-size: 14px; color: white;"><b>Slow</b> (10-20 km/h)</span></div>
            <div style="display: flex; align-items: center;"><div style="width: 14px; height: 14px; background-color: rgb(255, 165, 0); border-radius: 50%; margin-right: 8px;"></div><span style="font-size: 14px; color: white;"><b>Moderate</b> (20-35 km/h)</span></div>
            <div style="display: flex; align-items: center;"><div style="width: 14px; height: 14px; background-color: rgb(0, 255, 0); border-radius: 50%; margin-right: 8px;"></div><span style="font-size: 14px; color: white;"><b>FFast</b> (35+ km/h)</span></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.pydeck_chart(pdk.Deck(
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_DARK,
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"html": "<b>Location:</b> {Location} <br/> <b>Segment ID:</b> {Segment ID} <br/> <b>Est. Speed:</b> {Speed (km/h)} km/h"}
    ))
    
    st.divider()

    st.subheader("Speed Trajectory")
    
    avg_speeds = {mins: np.mean(preds) for mins, preds in predictions_kmh.items()}
    
    col0, col15, col30, col60 = st.columns(4)
    col0.metric("NOW (Live API)", f"{avg_speeds.get(0, 0):.1f} km/h")
    
    if 15 in avg_speeds:
        delta_15 = avg_speeds[15] - avg_speeds[0]
        col15.metric("+15 Mins Forecast", f"{avg_speeds[15]:.1f} km/h", f"{delta_15:.1f} km/h")
        
    if 30 in avg_speeds:
        delta_30 = avg_speeds[30] - avg_speeds[0]
        col30.metric("+30 Mins Forecast", f"{avg_speeds[30]:.1f} km/h", f"{delta_30:.1f} km/h")
        
    if 60 in avg_speeds:
        delta_60 = avg_speeds[60] - avg_speeds[0]
        col60.metric("+60 Mins Forecast", f"{avg_speeds[60]:.1f} km/h", f"{delta_60:.1f} km/h")

    st.divider()
    
    st.subheader("Most Congested Spots List")
    st.markdown("Segments requiring immediate attention based on future congestion drops.")
    
    results_data = {
        'Location': [idx_to_name[i] for i in range(n_nodes)],
        'Segment_ID': [idx_to_seg[i] for i in range(n_nodes)]
    }
    
    results_data['00 Min (Live)'] = predictions_kmh.get(0, np.zeros(n_nodes))
    if 15 in predictions_kmh: results_data['15 Min Forecast'] = predictions_kmh[15]
    if 30 in predictions_kmh: results_data['30 Min Forecast'] = predictions_kmh[30]
    if 60 in predictions_kmh: results_data['60 Min Forecast'] = predictions_kmh[60]
        
    results_df = pd.DataFrame(results_data)
    sort_target = '15 Min Forecast' if '15 Min Forecast' in results_df.columns else '00 Min (Live)'
    worst_segments = results_df.sort_values(by=sort_target, ascending=True).head(10).reset_index(drop=True)
    
    # FIX 1: Display the table using the full width of the screen
    st.dataframe(
        worst_segments.style.format(precision=1),
        use_container_width=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📉 1-Hour Plunge Trajectory (Top 5 Worst Segments)**")
    
    chart_df = worst_segments.head(5).copy()
    
    # FIX 2: Abbreviate long words to make the legend look incredibly clean!
    chart_df['Short_Loc'] = chart_df['Location'].apply(
        lambda x: x.replace(' Street', ' St.').replace(' Avenue', ' Ave.').replace(' Road', ' Rd.')
    )
    chart_df['Unique Location'] = chart_df['Short_Loc'] + " (" + chart_df['Segment_ID'].astype(str).str[-4:] + ")"
    
    # FIX 3: Draw the chart using the full width of the screen below the table
    top_5 = chart_df.drop(columns=['Segment_ID', 'Location', 'Short_Loc']).set_index('Unique Location').T
    st.line_chart(top_5, height=350)
        
    # ==============================================================================
    # 5. AUTO-REFRESH LOGIC
    # ==============================================================================
    if auto_refresh:
        time.sleep(60)
        st.rerun()
