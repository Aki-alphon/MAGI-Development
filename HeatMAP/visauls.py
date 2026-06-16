import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. Page Configuration & Custom Dark UI ---
st.set_page_config(page_title="MAYA Project Dashboard", layout="wide")

# Injecting Custom CSS to match the exact dark theme from your screenshots
st.markdown("""
    <style>
    .stApp {
        background-color: #18181c;
        color: #e0e0e0;
    }
    div[data-baseweb="tab-list"] {
        background-color: #25252d;
        border-radius: 8px;
        padding: 10px;
    }
    .metric-box {
        background-color: #25252d;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #a8b8d0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("MAYA Project: Edge AI & Spectral Optimization")
st.write("Interactive telemetry and algorithm debugging environment.")

# Create two tabs for the two different visualizations
tab1, tab2 = st.tabs(["Phase 1: Preprocessing Pipeline", "Phase 2: ML Model Evaluation"])

# ==========================================
# TAB 1: PREPROCESSING SIMULATOR
# ==========================================
with tab1:
    st.header("Image Preprocessing Simulator")
    st.write("Simulating the transformation of raw sensor data before it enters the MobileNetV2 architecture.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Pipeline Toggles")
        downsample = st.toggle("1. Downsample (224px)", value=False)
        normalize = st.toggle("2. Normalize Lighting (CLAHE)", value=False)
        tensor_scale = st.toggle("3. Tensor Scaling (0-1)", value=False)
        augment = st.toggle("4. Data Augmentation (Tilt)", value=False)
        
        if st.button("Reset Pipeline"):
            # Streamlit rerun resets toggles if we manage state, but we'll keep it simple
            st.rerun()

    with col2:
        st.subheader("Live Algorithm Status")
        
        # Dynamic UI updates based on toggles
        if not downsample:
            st.error("⚠️ Warning: Raw data (12MP) too heavy for edge inference. Raspberry Pi RAM will overflow.")
            shape = "(3000, 4000, 3)"
        else:
            st.success("✅ Resolution optimized for MobileNetV2.")
            shape = "(224, 224, 3)"

        if not tensor_scale:
            dtype = "uint8 (0-255)"
            st.warning("⚠️ Integers detected. Gradient descent may fail to converge.")
        else:
            dtype = "float32 (0.0-1.0)"
            
        st.markdown(f"""
        <div class="metric-box">
            <p><strong>Current Tensor Shape:</strong> <span style="color:#2ecc71;">{shape}</span></p>
            <p><strong>Current Data Type:</strong> <span style="color:#2ecc71;">{dtype}</span></p>
            <p><strong>Lighting Normalized:</strong> {'Yes (Shadows Removed)' if normalize else 'No (High Environmental Noise)'}</p>
            <p><strong>Terrain Robustness:</strong> {'High (Angles Simulated)' if augment else 'Low (Requires flat terrain)'}</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: ML EVALUATION DASHBOARD (SCATTER MATRIX)
# ==========================================
with tab2:
    st.header("ML Model Evaluation Dashboard")
    
    # 1. Generate Dummy Data (simulating your 100 validation samples)
    if 'eval_data' not in st.session_state:
        np.random.seed(42)
        # 50 Actually Diseased (higher average probability)
        diseased_probs = np.random.normal(loc=0.75, scale=0.15, size=50)
        # 50 Actually Healthy (lower average probability)
        healthy_probs = np.random.normal(loc=0.25, scale=0.15, size=50)
        
        probs = np.clip(np.concatenate([diseased_probs, healthy_probs]), 0.05, 0.95)
        actuals = np.concatenate([np.ones(50), np.zeros(50)])
        
        st.session_state.eval_data = pd.DataFrame({'Actual': actuals, 'Probability': probs})

    df = st.session_state.eval_data.copy()

    # 2. Controls
    threshold = st.slider("Classification Threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.01)
    
    # 3. Calculate Predictions based on threshold
    df['Predicted'] = (df['Probability'] >= threshold).astype(int)
    
    # Calculate Metrics
    tp = len(df[(df['Actual'] == 1) & (df['Predicted'] == 1)])
    tn = len(df[(df['Actual'] == 0) & (df['Predicted'] == 0)])
    fp = len(df[(df['Actual'] == 0) & (df['Predicted'] == 1)])
    fn = len(df[(df['Actual'] == 1) & (df['Predicted'] == 0)])
    
    accuracy = (tp + tn) / len(df)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 4. Display Top Metrics
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"<div class='metric-box'>ACCURACY<br><span class='metric-value'>{accuracy*100:.1f}%</span></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-box'>PRECISION<br><span class='metric-value'>{precision*100:.1f}%</span></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-box'>RECALL<br><span class='metric-value'>{recall*100:.1f}%</span></div>", unsafe_allow_html=True)

    st.write("---")
    
    # 5. Build the Plotly Scatter Plot (The 2x2 Matrix)
    # We add random jitter to x and y so points don't stack on top of each other
    df['jitter_y'] = df['Actual'] + np.random.uniform(-0.3, 0.3, size=len(df))
    df['jitter_x'] = df['Predicted'] + np.random.uniform(-0.3, 0.3, size=len(df))
    
    # Assign colors based on correctness
    conditions = [
        (df['Actual'] == 1) & (df['Predicted'] == 1), # TP (Green)
        (df['Actual'] == 0) & (df['Predicted'] == 0), # TN (Green)
        (df['Actual'] == 0) & (df['Predicted'] == 1), # FP (Red)
        (df['Actual'] == 1) & (df['Predicted'] == 0)  # FN (Orange)
    ]
    colors = ['#2ecc71', '#2ecc71', '#e74c3c', '#e67e22']
    df['Color'] = np.select(conditions, colors)

    fig = go.Figure()

    # Add the scatter points
    fig.add_trace(go.Scatter(
        x=df['jitter_x'],
        y=df['jitter_y'],
        mode='markers',
        marker=dict(size=12, color=df['Color'], opacity=0.8, line=dict(width=1, color='white')),
        hoverinfo='text',
        text=[f"Prob: {p:.2f}" for p in df['Probability']]
    ))

    # Formatting the Matrix Layout
    fig.update_layout(
        plot_bgcolor='#1e1e24',
        paper_bgcolor='#18181c',
        height=500,
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['Predicted Healthy (0)', 'Predicted Diseased (1)'],
            title="PREDICTED CLASS",
            range=[-0.5, 1.5],
            gridcolor='#333'
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['Actually Healthy (0)', 'Actually Diseased (1)'],
            title="ACTUAL CLASS",
            range=[-0.5, 1.5],
            gridcolor='#333'
        ),
        showlegend=False,
        shapes=[
            # Crosshairs to divide the quadrants
            dict(type="line", x0=0.5, x1=0.5, y0=-0.5, y1=1.5, line=dict(color="#555", width=2)),
            dict(type="line", x0=-0.5, x1=1.5, y0=0.5, y1=0.5, line=dict(color="#555", width=2))
        ]
    )
    
    # Add floating text labels for quadrants
    fig.add_annotation(x=1.3, y=1.4, text=f"True Positives: {tp}", showarrow=False, font=dict(color="#2ecc71", size=14))
    fig.add_annotation(x=-0.3, y=1.4, text=f"False Negatives: {fn}", showarrow=False, font=dict(color="#e67e22", size=14))
    fig.add_annotation(x=1.3, y=-0.4, text=f"False Positives: {fp}", showarrow=False, font=dict(color="#e74c3c", size=14))
    fig.add_annotation(x=-0.3, y=-0.4, text=f"True Negatives: {tn}", showarrow=False, font=dict(color="#2ecc71", size=14))

    st.plotly_chart(fig, use_container_width=True)

    if st.button("Regenerate Field Data"):
        del st.session_state.eval_data
        st.rerun()