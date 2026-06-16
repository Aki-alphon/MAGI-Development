import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. Page Configuration & Custom Dark UI ---
st.set_page_config(page_title="ARACHNE Control Interface", layout="wide")

# Injecting Custom CSS for a dark robotics/engineering theme
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
    .highlight-green { color: #2ecc71; font-weight: bold; }
    .highlight-red { color: #e74c3c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🕷️ ARACHNE: Spider Robot Control Center")
st.write("Live gait stabilization and actuator telemetry diagnostics.")

# Create two tabs for control and monitoring
tab1, tab2 = st.tabs(["Phase 1: Kinematics & Gait Control", "Phase 2: Actuator Telemetry Dashboard"])

# ==========================================
# TAB 1: KINEMATICS & GAIT CONTROL
# ==========================================
with tab1:
    st.header("Gait Stabilization Subsystem")
    st.write("Configure the active kinematics engines before initiating movement commands.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Subsystem Toggles")
        ik_engine = st.toggle("1. Inverse Kinematics (IK) Engine", value=True)
        terrain_adapt = st.toggle("2. IMU Terrain Adaptation", value=False)
        power_save = st.toggle("3. Low-Torque Standby Mode", value=False)
        overtravel = st.toggle("4. Joint Overtravel Limits", value=True)
        
        if st.button("Reboot Kinematics Core"):
            st.rerun()

    with col2:
        st.subheader("Live System Status")
        
        # Dynamic UI updates based on toggles
        if not ik_engine:
            st.error("⚠️ Warning: IK Disabled. Legs must be controlled via manual joint angles. High risk of toppling.")
            gait_mode = "Manual / Raw Angle"
        else:
            st.success("✅ IK Engine Active. Cartesian target coordinates enabled.")
            gait_mode = "Dynamic Tripod / Wave"

        if power_save:
            power_draw = "1.2 Amps (Standby)"
            st.warning("⚠️ Low-Torque Mode active. Robot will not support heavy payloads.")
        else:
            power_draw = "4.5 Amps (Active)"
            
        st.markdown(f"""
        <div class="metric-box">
            <p><strong>Current Gait Mode:</strong> <span class="highlight-green">{gait_mode}</span></p>
            <p><strong>System Power Draw:</strong> <span class="highlight-green">{power_draw}</span></p>
            <p><strong>Auto-Leveling:</strong> {'Active (IMU Linked)' if terrain_adapt else 'Disabled (Flat Plane Assumed)'}</p>
            <p><strong>Hardware Safety:</strong> {'Enabled (Collisions Prevented)' if overtravel else '<span class="highlight-red">Disabled (Risk of Servo Burnout)</span>'}</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: ACTUATOR TELEMETRY DASHBOARD
# ==========================================
with tab2:
    st.header("Actuator Diagnostics (Load vs. Temperature)")
    st.write("Monitoring all 24 servos (8 legs × 3 joints: Coxa, Femur, Tibia).")
    
    # 1. Generate Dummy Telemetry Data (simulating 24 servos)
    if 'telemetry_data' not in st.session_state:
        np.random.seed()
        # Leg IDs and Joint types
        legs = [f"L{i}" for i in range(1, 5)] + [f"R{i}" for i in range(1, 5)]
        joints = ["Coxa", "Femur", "Tibia"]
        
        servo_ids = [f"{leg}-{joint}" for leg in legs for joint in joints]
        
        # Simulate Loads (0-100%) and Temps (30C - 90C)
        loads = np.random.normal(loc=40, scale=20, size=24)
        temps = np.random.normal(loc=45, scale=15, size=24)
        
        # Correlate temp and load slightly (higher load = higher temp)
        temps += (loads * 0.3)
        
        # Clip to realistic ranges
        loads = np.clip(loads, 5, 100)
        temps = np.clip(temps, 25, 95)
        
        st.session_state.telemetry_data = pd.DataFrame({
            'Servo_ID': servo_ids,
            'Load_Pct': loads,
            'Temp_C': temps
        })

    df = st.session_state.telemetry_data.copy()

    # 2. Controls for Safety Thresholds
    colA, colB = st.columns(2)
    with colA:
        load_threshold = st.slider("Max Safe Torque Load (%)", min_value=30, max_value=90, value=70, step=5)
    with colB:
        temp_threshold = st.slider("Max Safe Temperature (°C)", min_value=40, max_value=85, value=65, step=1)
    
    # Calculate Diagnostics Metrics
    avg_temp = df['Temp_C'].mean()
    peak_load = df['Load_Pct'].max()
    
    # Find critical servos (violating BOTH thresholds)
    critical_servos = len(df[(df['Load_Pct'] >= load_threshold) & (df['Temp_C'] >= temp_threshold)])

    # 4. Display Top Metrics
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"<div class='metric-box'>AVG SERVO TEMP<br><span class='metric-value'>{avg_temp:.1f} °C</span></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-box'>PEAK SYSTEM LOAD<br><span class='metric-value'>{peak_load:.1f}%</span></div>", unsafe_allow_html=True)
    
    crit_color = "#e74c3c" if critical_servos > 0 else "#2ecc71"
    m3.markdown(f"<div class='metric-box'>CRITICAL FAILURES<br><span class='metric-value' style='color:{crit_color};'>{critical_servos}</span></div>", unsafe_allow_html=True)

    st.write("---")
    
    # 5. Build the Plotly Scatter Plot (Safety Matrix)
    # Assign colors based on threshold quadrant
    conditions = [
        (df['Load_Pct'] < load_threshold) & (df['Temp_C'] < temp_threshold),    # Safe (Green)
        (df['Load_Pct'] >= load_threshold) & (df['Temp_C'] < temp_threshold),   # Torque Warning (Orange)
        (df['Load_Pct'] < load_threshold) & (df['Temp_C'] >= temp_threshold),   # Thermal Warning (Orange)
        (df['Load_Pct'] >= load_threshold) & (df['Temp_C'] >= temp_threshold)   # Critical (Red)
    ]
    colors = ['#2ecc71', '#e67e22', '#e67e22', '#e74c3c']
    df['Color'] = np.select(conditions, colors)

    fig = go.Figure()

    # Add the scatter points for the servos
    fig.add_trace(go.Scatter(
        x=df['Load_Pct'],
        y=df['Temp_C'],
        mode='markers+text',
        marker=dict(size=14, color=df['Color'], opacity=0.9, line=dict(width=1, color='white')),
        text=df['Servo_ID'],
        textposition="top center",
        hoverinfo='text',
        hovertext=[f"{id}<br>Load: {l:.1f}%<br>Temp: {t:.1f}°C" for id, l, t in zip(df['Servo_ID'], df['Load_Pct'], df['Temp_C'])]
    ))

    # Formatting the Matrix Layout
    fig.update_layout(
        plot_bgcolor='#1e1e24',
        paper_bgcolor='#18181c',
        height=600,
        xaxis=dict(
            title="ACTUATOR TORQUE LOAD (%)",
            range=[0, 105],
            gridcolor='#333'
        ),
        yaxis=dict(
            title="ACTUATOR TEMPERATURE (°C)",
            range=[20, 100],
            gridcolor='#333'
        ),
        showlegend=False,
        shapes=[
            # Crosshairs to divide the quadrants based on dynamic slider thresholds
            dict(type="line", x0=load_threshold, x1=load_threshold, y0=20, y1=100, line=dict(color="#555", width=2, dash="dash")),
            dict(type="line", x0=0, x1=105, y0=temp_threshold, y1=temp_threshold, line=dict(color="#555", width=2, dash="dash"))
        ]
    )
    
    # Add floating text labels for quadrants
    fig.add_annotation(x=load_threshold / 2, y=temp_threshold - 5, text="Optimal Operation", showarrow=False, font=dict(color="#2ecc71", size=16))
    fig.add_annotation(x=load_threshold + 15, y=temp_threshold - 5, text="Torque Overload", showarrow=False, font=dict(color="#e67e22", size=16))
    fig.add_annotation(x=load_threshold / 2, y=temp_threshold + 5, text="Thermal Warning", showarrow=False, font=dict(color="#e67e22", size=16))
    fig.add_annotation(x=load_threshold + 15, y=temp_threshold + 5, text="Critical Failure Zone", showarrow=False, font=dict(color="#e74c3c", size=16))

    st.plotly_chart(fig, use_container_width=True)

    if st.button("Fetch New Telemetry Batch"):
        del st.session_state.telemetry_data
        st.rerun()