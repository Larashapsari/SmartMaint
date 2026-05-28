import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartMaint — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Rajdhani', sans-serif; }

.main { background: #0a0e1a; }
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 100%); }
[data-testid="stSidebar"] { background: #0d1225 !important; border-right: 1px solid #1e2d4a; }

.metric-card {
    background: linear-gradient(135deg, #111827, #1a2540);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,100,255,0.1);
}
.metric-value { font-family: 'Rajdhani', sans-serif; font-size: 2.5rem; font-weight: 700; color: #38bdf8; }
.metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

.status-critical { color: #ef4444; font-weight: 700; font-size: 1.1rem; }
.status-warning  { color: #f59e0b; font-weight: 700; font-size: 1.1rem; }
.status-normal   { color: #22c55e; font-weight: 700; font-size: 1.1rem; }

.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem; font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #38bdf8;
    padding-left: 12px;
    margin: 20px 0 12px 0;
}
.stButton>button {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white; border: none; border-radius: 8px;
    font-family: 'Rajdhani', sans-serif; font-weight: 600;
    font-size: 1rem; padding: 10px 30px;
    transition: all 0.2s;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(14,165,233,0.4); }

.schedule-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    try:
        import tensorflow as tf
        import joblib
        model  = tf.keras.models.load_model('model_lstm_rul.h5')
        scaler = joblib.load('scaler.pkl')
        with open('model_config.json') as f:
            config = json.load(f)
        return model, scaler, config, True
    except Exception as e:
        return None, None, None, False

model, scaler, config, model_loaded = load_model_and_scaler()

# ─── GENETIC ALGORITHM ─────────────────────────────────────────────────────────
def run_genetic_algorithm(machine_ruls, n_slots=30, n_gen=100, pop_size=200):
    import random
    COST_CORRECTIVE = 5000
    COST_PREVENTIVE = 1000
    n_machines = len(machine_ruls)

    def fitness(schedule):
        cost = 0
        for i, day in enumerate(schedule):
            rul = machine_ruls[i]
            if day > rul:
                cost += COST_CORRECTIVE
            elif day <= 0:
                cost += COST_PREVENTIVE * 1.5
            else:
                cost += COST_PREVENTIVE
        return cost

    # Init population
    population = [[random.randint(1, n_slots) for _ in range(n_machines)] for _ in range(pop_size)]

    for gen in range(n_gen):
        population = sorted(population, key=fitness)
        survivors = population[:pop_size//2]
        children = []
        while len(children) < pop_size//2:
            p1, p2 = random.sample(survivors, 2)
            cut = random.randint(1, n_machines-1)
            child = p1[:cut] + p2[cut:]
            # mutasi
            if random.random() < 0.3:
                idx = random.randint(0, n_machines-1)
                child[idx] = random.randint(1, n_slots)
            children.append(child)
        population = survivors + children

    best = sorted(population, key=fitness)[0]
    return best, fitness(best)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ SmartMaint")
    st.markdown("*Predictive Maintenance System*")
    st.markdown("---")

    st.markdown("### 🏭 Konfigurasi Pabrik")
    n_machines = st.slider("Jumlah Mesin", 2, 15, 5)
    planning_days = st.slider("Horizon Perencanaan (hari)", 7, 60, 30)

    st.markdown("---")
    st.markdown("### 📊 Mode Input")
    input_mode = st.radio("", ["Manual Input", "Demo Otomatis"])

    st.markdown("---")
    if model_loaded:
        st.success("✅ Model LSTM loaded")
    else:
        st.warning("⚠️ Model belum diload\n\nPastikan file berikut ada di folder yang sama:\n- model_lstm_rul.h5\n- scaler.pkl\n- model_config.json")
        st.info("Mode Demo aktif — menggunakan simulasi RUL")

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0'>
    <h1 style='font-family:Rajdhani; font-size:2.8rem; color:#e2e8f0; margin:0'>
        ⚙️ SMART<span style='color:#38bdf8'>MAINT</span>
    </h1>
    <p style='color:#64748b; font-size:0.95rem; margin:4px 0 0 0'>
        Sistem Prediksi & Optimasi Penjadwalan Preventive Maintenance — Industri Otomotif
    </p>
</div>
<hr style='border-color:#1e2d4a; margin: 10px 0 20px 0'>
""", unsafe_allow_html=True)

# ─── TAB LAYOUT ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi RUL", "📅 Jadwal Maintenance", "📈 Dashboard"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI RUL
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Input Data Sensor Mesin</div>', unsafe_allow_html=True)

    if input_mode == "Demo Otomatis":
        st.info("🤖 Mode Demo — data sensor digenerate otomatis")
        if st.button("🎲 Generate Data Sensor & Prediksi", use_container_width=True):
            np.random.seed(int(datetime.now().second))

            # Generate demo sensor data per mesin
            results = []
            for i in range(n_machines):
                # Simulasi degradasi: mesin dengan cycle lebih tinggi → RUL lebih rendah
                deg = np.random.uniform(0.3, 1.0)
                rul_sim = int(np.random.uniform(5, 150) * (1 - deg * 0.7))
                rul_sim = max(1, rul_sim)

                results.append({
                    'Mesin': f'M{i+1:02d}',
                    'Suhu (°C)': round(60 + deg * 30 + np.random.normal(0,2), 1),
                    'Vibrasi': round(0.5 + deg * 3.5 + np.random.normal(0,0.2), 2),
                    'Tekanan': round(4.5 - deg * 1.5 + np.random.normal(0,0.1), 2),
                    'RPM': int(1500 - deg * 200 + np.random.normal(0,20)),
                    'RUL Prediksi (cycle)': rul_sim,
                    'Status': '🔴 KRITIS' if rul_sim <= 20 else ('🟡 WASPADA' if rul_sim <= 50 else '🟢 NORMAL')
                })

            df_results = pd.DataFrame(results)
            st.session_state['machine_results'] = results

            # Tampilkan hasil
            st.markdown('<div class="section-title">Hasil Prediksi RUL</div>', unsafe_allow_html=True)

            # Metric cards
            cols = st.columns(min(n_machines, 5))
            for idx, row in enumerate(results[:5]):
                with cols[idx]:
                    color = "#ef4444" if row['RUL Prediksi (cycle)'] <= 20 else ("#f59e0b" if row['RUL Prediksi (cycle)'] <= 50 else "#22c55e")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style='color:#94a3b8; font-size:0.8rem'>{row['Mesin']}</div>
                        <div style='font-family:Rajdhani; font-size:2rem; font-weight:700; color:{color}'>
                            {row['RUL Prediksi (cycle)']}
                        </div>
                        <div style='color:#64748b; font-size:0.7rem'>cycles tersisa</div>
                        <div style='margin-top:6px'>{row['Status']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("")

            # Tabel lengkap
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Chart RUL
            fig = go.Figure()
            colors = ['#ef4444' if r['RUL Prediksi (cycle)'] <= 20 else ('#f59e0b' if r['RUL Prediksi (cycle)'] <= 50 else '#22c55e') for r in results]
            fig.add_trace(go.Bar(
                x=[r['Mesin'] for r in results],
                y=[r['RUL Prediksi (cycle)'] for r in results],
                marker_color=colors,
                text=[r['RUL Prediksi (cycle)'] for r in results],
                textposition='outside'
            ))
            fig.add_hline(y=20, line_dash="dash", line_color="#ef4444", annotation_text="Batas Kritis (20)")
            fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b", annotation_text="Batas Waspada (50)")
            fig.update_layout(
                title="Prediksi RUL per Mesin",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,22,41,0.8)',
                font_color='#e2e8f0',
                xaxis=dict(gridcolor='#1e2d4a'),
                yaxis=dict(gridcolor='#1e2d4a', title='RUL (cycles)'),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("Masukkan nilai sensor untuk **satu mesin**:")
        c1, c2, c3 = st.columns(3)
        with c1:
            temp    = st.number_input("🌡️ Suhu (°C)", 40.0, 120.0, 75.0, step=0.1)
            vibr    = st.number_input("📳 Vibrasi (mm/s)", 0.1, 10.0, 1.8, step=0.1)
        with c2:
            press   = st.number_input("💨 Tekanan (bar)", 1.0, 10.0, 4.2, step=0.1)
            rpm     = st.number_input("⚡ RPM", 500, 3000, 1450, step=10)
        with c3:
            oil     = st.number_input("🛢️ Level Oli (%)", 0, 100, 82)
            cycle   = st.number_input("🔄 Jumlah Cycle", 0, 100000, 15000, step=100)

        if st.button("🔮 Prediksi RUL Mesin Ini", use_container_width=True):
            # Estimasi RUL sederhana dari sensor
            degradasi = (temp - 60) / 60 + (vibr - 0.5) / 9.5 + (100 - oil) / 100
            rul_est = max(1, int(200 * (1 - degradasi / 3)))

            col1, col2, col3 = st.columns(3)
            color = "#ef4444" if rul_est <= 20 else ("#f59e0b" if rul_est <= 50 else "#22c55e")
            status = "🔴 KRITIS — Segera maintenance!" if rul_est <= 20 else ("🟡 WASPADA — Jadwalkan maintenance" if rul_est <= 50 else "🟢 NORMAL")

            with col1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">RUL Prediksi</div>
                    <div style='font-family:Rajdhani; font-size:3rem; font-weight:700; color:{color}'>{rul_est}</div>
                    <div class="metric-label">cycles tersisa</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                maint_date = datetime.now() + timedelta(days=rul_est)
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Estimasi Tanggal Rusak</div>
                    <div style='font-family:Rajdhani; font-size:1.6rem; font-weight:700; color:#38bdf8'>{maint_date.strftime('%d %b %Y')}</div>
                    <div class="metric-label">jika tidak dimaintenance</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Status Mesin</div>
                    <div style='font-size:1.1rem; margin-top:10px'>{status}</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — JADWAL MAINTENANCE (GA)
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Optimasi Jadwal Maintenance — Genetic Algorithm</div>', unsafe_allow_html=True)

    st.markdown("Masukkan prediksi RUL untuk setiap mesin, lalu GA akan mencari jadwal maintenance paling optimal (biaya minimum):")

    # Input RUL per mesin
    rul_inputs = []
    cols_input = st.columns(min(n_machines, 5))
    for i in range(n_machines):
        with cols_input[i % 5]:
            # Ambil dari session state jika ada
            default_rul = 25
            if 'machine_results' in st.session_state and i < len(st.session_state['machine_results']):
                default_rul = st.session_state['machine_results'][i]['RUL Prediksi (cycle)']
            rul_val = st.number_input(f"M{i+1:02d} RUL", 1, 200, default_rul, key=f"rul_{i}")
            rul_inputs.append(rul_val)

    st.markdown("")
    if st.button("🧬 Jalankan Genetic Algorithm — Cari Jadwal Optimal", use_container_width=True):
        with st.spinner("Menjalankan Genetic Algorithm... 🧬"):
            best_schedule, total_cost = run_genetic_algorithm(rul_inputs, n_slots=planning_days)

        st.success(f"✅ Jadwal optimal ditemukan! Estimasi total biaya: **USD {total_cost:,}**")

        st.markdown('<div class="section-title">Jadwal Maintenance Optimal</div>', unsafe_allow_html=True)

        # Gantt chart
        today = datetime.now()
        gantt_data = []
        schedule_rows = []

        for i, day in enumerate(best_schedule):
            rul = rul_inputs[i]
            maint_date = today + timedelta(days=day)
            fail_date  = today + timedelta(days=rul)
            status = "🔴 TERLAMBAT" if day > rul else ("🟡 Mepet" if day >= rul - 3 else "✅ Tepat Waktu")
            biaya = "USD 5,000 (corrective)" if day > rul else "USD 1,000 (preventive)"

            schedule_rows.append({
                'Mesin': f'M{i+1:02d}',
                'RUL (cycles)': rul,
                'Jadwal Maintenance': maint_date.strftime('%d %b %Y'),
                'Hari ke-': day,
                'Status': status,
                'Estimasi Biaya': biaya
            })

            gantt_data.append(dict(
                Task=f"M{i+1:02d}",
                Start=today.strftime('%Y-%m-%d'),
                Finish=maint_date.strftime('%Y-%m-%d'),
                Status="Terlambat" if day > rul else "Optimal"
            ))

        df_schedule = pd.DataFrame(schedule_rows)
        st.dataframe(df_schedule, use_container_width=True, hide_index=True)

        # Bar chart jadwal
        fig2 = go.Figure()
        bar_colors = ['#ef4444' if r['Status'] == '🔴 TERLAMBAT' else '#22c55e' for r in schedule_rows]
        fig2.add_trace(go.Bar(
            x=[r['Mesin'] for r in schedule_rows],
            y=[r['Hari ke-'] for r in schedule_rows],
            name='Hari Maintenance',
            marker_color=bar_colors,
            text=[f"Hari {r['Hari ke-']}" for r in schedule_rows],
            textposition='outside'
        ))
        fig2.add_trace(go.Scatter(
            x=[r['Mesin'] for r in schedule_rows],
            y=[r['RUL (cycles)'] for r in schedule_rows],
            mode='markers+lines',
            name='RUL Mesin',
            marker=dict(color='#f59e0b', size=10, symbol='diamond'),
            line=dict(color='#f59e0b', dash='dot')
        ))
        fig2.update_layout(
            title="Jadwal Maintenance vs RUL Mesin",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,22,41,0.8)',
            font_color='#e2e8f0',
            xaxis=dict(gridcolor='#1e2d4a'),
            yaxis=dict(gridcolor='#1e2d4a', title='Hari ke-'),
            height=380,
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Ringkasan biaya
        c1, c2, c3 = st.columns(3)
        n_corrective = sum(1 for r in schedule_rows if '5,000' in r['Estimasi Biaya'])
        n_preventive = n_machines - n_corrective
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Total Biaya Estimasi</div>
                <div class="metric-value">USD {total_cost:,}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Preventive Maintenance</div>
                <div style='font-family:Rajdhani;font-size:2.5rem;font-weight:700;color:#22c55e'>{n_preventive}</div>
                <div class="metric-label">mesin</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Corrective (Terlambat)</div>
                <div style='font-family:Rajdhani;font-size:2.5rem;font-weight:700;color:#ef4444'>{n_corrective}</div>
                <div class="metric-label">mesin</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Dashboard Overview Pabrik</div>', unsafe_allow_html=True)

    # Simulasi data historis
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    df_history = pd.DataFrame({
        'Tanggal': dates,
        'OEE (%)': np.clip(85 + np.cumsum(np.random.normal(0, 1, 30)), 70, 98),
        'Downtime (jam)': np.abs(np.random.normal(2, 1.5, 30)),
        'Biaya Maintenance (USD)': np.random.choice([1000, 5000], 30, p=[0.8, 0.2])
    })

    # OEE trend
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_history['Tanggal'], y=df_history['OEE (%)'],
        fill='tozeroy', fillcolor='rgba(56,189,248,0.1)',
        line=dict(color='#38bdf8', width=2),
        name='OEE (%)'
    ))
    fig3.add_hline(y=85, line_dash="dash", line_color="#22c55e", annotation_text="Target OEE 85%")
    fig3.update_layout(
        title="Trend OEE Pabrik (30 Hari Terakhir)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,22,41,0.8)',
        font_color='#e2e8f0',
        xaxis=dict(gridcolor='#1e2d4a'),
        yaxis=dict(gridcolor='#1e2d4a', title='OEE (%)'),
        height=300
    )
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig4 = px.bar(df_history.tail(14), x='Tanggal', y='Downtime (jam)',
                      title="Downtime 14 Hari Terakhir",
                      color='Downtime (jam)', color_continuous_scale='RdYlGn_r')
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,22,41,0.8)',
                           font_color='#e2e8f0', height=280)
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        total_biaya = df_history['Biaya Maintenance (USD)'].sum()
        biaya_preventive = df_history[df_history['Biaya Maintenance (USD)']==1000].shape[0] * 1000
        biaya_corrective = total_biaya - biaya_preventive
        fig5 = go.Figure(go.Pie(
            labels=['Preventive', 'Corrective'],
            values=[biaya_preventive, biaya_corrective],
            hole=0.5,
            marker_colors=['#22c55e', '#ef4444']
        ))
        fig5.update_layout(
            title="Komposisi Biaya Maintenance",
            paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=280
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Summary metrics
    st.markdown('<div class="section-title">Ringkasan 30 Hari</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Rata-rata OEE</div>
            <div class="metric-value">{df_history['OEE (%)'].mean():.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Downtime</div>
            <div class="metric-value">{df_history['Downtime (jam)'].sum():.0f}h</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Biaya</div>
            <div class="metric-value">${total_biaya:,}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        saved = biaya_corrective * 0.8
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Penghematan Est.</div>
            <div style='font-family:Rajdhani;font-size:2rem;font-weight:700;color:#22c55e'>${saved:,.0f}</div>
        </div>""", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e2d4a; margin-top:30px'>
<p style='text-align:center; color:#334155; font-size:0.8rem'>
SmartMaint — Predictive Maintenance System | Industri Otomotif | LSTM + Genetic Algorithm
</p>
""", unsafe_allow_html=True)
