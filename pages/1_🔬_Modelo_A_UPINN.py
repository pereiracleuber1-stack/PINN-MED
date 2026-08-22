import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.telemetry_stream import fetch_telemetry_tick
from database.audit_db import log_event

st.set_page_config(page_title="U-PINN Residual UTI", layout="wide")
st.title("🔬 Modelo A: U-PINN Residual & Streaming Contínuo")

# Painel de Identificação Médica
with st.sidebar:
    st.header("Sessão Clínica")
    operator = st.text_input("Operador (CRM / ID)", value="CRM-MT 8492")
    patient_id = st.text_input("ID do Paciente", value="PAC-UTI-084")
    streaming_mode = st.toggle("Ativar Streaming de Telemetria (Tempo Real)", value=False)
    janela_horas = st.slider("Janela de Observação (Horas)", 12, 72, 36)

if streaming_mode:
    st.info("📡 **Streaming de Leito Hospitalar Ativo:** Recebendo pacotes de dados contínuos...")
    placeholder_metrics = st.empty()
    placeholder_charts = st.empty()
    
    # Histórico simulado de streaming
    hist_t = []
    hist_pam = []
    hist_lact = []
    
    for h in range(1, janela_horas + 1):
        tick = fetch_telemetry_tick(h)
        hist_t.append(tick["hour"])
        hist_pam.append(tick["pam"])
        hist_lact.append(tick["lactato"])
        
        with placeholder_metrics.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PAM Atual", f"{tick['pam']} mmHg", delta=f"{tick['pam']-80:.1f}")
            m2.metric("Lactato", f"{tick['lactato']} mmol/L", delta=f"{tick['lactato']-1.5:.2f}", delta_color="inverse")
            m3.metric("Freq. Cardíaca", f"{tick['fc']} bpm")
            m4.metric("SpO2", f"{tick['spo2']}%")
            
        with placeholder_charts.container():
            fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
            ax[0].plot(hist_t, hist_pam, color="royalblue", lw=2, marker="o", markersize=3, label="PAM (mmHg)")
            ax[0].axhline(65, color="red", ls="--", label="Limite Crítico (65 mmHg)")
            ax[0].set_title("Streaming Hemodinâmico"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
            
            ax[1].plot(hist_t, hist_lact, color="crimson", lw=2, marker="s", markersize=3, label="Lactato (mmol/L)")
            ax[1].axhline(2.0, color="orange", ls="--", label="Alerta Sepse")
            ax[1].axhline(4.0, color="red", ls="--", label="Choque Séptico")
            ax[1].set_title("Evolução Metabólica"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        time.sleep(0.08)
        
    log_event(operator, patient_id, "Streaming Telemetria", "U-PINN-A", {"horas": janela_horas}, "ALERTA" if hist_lact[-1] > 2.0 else "NORMAL")
else:
    st.markdown("Selecione os parâmetros e execute a inferência determinística do operador diferencial:")
    if st.button("Executar Inferência Estática e Registrar Auditoria", use_container_width=True):
        t = np.linspace(0, janela_horas, 200)
        patogeno = np.exp(0.08 * t) / (1 + 0.05 * np.exp(0.08 * t))
        pam = 85.0 - 15.0 / (1 + np.exp(-(t - 18) / 3))
        resíduo = 0.18 * np.exp(-((t - 18) ** 2) / 25)
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
        ax[0].plot(t, patogeno, color="crimson", lw=2, label="Patógeno P(t)"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
        ax[1].plot(t, pam, color="royalblue", lw=2, label="PAM (mmHg)"); ax[1].axhline(65, color="red", ls="--"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
        ax[2].plot(t, resíduo, color="purple", lw=2, label="Termo Descoberto N_phi"); ax[2].fill_between(t, 0, resíduo, color="purple", alpha=0.2); ax[2].legend(); ax[2].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        hash_id = log_event(operator, patient_id, "Inferência Estática", "U-PINN-A", {"janela": janela_horas}, "MONITORAMENTO")
        st.success(f"Inferência registrada com sucesso no banco de auditoria. Assinatura: `{hash_id[:16]}...`")
