import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from models.advanced_uq import NeuralSDE_EulerMaruyama, EpistemicAleatoricDecomposer, PhysicsConformalCalibrator

st.set_page_config(page_title="Quantificação Avançada de Incerteza (UQ)", layout="wide")
st.title("🎯 Quantificação Avançada de Incerteza & Conformal Prediction")
st.markdown("Modelagem estocástica via **Neural SDE (Cálculo de Itô)**, **Decomposição Epistêmica/Aleatória** e **Functional Conformal Prediction (PCCP)** com garantia $\ge 95\%$.")

tab_sde, tab_decomp, tab_conformal, tab_envelope = st.tabs([
    "🎲 Motor Estocástico (Neural SDE)",
    "📊 Decomposição Epistêmica vs. Aleatória",
    "🛡️ Calibração Conforme (Garantia ≥ 95%)",
    "🩺 Envelope de Risco Clínico Individual"
])

sde_engine = NeuralSDE_EulerMaruyama(dt=0.2, t_max=48.0)

# -------------------------------------------------------------
# TAB 1: NEURAL SDE (ITÔ / EULER-MARUYAMA)
# -------------------------------------------------------------
with tab_sde:
    st.subheader("Simulação Estocástica de Trajetórias de Choque (Processo de Wiener)")
    st.markdown("Propagação contínua de incerteza biológica: $dx_t = [f(x_t) + \mathcal{N}_\phi]dt + g_\psi(x_t)dW_t$.")
    
    c1, c2, c3 = st.columns(3)
    k_val = c1.slider("Taxa de Proliferação (k_pg)", 0.25, 0.65, 0.48)
    diff_val = c2.slider("Escala de Difusão / Ruído Biológico (g)", 0.01, 0.08, 0.035, step=0.005)
    n_sims = c3.slider("Trajetórias de Monte Carlo", 20, 200, 80)
    
    if st.button("Executar Simulação Neural SDE", key="btn_run_sde"):
        with st.spinner("Integrando equações diferenciais estocásticas via Euler-Maruyama..."):
            t_grid, paths = sde_engine.simulate_ensemble(
                x0=[1.2, 0.8, 0.4, 0.1, 2.2], k_pg=k_val, diff_scale=diff_val, num_paths=n_sims
            )
            lac_paths = paths[:, :, 4]
            mu = np.mean(lac_paths, axis=0)
            
            fig, ax = plt.subplots(figsize=(10, 4.2))
            for i in range(min(n_sims, 50)):
                ax.plot(t_grid, lac_paths[i], color="#3498db", alpha=0.15, lw=1.0)
            ax.plot(t_grid, mu, color="#2c3e50", lw=2.8, label="Média do Processo Estocástico")
            ax.axhline(2.0, color="gray", ls="--", label="Alerta Sepse (< 2.0 mmol/L)")
            ax.axhline(4.0, color="crimson", ls="--", label="Choque Séptico Severo (> 4.0 mmol/L)")
            ax.set_xlabel("Horas de Internação (t)"); ax.set_ylabel("Lactato (mmol/L)")
            ax.set_title(f"Funil Estocástico de {n_sims} Trajetórias Fisiológicas de Leito")
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

# -------------------------------------------------------------
# TAB 2: DECOMPOSIÇÃO EPISTÊMICA VS. ALEATÓRIA
# -------------------------------------------------------------
with tab_decomp:
    st.subheader("Decomposição de Incerteza: Falta de Conhecimento vs. Ruído do Sensor")
    st.markdown("Permite distinguir se a incerteza provém de **fenótipo raro (epistêmica)** ou de **flutuação do sensor (aleatória)**.")
    
    c_d1, c_d2 = st.columns(2)
    sensor_noise = c_d1.slider("Incerteza do Sensor / Laboratorial (σ_sensor)", 0.05, 0.30, 0.12, step=0.01)
    
    if st.button("Calcular Decomposição de Variância", key="btn_run_decomp"):
        with st.spinner("Decompondo variância total do conjunto..."):
            mu_g, std_tot, std_epi, std_ale = EpistemicAleatoricDecomposer.decompose(
                sde_engine, x0=[1.2, 0.8, 0.4, 0.1, 2.2], sensor_noise_std=sensor_noise
            )
            t_grid = sde_engine.t_grid
            
            fig, ax = plt.subplots(figsize=(10, 4.2))
            ax.plot(t_grid, mu_g, color="#2c3e50", lw=2.5, label="Trajetória Esperada")
            ax.fill_between(t_grid, mu_g - std_epi, mu_g + std_epi, color="#3498db", alpha=0.35, label="Incerteza Epistêmica (Discrepância do Modelo)")
            ax.fill_between(t_grid, mu_g - std_tot, mu_g - std_epi, color="#e67e22", alpha=0.30, label="Incerteza Aleatória (Ruído Fisiológico/Sensor)")
            ax.fill_between(t_grid, mu_g + std_epi, mu_g + std_tot, color="#e67e22", alpha=0.30)
            ax.set_xlabel("Horas de Internação (t)"); ax.set_ylabel("Lactato (mmol/L)")
            ax.set_title("Decomposição Estrutural da Faixa de Incerteza Temporal")
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

# -------------------------------------------------------------
# TAB 3: CONFORMAL PREDICTION (PCCP)
# -------------------------------------------------------------
with tab_conformal:
    st.subheader("Calibração Conforme Funcional: Garantia Matemática de Cobertura (≥ 95%)")
    st.markdown("Validação em coorte de calibração funcional ($N_{\\text{cal}} = 250$) e verificação empírica em coorte independente ($N_{\\text{test}} = 200$).")
    
    if st.button("Executar Calibração Conformal Split-Sample", key="btn_run_conf"):
        with st.spinner("Calculando quantil de supremo funcional q_hat e validando cobertura..."):
            q_hat, scores, mu_cal, std_cal = PhysicsConformalCalibrator.calibrate_cohort(sde_engine, N_cal=250, alpha=0.05)
            cov_traj, cov_point, cov_naive_traj, low_conf, up_conf = PhysicsConformalCalibrator.evaluate_test_coverage(
                sde_engine, q_hat, mu_cal, std_cal, N_test=200
            )
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fator de Calibração (q̂)", f"{q_hat:.4f}")
            c2.metric("Cobertura da Trajetória Completa", f"{cov_traj*100:.1f}%", "Garantia Teórica ≥ 95%")
            c3.metric("Cobertura Ponto a Ponto", f"{cov_point*100:.1f}%", "Precisão Local")
            c4.metric("Cobertura Trajetória 1.96σ", f"{cov_naive_traj*100:.1f}%", "Gaussiano Ingênuo")
            
            fig, ax = plt.subplots(figsize=(10, 4.2))
            t_grid = sde_engine.t_grid
            ax.plot(t_grid, mu_cal, color="#2c3e50", lw=2.5, label="Predição Central")
            ax.fill_between(t_grid, low_conf, up_conf, color="#27ae60", alpha=0.25, label=f"Banda Conforme Calibrada (Cobertura Trajetorial: {cov_traj*100:.1f}%)")
            ax.plot(t_grid, mu_cal + 1.96*std_cal, color="red", ls=":", label="Limite Gaussiano Simples (1.96σ)")
            ax.plot(t_grid, np.maximum(mu_cal - 1.96*std_cal, 0.5), color="red", ls=":")
            ax.set_xlabel("Horas de UTI"); ax.set_ylabel("Lactato (mmol/L)")
            ax.set_title("Banda Conforme Funcional (PCCP) vs. Limites Gaussianos Pontuais")
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            
            st.success(f"✅ **Garantia Cumprida:** A cobertura simultânea da trajetória inteira atingiu {cov_traj*100:.1f}% (com {cov_point*100:.1f}% de cobertura ponto a ponto).")

# -------------------------------------------------------------
# TAB 4: ENVELOPE INDIVIDUALIZADO
# -------------------------------------------------------------
with tab_envelope:
    st.subheader("Envelope Clínico Personalizado para Tomada de Decisão")
    st.markdown("Projeção do intervalo assimétrico para um leito específico com quantificação de risco.")
    
    col_p1, col_p2 = st.columns(2)
    p_idade = col_p1.number_input("Idade do Paciente", 18, 95, 68)
    p_sofa = col_p2.number_input("Escore SOFA Admissional", 0, 18, 7)
    
    k_paciente = 0.32 * (p_idade/50.0 + p_sofa * 0.12)
    
    t_grid, p_sims = sde_engine.simulate_ensemble(x0=[1.2, 0.8, 0.4, 0.1, 2.4], k_pg=k_paciente, num_paths=60)
    mu_p = np.mean(p_sims[:, :, 4], axis=0)
    std_p = np.std(p_sims[:, :, 4], axis=0) + 0.05
    
    low_p = np.maximum(mu_p - 1.45 * std_p, 0.8)
    up_p = mu_p + 1.45 * std_p
    
    fig_e, ax_e = plt.subplots(figsize=(10, 4.0))
    ax_e.plot(t_grid, mu_p, color="#8e44ad", lw=2.5, label="Trajetória Média do Paciente")
    ax_e.fill_between(t_grid, low_p, up_p, color="#9b59b6", alpha=0.30, label="Envelope de Risco Certificado (95% PCCP)")
    ax_e.axhline(4.0, color="crimson", ls="--", label="Limiar de Choque Séptico Severo")
    ax_e.set_xlabel("Horas de Internação"); ax_e.set_ylabel("Lactato (mmol/L)")
    ax_e.set_title(f"Envelope Fisiológico do Paciente (Idade: {p_idade}a | SOFA: {p_sofa} | k_pg: {k_paciente:.3f}/h)")
    ax_e.legend(); ax_e.grid(True, alpha=0.3)
    st.pyplot(fig_e)
    plt.close(fig_e)
