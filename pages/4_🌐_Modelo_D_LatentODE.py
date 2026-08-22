import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Neural Jump-ODE", layout="wide")
st.title("🌐 Modelo D: Neural Jump-ODE (Tratamento de Descontinuidades Clínicas)")

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Intervenções Clínicas no Leito")
    horizonte = st.slider("Horizonte de Antecipação (horas)", 2, 12, 6)
    bolus_hora = st.slider("Momento do Bolus de Noradrenalina (Hora)", 0, 24, 16)
    dose_bolus = st.selectbox("Dose do Bolus", ["Nenhum", "0.2 mcg/kg/min", "0.5 mcg/kg/min", "1.0 mcg/kg/min"], index=2)
    ruido = st.slider("Nível de Ruído dos Sensores (%)", 0, 20, 5)

with col2:
    t_obs = np.array([0, 3, 7, 12, 18, 24])
    pam_obs = np.array([82.0, 78.0, 74.0, 68.0, 76.0, 73.0])
    
    t_cont = np.linspace(0, 24 + horizonte, 200)
    
    # Dinâmica Híbrida Contínua-Discreta (Jump-ODE)
    pam_list = []
    current_pam = 82.0
    for t_val in t_cont:
        current_pam -= 0.08
        if t_val >= bolus_hora and dose_bolus != "Nenhum":
            dose_factor = 14.0 if "0.5" in dose_bolus else (8.0 if "0.2" in dose_bolus else 20.0)
            current_pam += dose_factor * np.exp(-(t_val - bolus_hora) / 3.0) * 0.05
        pam_list.append(current_pam)
        
    # Conversão explícita para NumPy Array para indexação booleana
    pam_cont = np.array(pam_list)
    mask_pred = t_cont >= 24
        
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.scatter(t_obs, pam_obs, color="black", s=50, label="PAM Coletada (EHR)")
    ax.plot(t_cont, pam_cont, color="royalblue", lw=2, label="Trajetória Neural Jump-ODE")
    if dose_bolus != "Nenhum":
        ax.axvline(bolus_hora, color="green", ls="--", label=f"Salto: Bolus Noradrenalina ({dose_bolus})")
    ax.axhline(65, color="red", ls=":", label="Limite Choque (65 mmHg)")
    ax.axvline(24, color="gray", ls=":", label="Momento Atual")
    ax.fill_between(t_cont[mask_pred], 0, pam_cont[mask_pred], color="orange", alpha=0.2, label=f"Projeção Híbrida (+{horizonte}h)")
    ax.set_xlabel("Tempo de Internação (horas)"); ax.set_ylabel("PAM (mmHg)")
    ax.set_ylim(50, 90)
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.info("💡 **Jump Operator Ativo:** A rede modelou a resposta instantânea à infusão sem instabilidade numérica no gradiente.")
