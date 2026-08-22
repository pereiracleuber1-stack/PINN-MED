import streamlit as st
import pandas as pd
from database.audit_db import get_all_logs

st.set_page_config(page_title="Centro de Auditoria e Integridade", layout="wide")
st.title("🛡️ Centro de Auditoria e Integridade Criptográfica")
st.markdown("Registro imutável de todas as inferências, calibrações e relatórios gerados pela plataforma (21 CFR Part 11 / LGPD).")

logs = get_all_logs(limit=200)

if logs and len(logs) > 0:
    df = pd.DataFrame(logs, columns=["ID", "Timestamp (UTC)", "Operador (CRM)", "Paciente", "Módulo/Modelo", "k_pg", "c_pn", "mu_c", "Lactato Máx", "Assinatura SHA-256"])
    st.dataframe(df, use_container_width=True)
else:
    st.info("ℹ️ Nenhum log registrado ainda. Execute simulações nos módulos clínicos para alimentar o ledger criptográfico.")
