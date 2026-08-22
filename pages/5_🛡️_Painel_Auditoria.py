import streamlit as st
import pandas as pd
from database.audit_db import get_all_logs

st.set_page_config(page_title="Painel de Auditoria", layout="wide")
st.title("🛡️ Centro de Auditoria e Integridade Criptográfica")
st.markdown("Registro imutável de todas as inferências, calibrações e relatórios gerados pela plataforma.")

logs = get_all_logs(limit=200)

if logs:
    df = pd.DataFrame(logs, columns=["ID", "Timestamp (UTC)", "Operador", "Paciente", "Ação", "Modelo", "Risco", "Hash Criptográfico SHA-256"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("Exportar Relatório de Conformidade")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Exportar Log de Auditoria (CSV)",
        data=csv,
        file_name="audit_logs_pinn_enterprise.csv",
        mime="text/csv"
    )
else:
    st.info("Nenhum registro de auditoria encontrado até o momento.")
