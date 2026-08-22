import streamlit as st
from database.audit_db import init_db, get_all_logs

st.set_page_config(page_title="SGP-PINN Enterprise V20.0", page_icon="🧬", layout="wide")
init_db()

st.title("🧬 SGP-PINN ENTERPRISE V20.0 | Plataforma Fisiológica")
st.caption("Sistema de Monitoramento Preditivo de Choque Séptico & Descoberta de Dinâmicas Não Lineares")

col_head1, col_head2, col_head3, col_head4 = st.columns(4)
col_head1.metric("Status do Sistema", "ONLINE (Produção)")
col_head2.metric("Motor Numérico", "PyTorch Autograd / RK4")
col_head3.metric("Rastreabilidade", "Ativa (SHA-256)")
col_head4.metric("Conformidade", "LGPD / HIPAA Audit Ready")

st.markdown("---")
st.subheader("Módulos Clínicos e de Pesquisa Disponíveis")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.subheader("🔬 1. U-PINN Residual & Streaming UTI")
        st.write("Inferência em tempo real com separação de termos conhecidos e termo neural residual.")
        st.page_link("pages/1_🔬_Modelo_A_UPINN.py", label="Abrir Terminal de Monitoramento", icon="🔬")

    with st.container(border=True):
        st.subheader("📜 3. Descoberta Simbólica (PINN + SINDy)")
        st.write("Identificação de novas equações analíticas auditáveis e geração de formulações explícitas.")
        st.page_link("pages/3_📜_Modelo_C_SINDy.py", label="Abrir Laboratório Simbólico", icon="📜")

with c2:
    with st.container(border=True):
        st.subheader("👤 2. PINN Inversa & Calibração Individual")
        st.write("Adaptação dinâmica de constantes fisiológicas metabólicas e emissão de laudo em PDF.")
        st.page_link("pages/2_👤_Modelo_B_Inversa.py", label="Abrir Terminal de Calibração", icon="👤")

    with st.container(border=True):
        st.subheader("🛡️ 5. Centro de Auditoria e Logs Criptográficos")
        st.write("Painel de rastreabilidade completa de acessos, inferências e assinaturas de integridade.")
        st.page_link("pages/5_🛡️_Painel_Auditoria.py", label="Abrir Painel de Auditoria", icon="🛡️")
