import streamlit as st
import json
import hashlib
import datetime

st.set_page_config(page_title="Regulação ANVISA & SaMD", layout="wide")
st.title("📋 Dossiê Técnico ANVISA (RDC 657/2022) & Gestão de Risco (ISO 14971)")
st.markdown("Documentação regulatória de **Software as a Medical Device (SaMD - Classe II)** e elaboração de propostas para editais **FINEP / PIPE-FAP**.")

if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Faça login para acessar os documentos regulatórios.")
    st.stop()

tab_anvisa, tab_fmea, tab_grant = st.tabs([
    "🏛️ Dossiê Técnico ANVISA / IEC 62304",
    "🛡️ Matriz de Gestão de Riscos (ISO 14971)",
    "💰 Gerador de Proposta de Fomento (FINEP / FAP)"
])

with tab_anvisa:
    st.subheader("Enquadramento Regulatório de Software Médico")
    c1, c2, c3 = st.columns(3)
    c1.info("**Classificação Regulatória:**\nSaMD Classe II (Risco Moderado)")
    c2.info("**Indicação de Uso:**\nSuporte à Decisão Clínica em Choque Séptico")
    c3.info("**Padrão do Ciclo de Vida:**\nABNT NBR IEC 62304:2016")

    st.markdown("#### Especificações do Dispositivo")
    st.write("""
    * **Nome do Produto:** SGP-PINN ENTERPRISE V25.0
    * **Finalidade:** Monitoramento hemodinâmico preditivo e quantificação de incerteza em leitos de Terapia Intensiva.
    * **Restrição Operacional:** Sistema de Suporte à Decisão Clínica (*CDS*), operando sob supervisão médica direta.
    """)

with tab_fmea:
    st.subheader("Matriz de Análise e Gerenciamento de Riscos Clínicos (ISO 14971:2020)")
    fmea_data = [
        {"Perigo / Modo de Falha": "Falso Alarme de Choque Séptico", "Causa Raiz": "Ruído em sinais vitais", "Severidade": "Moderada", "Mitigação Técnica": "Calibração Conformal (PCCP >= 95%) e RK4", "Risco Residual": "BAIXO"},
        {"Perigo / Modo de Falha": "Atraso na Detecção", "Causa Raiz": "Latência no resultado de lactato", "Severidade": "Crítica", "Mitigação Técnica": "Assimilação contínua de PAM via Neural Jump-ODE", "Risco Residual": "ACEITÁVEL"},
        {"Perigo / Modo de Falha": "Violação de Privacidade", "Causa Raiz": "Acesso não autorizado", "Severidade": "Grave", "Mitigação Técnica": "PBKDF2-HMAC-SHA256 e RBAC com logs imutáveis", "Risco Residual": "DESPREZÍVEL"}
    ]
    st.dataframe(fmea_data, use_container_width=True)

with tab_grant:
    st.subheader("Gerador de Proposta de Subvenção Econômica (FINEP / PIPE-FAP)")
    edital_nome = st.selectbox("Edital Alvo", [
        "FINEP - Inovação em Saúde e Inteligência Artificial",
        "FAPEMAT / FAPESP - PIPE Fase 1 (Prova de Viabilidade)",
        "FAPEMAT / FAPESP - PIPE Fase 2 (Desenvolvimento de Produto Industrial)"
    ])
    orcamento = st.selectbox("Orçamento Solicitado", ["R$ 300.000 (Fase 1)", "R$ 750.000 (Fase 2)", "R$ 1.500.000 (FINEP)"])
    prazo = st.selectbox("Duração", ["12 Meses", "18 Meses", "24 Meses"])
    
    if st.button("Gerar Pacote Técnico para Edital"):
        proposal = {
            "titulo_projeto": "SGP-PINN: Plataforma Fisiológica e Redes Neurais Informadas pela Física para Redução de Mortalidade por Choque Séptico",
            "edital_alvo": edital_nome,
            "orcamento": orcamento,
            "duracao": prazo,
            "sumario": "Sistema SaMD baseado em PINNs e DeepONet para detecção precoce de choque séptico com antecedência de 8.4h e conformidade ANVISA RDC 657.",
            "impacto_financeiro": "Redução estimada de 22% a 35% nos dias de leito de UTI por sepse grave."
        }
        json_str = json.dumps(proposal, indent=2, ensure_ascii=False)
        st.success("✅ Proposta Estruturada com Sucesso!")
        st.json(proposal)
        st.download_button("📥 Baixar Proposta (JSON)", json_str, "Proposta_Edital_FINEP.json", "application/json")
