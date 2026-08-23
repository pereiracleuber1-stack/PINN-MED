import streamlit as st
import json
import datetime

st.set_page_config(page_title="Minuta Regulatória SaMD (P&D)", layout="wide")
st.title("📋 Minuta Técnica Preparatória para SaMD (ANVISA RDC 657) & Gestão de Riscos")
st.markdown("""
Documentação técnica de engenharia elaborada como **minuta preparatória de controles de design (IEC 62304 / ISO 14971)** 
para futuras submissões regulatórias e captação de recursos para validação clínica (**FINEP / PIPE**).
""")

tab_dossie, tab_fmea, tab_fomento = st.tabs([
    "🏛️ Minuta do Dossiê Técnico (IEC 62304)",
    "🛡️ Matriz Preliminar de Riscos (ISO 14971)",
    "💰 Propostas de Captação para Validação Clínica"
])

with tab_dossie:
    st.subheader("Enquadramento Alvo para Software Médico (SaMD)")
    c1, c2, c3 = st.columns(3)
    c1.info("**Classificação-Alvo:**\nClasse II (Suporte à Decisão Clínica)")
    c2.info("**Finalidade Prevista:**\nModelagem hemodinâmica preditiva sob supervisão médica")
    c3.info("**Diretriz de Ciclo de Vida:**\nAlinhamento aos controles da IEC 62304")

    st.write("""
    * **Natureza do Documento:** Minuta técnica de engenharia de software (TRL 4–5).
    * **Investigador / Autor:** Prof. Me. Cleuber Pereira Ramos (Mestre em Matemática - PROFMAT).
    * **Restrição Operacional:** O sistema não realiza diagnósticos autônomos ou intervenções terapêuticas sem validação médica presencial.
    """)

with tab_fmea:
    st.subheader("Matriz FMEA Preliminar de Gestão de Riscos Técnicos (ISO 14971)")
    fmea = [
        {"Perigo Identificado": "Falso Alarme por Ruído de Sensor", "Causa Primária": "Artefato em telemetria", "Mitigação no Algoritmo": "Calibração Conformal (PCCP 95%) e suavização EDO", "Severidade": "Moderada", "Risco Residual": "Monitorado"},
        {"Perigo Identificado": "Latência de Exame Laboratorial", "Causa Primária": "Demora na dosagem de lactato", "Mitigação no Algoritmo": "Assimilação contínua de PAM via Jump-ODE", "Severidade": "Grave", "Risco Residual": "Aceitável para P&D"},
        {"Perigo Identificado": "Acesso Não Autorizado a Dados", "Causa Primária": "Falha de credenciais", "Mitigação no Algoritmo": "Criptografia PBKDF2 e Ledger imutável SHA-256", "Severidade": "Crítica", "Risco Residual": "Controlado"}
    ]
    st.dataframe(fmea, use_container_width=True)

with tab_fomento:
    st.subheader("Proposta para Editais de Fomento à Validação Clínica Prospectiva")
    st.markdown("Estruturação técnica para captação de recursos a fundo perdido com foco em financiar ensaios em ambiente de UTI.")
    
    edital = st.selectbox("Edital Alvo", [
        "FAPEMAT / FAPESP - PIPE Fase 1 (Viabilidade Técnica e Econômica)",
        "FAPEMAT / FAPESP - PIPE Fase 2 (Desenvolvimento e Ensaio Clínico em UTI)",
        "FINEP - Subvenção Econômica em Saúde Digital e Inteligência Artificial"
    ])
    
    if st.button("Gerar Resumo Executivo para Submissão"):
        prop = {
            "titulo": "Validação Clínica e Escalabilidade de Redes Neurais Informadas pela Física na Predição Precoce de Choque Séptico",
            "edital": edital,
            "pesquisador_principal": "Prof. Me. Cleuber Pereira Ramos (PROFMAT)",
            "objetivo": "Transição de protótipo funcional TRL 4/5 para TRL 7 através de estudo clínico prospectivo em UTI e certificação ANVISA RDC 657."
        }
        st.success("✅ Resumo Estruturado com Sucesso!")
        st.json(prop)
