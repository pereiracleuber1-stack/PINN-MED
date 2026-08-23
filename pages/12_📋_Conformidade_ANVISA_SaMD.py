import streamlit as st
import json
import hashlib
import datetime

st.set_page_config(page_title="Dossiê de Arquitetura Regulatória", layout="wide")
st.title("📋 Dossiê Preparatório para SaMD (ANVISA RDC 657) & Gestão de Riscos")
st.markdown("""
Documentação técnica de engenharia de software estruturada conforme os requisitos da **ABNT NBR IEC 62304:2016** 
e **ISO 14971:2020**, elaborada como etapa preparatória para submissões regulatórias e editais de fomento tecnológico (**FINEP / PIPE**).
""")

tab_anvisa, tab_fmea, tab_grant = st.tabs([
    "🏛️ Arquitetura Regulatória / IEC 62304",
    "🛡️ Gestão de Riscos Técnicos (ISO 14971)",
    "💰 Propostas de Fomento e Validação (PIPE / FINEP)"
])

with tab_anvisa:
    st.subheader("Enquadramento de Projeto para Software Médico (SaMD)")
    c1, c2, c3 = st.columns(3)
    c1.info("**Classificação-Alvo (ANVISA):**\nClasse II (Suporte à Decisão Clínica - CDS)")
    c2.info("**Finalidade Prevista (Intended Use):**\nModelagem hemodinâmica e alerta preditivo sob supervisão médica")
    c3.info("**Padrão de Ciclo de Vida Adotado:**\nABNT NBR IEC 62304 (Design Controls)")

    st.markdown("#### Especificações do Projeto de Pesquisa")
    st.write("""
    * **Título do Projeto:** SGP-PINN ENTERPRISE (Laboratório de Biomatemática Computacional)
    * **Investigador / Autor:** Prof. Me. Cleuber Pereira Ramos (Mestre em Matemática - PROFMAT)
    * **Status Tecnológico:** Nível de Maturidade TRL 4–5 (Prova de Conceito / Protótipo Funcional).
    * **Declaração de Limitação:** O sistema opera em regime estrito de *Clinical Decision Support (CDS)* para pesquisa, sem comando direto de atuadores hospitalares.
    """)

with tab_fmea:
    st.subheader("Matriz FMEA Preliminar de Gerenciamento de Riscos (ISO 14971)")
    fmea_data = [
        {"Perigo Potencial": "Falso Alarme por Ruído de Sinal", "Causa": "Artefato em telemetria", "Mitigação no Algoritmo": "Calibração Conformal (PCCP >= 95%) e suavização por EDO", "Risco Residual": "BAIXO"},
        {"Perigo Potencial": "Atraso de Detecção por Latência de Exame", "Causa": "Demora na dosagem de lactato", "Mitigação no Algoritmo": "Assimilação contínua de PAM via Neural Jump-ODE", "Risco Residual": "ACEITÁVEL"},
        {"Perigo Potencial": "Acesso Indevido a Dados", "Causa": "Falha de autenticação", "Mitigação no Algoritmo": "Criptografia PBKDF2-HMAC-SHA256 e trilha de auditoria SQLite imutável", "Risco Residual": "DESPREZÍVEL"}
    ]
    st.dataframe(fmea_data, use_container_width=True)

with tab_grant:
    st.subheader("Estruturação para Editais de Fomento à Validação Clínica")
    st.markdown("Proposta técnica para captação de recursos a fundo perdido com o objetivo de financiar a **validação clínica prospectiva em UTI hospitalar**.")
    
    edital_nome = st.selectbox("Edital Alvo", [
        "FAPEMAT / FAPESP - PIPE Fase 1 (Prova de Viabilidade Técnica e Comercial)",
        "FAPEMAT / FAPESP - PIPE Fase 2 (Desenvolvimento e Validação Clínica Hospitalar)",
        "FINEP - Subvenção Econômica à Inovação em Saúde Digital"
    ])
    orcamento = st.selectbox("Orçamento Proposto", ["R$ 300.000 (PIPE Fase 1)", "R$ 750.000 (PIPE Fase 2)", "R$ 1.500.000 (FINEP)"])
    
    if st.button("Gerar Estrutura de Proposta para Edital"):
        proposal = {
            "titulo": "Validação Clínica e Escalabilidade de Redes Neurais Informadas pela Física (PINNs) para Predição Precoce de Choque Séptico",
            "edital": edital_nome,
            "orcamento": orcamento,
            "pesquisador_responsavel": "Prof. Me. Cleuber Pereira Ramos (PROFMAT)",
            "objetivo_central": "Evolução do protótipo TRL 4/5 para TRL 7 através de ensaio observacional prospectivo em UTI e certificação ANVISA RDC 657.",
            "impacto_economico": "Redução estimada de 22% a 35% no tempo de internação por choque séptico, com economia média de R$ 1.2M/ano para hospitais de 50 leitos."
        }
        st.success("✅ Proposta Técnica Estruturada com Sucesso!")
        st.json(proposal)
