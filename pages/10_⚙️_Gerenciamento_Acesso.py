import streamlit as st
import pandas as pd
import json
from database.auth_manager import AuthManager

st.set_page_config(page_title="Gerenciamento Modular & Usuários", layout="wide")
st.title("⚙️ Gestão de Acessos, Licenciamento Modular & Senhas")
st.markdown("Controle de acesso granular por motor matemático e credenciamento institucional com criptografia **PBKDF2-HMAC-SHA256**.")

auth_db = AuthManager()

if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Faça login na página inicial para acessar o painel de gestão.")
    st.stop()

tab_perfil, tab_novo, tab_admin = st.tabs([
    "🔑 Meu Perfil & Trocar Senha",
    "➕ Licenciar Operador / Cliente Modular",
    "👥 Quadro Geral de Licenças e Módulos"
])

with tab_perfil:
    st.subheader(f"Perfil: {st.session_state.get('user_name', 'Pesquisador')}")
    c1, c2, c3 = st.columns(3)
    c1.info(f"**Usuário:** `{st.session_state.get('username', 'admin')}`")
    c2.info(f"**Titulação / Registro:**\n`{st.session_state.get('user_crm', 'Pesquisador de Matemática')}`")
    c3.info(f"**Função no Sistema:**\n`{st.session_state.get('user_role', 'Administrador')}`")
    
    st.markdown("#### Redefinição de Senha")
    with st.form("form_pwd"):
        old_p = st.text_input("Senha Atual", type="password")
        new_p = st.text_input("Nova Senha", type="password")
        confirm_p = st.text_input("Confirmar Nova Senha", type="password")
        btn = st.form_submit_button("Atualizar Minha Senha")
        if btn:
            if new_p != confirm_p:
                st.error("Confirmação de senha divergente.")
            elif len(new_p) < 6:
                st.error("A senha deve ter no mínimo 6 dígitos.")
            else:
                ok, msg = auth_db.change_password(st.session_state.get("username", "admin"), old_p, new_p)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

with tab_novo:
    st.subheader("Cadastrar Novo Operador ou Cliente Modular")
    st.markdown("Defina exatamente quais motores o cliente ou especialista terá acesso.")
    
    with st.form("form_novo_user"):
        c_u1, c_u2 = st.columns(2)
        novo_user = c_u1.text_input("Nome de Usuário (Login)", placeholder="ex: cliente.pharma")
        nova_pwd = c_u2.text_input("Senha Provisória", type="password")
        
        c_u3, c_u4 = st.columns(2)
        novo_nome = c_u3.text_input("Nome Completo / Instituição", placeholder="ex: Laboratório BioAnalysis")
        novo_reg = c_u4.text_input("Registro Profissional / Contrato", placeholder="ex: Contrato MaaS #2026-08")
        
        novo_papel = st.selectbox("Perfil", [
            "Pesquisador em Matemática Aplicada",
            "Cientista de Dados Farmacêutico",
            "Médico Intensivista (UTI)",
            "Auditor Externo / Regulatório",
            "Administrador de Sistema"
        ])
        
        st.markdown("#### 🎯 Motores Fisiológicos e Módulos Autorizados")
        mod_opcoes = [
            "TODOS (Acesso Pleno)",
            "Modelo A: U-PINN Residual",
            "Modelo B: PINN Inversa & Laudos",
            "Modelo C: Descoberta Simbólica (SINDy)",
            "Modelo D: Neural Jump-ODE",
            "Módulo FHIR Gateway & What-If",
            "Módulo Quantificação de Incerteza (UQ)",
            "Módulo Benchmark MIMIC-IV"
        ]
        mods_selecionados = st.multiselect("Selecione os módulos licenciados para este login:", mod_opcoes, default=["TODOS (Acesso Pleno)"])
        
        if st.form_submit_button("Emitir Licença de Acesso"):
            if not novo_user or not nova_pwd or not novo_nome:
                st.error("Preencha os campos obrigatórios.")
            else:
                ok, msg = auth_db.register_user(novo_user, nova_pwd, novo_nome, novo_reg, novo_papel, mods_selecionados)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

with tab_admin:
    st.subheader("Quadro de Clientes, Pesquisadores e Módulos Habilitados")
    users = auth_db.list_all_users()
    df_u = pd.DataFrame(users, columns=["ID", "Login", "Nome / Razão Social", "Identificação / Registro", "Perfil", "Módulos Licenciados", "Status", "Data de Emissão"])
    st.dataframe(df_u, use_container_width=True)
