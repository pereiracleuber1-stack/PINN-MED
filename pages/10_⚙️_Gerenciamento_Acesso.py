import streamlit as st
import pandas as pd
from database.auth_manager import AuthManager

st.set_page_config(page_title="Gerenciamento de Usuários & Senhas", layout="wide")
st.title("⚙️ Gerenciamento de Usuários, Senhas & Perfis (RBAC)")
st.markdown("Controle de acesso, cadastro de operadores clínicos, redefinição de senhas com criptografia **PBKDF2-HMAC-SHA256**.")

auth_db = AuthManager()

if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Faça login na página inicial para acessar as funções de administração.")
    st.stop()

tab_perfil, tab_novo, tab_admin = st.tabs([
    "🔑 Meu Perfil & Trocar Senha",
    "➕ Cadastrar Novo Operador",
    "👥 Quadro de Usuários & Permissões"
])

with tab_perfil:
    st.subheader(f"Perfil do Operador: {st.session_state.get('user_name', 'Operador')}")
    c1, c2, c3 = st.columns(3)
    c1.info(f"**Usuário:** `{st.session_state.get('username', 'admin')}`")
    c2.info(f"**CRM / Identificação:** `{st.session_state.get('user_crm', 'CRM-MT 10452')}`")
    c3.info(f"**Perfil de Acesso:** `{st.session_state.get('user_role', 'Administrador de Sistema')}`")
    
    st.markdown("#### Redefinição de Senha")
    with st.form("form_change_pwd"):
        old_p = st.text_input("Senha Atual", type="password")
        new_p = st.text_input("Nova Senha", type="password")
        confirm_p = st.text_input("Confirmar Nova Senha", type="password")
        btn_change = st.form_submit_button("Atualizar Minha Senha")
        
        if btn_change:
            if not old_p or not new_p:
                st.error("Preencha todos os campos.")
            elif new_p != confirm_p:
                st.error("A nova senha e a confirmação não coincidem.")
            elif len(new_p) < 6:
                st.error("A nova senha deve possuir no mínimo 6 caracteres.")
            else:
                ok, msg = auth_db.change_password(st.session_state.get("username", "admin"), old_p, new_p)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

with tab_novo:
    st.subheader("Cadastro de Novo Operador / Especialista")
    st.markdown("Crie credenciais para médicos intensivistas, pesquisadores ou auditores externos.")
    
    with st.form("form_register_user"):
        col_u1, col_u2 = st.columns(2)
        novo_username = col_u1.text_input("Nome de Usuário (Login)", placeholder="ex: dr.souza")
        nova_senha = col_u2.text_input("Senha Inicial Provisória", type="password")
        
        col_u3, col_u4 = st.columns(2)
        novo_nome = col_u3.text_input("Nome Completo", placeholder="ex: Dr. Carlos Eduardo Souza")
        novo_crm = col_u4.text_input("CRM / Matrícula Institucional", placeholder="ex: CRM-SP 189204")
        
        novo_papel = st.selectbox(
            "Papel no Sistema (Perfil RBAC)",
            [
                "Médico Intensivista (UTI)",
                "Pesquisador / Data Scientist",
                "Auditor Clínico / Comitê Externo",
                "Administrador de Sistema"
            ]
        )
        
        btn_cadastrar = st.form_submit_button("Criar Registro de Operador")
        
        if btn_cadastrar:
            if not novo_username or not nova_senha or not novo_nome or not novo_crm:
                st.error("Preencha todos os campos obrigatórios.")
            else:
                ok, msg = auth_db.register_user(novo_username, nova_senha, novo_nome, novo_crm, novo_papel)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

with tab_admin:
    st.subheader("Quadro Geral de Operadores Cadastrados")
    st.markdown("Visualização de contas ativas e permissões institucionais.")
    
    users = auth_db.list_all_users()
    df_users = pd.DataFrame(users, columns=["ID", "Login", "Nome Completo", "CRM / Matrícula", "Perfil RBAC", "Status", "Data de Criação"])
    st.dataframe(df_users, use_container_width=True)
    
    st.markdown("#### Gerenciar Status da Conta")
    col_s1, col_s2, col_s3 = st.columns(3)
    user_target = col_s1.selectbox("Selecione o Usuário", [u[1] for u in users if u[1] != "admin"])
    novo_status = col_s2.selectbox("Novo Status", ["ACTIVE", "INACTIVE"])
    if col_s3.button("Atualizar Status do Operador"):
        auth_db.set_user_status(user_target, novo_status)
        st.success(f"Status do usuário '{user_target}' alterado para {novo_status}!")
        st.rerun()
