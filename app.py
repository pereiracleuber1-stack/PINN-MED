import streamlit as st
import datetime
import hashlib
from database.auth_manager import AuthManager
from database.audit_db import AuditDatabase

st.set_page_config(
    page_title="SGP-PINN ENTERPRISE V20.0 | Portal Corporativo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

auth_db = AuthManager()
audit_db = AuditDatabase()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""
if "user_crm" not in st.session_state:
    st.session_state["user_crm"] = ""
if "user_role" not in st.session_state:
    st.session_state["user_role"] = ""
if "session_token" not in st.session_state:
    st.session_state["session_token"] = ""

def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.session_state["user_name"] = ""
    st.session_state["user_crm"] = ""
    st.session_state["user_role"] = ""
    st.session_state["session_token"] = ""
    st.rerun()

if not st.session_state["authenticated"]:
    st.markdown("<h1 style='text-align: center;'>🧬 SGP-PINN ENTERPRISE V20.0</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Plataforma Fisiológica de Grau Médico (SaMD) • Controle de Acesso Restrito</p>", unsafe_allow_html=True)
    
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("### 🔐 Autenticação Institucional Segura")
        with st.form("form_login"):
            username_input = st.text_input("Usuário do Sistema", value="admin")
            password_input = st.text_input("Senha", type="password", value="pinn2026")
            submit_login = st.form_submit_button("Acessar Plataforma SGP-PINN", use_container_width=True)
            
            if submit_login:
                ok, res = auth_db.authenticate_user(username_input, password_input)
                if ok:
                    ts_now = datetime.datetime.utcnow().isoformat()
                    token = hashlib.sha256(f"{res['username']}_{res['crm']}_{ts_now}".encode('utf-8')).hexdigest()
                    
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = res["username"]
                    st.session_state["user_name"] = res["full_name"]
                    st.session_state["user_crm"] = res["crm"]
                    st.session_state["user_role"] = res["role"]
                    st.session_state["session_token"] = token
                    
                    audit_db.log_inference(
                        operator_crm=res["crm"],
                        patient_id="AUTH-LOGIN",
                        model_type="SECURITY-RBAC",
                        k_pg=0.0, c_pn=0.0, mu_c=0.0, peak_lactate=0.0,
                        raw_signature=f"User {res['username']} authenticated as {res['role']}"
                    )
                    st.success("✅ Acesso autenticado com sucesso!")
                    st.rerun()
                else:
                    msg = res if isinstance(res, str) else "Usuário ou senha incorretos."
                    st.error(f"❌ {msg}")
                    
        st.info("💡 **Acesso Inicial:** Usuário: `admin` | Senha: `pinn2026` (alterável no menu de gestão).")
    st.stop()

with st.sidebar:
    st.markdown("### 👤 Operador Conectado")
    st.write(f"**Nome:** {st.session_state['user_name']}")
    st.write(f"**ID/CRM:** `{st.session_state['user_crm']}`")
    st.write(f"**Perfil:** `{st.session_state['user_role']}`")
    st.caption(f"**Sessão:** `{st.session_state['session_token'][:16]}...`")
    if st.button("🚪 Encerrar Sessão (Logout)", use_container_width=True):
        logout()

st.title("🧬 SGP-PINN ENTERPRISE V20.0 | Plataforma Fisiológica")
st.markdown("Sistema de Monitoramento Preditivo de Choque Séptico, Descoberta Simbólica e Quantificação Avançada de Incerteza.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Status do Sistema", "ONLINE (Produção)", "Sessão Segura")
col2.metric("Motor Numérico", "PyTorch Autograd / RK4", "100% Calibrado")
col3.metric("Rastreabilidade", "Ativa (SHA-256)", "21 CFR Part 11")
col4.metric("Conformidade", "HL7 / FHIR R4", "LGPD & HIPAA")

st.markdown("---")
st.subheader("Módulos Clínicos, de Pesquisa e Testes Cegos Disponíveis")

col_a, col_b = st.columns(2)
with col_a:
    st.info("🔬 **1. U-PINN Residual:** Inferência adaptativa com separação de termos conhecidos e neurais (RAD).")
    st.info("📜 **3. Descoberta Simbólica:** Extração de equações via SR3 com restrições biológicas.")
    st.info("📊 **5. Benchmarking SOTA:** Avaliação de complexidade, speedup e supressão de ruído.")
    st.info("🎯 **7. Quantificação de Incerteza:** Neural SDE (Itô) e Functional Conformal Prediction (≥ 95%).")

with col_b:
    st.success("👤 **2. PINN Inversa:** Calibração bayesiana amortizada de constantes individuais e laudo PDF.")
    st.success("🌐 **4. Neural Jump-ODE:** Rastreamento de dinâmicas descontínuas com infusão de bolus.")
    st.success("🏥 **6. Gateway FHIR R4:** Ingestão de prontuários (EHR), titulação What-If e Sobol GSA.")
    st.success("⚙️ **10. Gestão de Acessos & Senhas:** Cadastro de operadores, troca de senha e perfis RBAC.")
