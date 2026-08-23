import streamlit as st
import datetime
import hashlib
from database.auth_manager import AuthManager
from database.audit_db import AuditDatabase

st.set_page_config(
    page_title="SGP-PINN ENTERPRISE | Console Científico de P&D",
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
if "user_reg" not in st.session_state:
    st.session_state["user_reg"] = ""
if "user_role" not in st.session_state:
    st.session_state["user_role"] = ""
if "allowed_modules" not in st.session_state:
    st.session_state["allowed_modules"] = ["ALL"]
if "session_token" not in st.session_state:
    st.session_state["session_token"] = ""

def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.session_state["user_name"] = ""
    st.session_state["user_reg"] = ""
    st.session_state["user_role"] = ""
    st.session_state["allowed_modules"] = ["ALL"]
    st.session_state["session_token"] = ""
    st.rerun()

if not st.session_state["authenticated"]:
    st.markdown("<h1 style='text-align: center;'>🧬 SGP-PINN ENTERPRISE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Console Científico de P&D • Redes Neurais Informadas pela Física & Modelagem Biomatemática</p>", unsafe_allow_html=True)
    
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("### 🔐 Autenticação Institucional Segura")
        with st.form("form_login"):
            username_input = st.text_input("Usuário do Sistema / Licença", value="admin")
            password_input = st.text_input("Senha", type="password", value="pinn2026")
            submit_login = st.form_submit_button("Acessar Console de Pesquisa", use_container_width=True)
            
            if submit_login:
                ok, res = auth_db.authenticate_user(username_input, password_input)
                if ok:
                    ts_now = datetime.datetime.utcnow().isoformat()
                    token = hashlib.sha256(f"{res['username']}_{res['reg_id']}_{ts_now}".encode('utf-8')).hexdigest()
                    
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = res["username"]
                    st.session_state["user_name"] = res["full_name"]
                    st.session_state["user_reg"] = res["reg_id"]
                    st.session_state["user_role"] = res["role"]
                    st.session_state["allowed_modules"] = res.get("allowed_modules", ["ALL"])
                    st.session_state["session_token"] = token
                    
                    try:
                        audit_db.log_inference(
                            operator_crm=res["reg_id"],
                            patient_id="AUTH-SESSION",
                            model_type="SECURITY-RBAC",
                            raw_signature=f"User {res['username']} logged in"
                        )
                    except Exception:
                        pass
                        
                    st.success("✅ Acesso autenticado com sucesso!")
                    st.rerun()
                else:
                    msg = res if isinstance(res, str) else "Credenciais incorretas."
                    st.error(f"❌ {msg}")
                    
        st.info("💡 **Acesso Inicial:** Usuário: `admin` | Senha: `pinn2026` (redefinível no painel de controle).")
    st.stop()

with st.sidebar:
    st.markdown("### 👤 Operador Conectado")
    st.write(f"**Pesquisador:** {st.session_state['user_name']}")
    st.write(f"**Identificação:** `{st.session_state['user_reg']}`")
    st.write(f"**Perfil:** `{st.session_state['user_role']}`")
    st.caption(f"**Sessão:** `{st.session_state['session_token'][:16]}...`")
    if st.button("🚪 Encerrar Sessão (Logout)", use_container_width=True):
        logout()

st.title("🧬 SGP-PINN ENTERPRISE | Laboratório de Biomatemática")
st.markdown("**Plataforma Experimental de Física Informada, Inferência Amortizada e Descoberta Simbólica em Fisiologia.**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Maturidade Técnica", "TRL 4 -> 5 (Protótipo)", "In Silico / P&D")
col2.metric("Motor Algorítmico", "PyTorch / DeepONet / SR3", "Física Informada")
col3.metric("Rastreabilidade", "Trilha Criptográfica", "Padrão SHA-256")
col4.metric("Arquitetura Regulatória", "Alinhamento SaMD", "Design p/ RDC 657")

st.markdown("---")
st.subheader("Motores e Laboratórios Numéricos Disponíveis")

col_a, col_b = st.columns(2)
with col_a:
    st.info("🔬 **1. U-PINN Residual:** Inferência com amostragem adaptativa (RAD) e separação de termos neurais.")
    st.info("📜 **3. Descoberta Simbólica (SINDy):** Extração de equações via SR3 com restrições biológicas.")
    st.info("📊 **5. Benchmarking Algorítmico:** Avaliação de complexidade assintótica e supressão de ruído in silico.")
    st.info("🎯 **7. Quantificação de Incerteza:** Neural SDE (Itô) e Functional Conformal Prediction (PCCP >= 95%).")
    st.info("🔬 **11. MIMIC-IV Benchmark Sepsis:** Avaliação em distribuições estatísticas de UTI (Surrogates).")

with col_b:
    st.success("👤 **2. PINN Inversa:** Calibração bayesiana amortizada de parâmetros fisiológicos individuais.")
    st.success("🌐 **4. Neural Jump-ODE:** Rastreamento de dinâmicas descontínuas com intervenções em bolus.")
    st.success("🏥 **6. Gateway FHIR R4:** Ingestão de mensagens de prontuário, titulação What-If e Sobol GSA.")
    st.success("⚙️ **10. Gestão & Licenciamento:** Controle granular de acessos e usuários por motor.")
    st.success("📋 **12. Conformidade & Editais:** Dossiê Técnico Preparatório (ANVISA RDC 657) e Propostas PIPE/FINEP.")
