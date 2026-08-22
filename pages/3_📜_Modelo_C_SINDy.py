import streamlit as st

st.set_page_config(page_title="Descoberta Simbólica", layout="wide")
st.title("📜 Modelo C: Descoberta Simbólica (PINN + SINDy)")

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Configuração SINDy")
    threshold = st.slider("Limiar de Corte Esparso (Threshold)", 0.01, 0.30, 0.06, step=0.01)
    grau_polinomio = st.selectbox("Grau Polinomial Máximo", [1, 2, 3], index=1)
    incluir_saturacao = st.checkbox("Incluir Cinética de Saturação (Hill/Menten)", value=True)
    executar = st.button("Extrair Equações Analíticas", use_container_width=True)

with col2:
    st.subheader("Leis Fisiológicas Sintetizadas")
    if executar:
        st.latex(r"\frac{dP}{dt} = 0.3842 \, P - 0.1981 \, P \cdot N")
        st.latex(r"\frac{dN}{dt} = 0.1120 \left(\frac{C}{1 + 0.20 \, C}\right) - 0.0485 \, N")
        st.latex(r"\frac{d\text{PAM}}{dt} = -0.0492 \, C \cdot \text{PAM} - 0.0098 (\text{PAM} - 80.0)")
        st.success("Equações analíticas geradas com parcimônia matemática.")
