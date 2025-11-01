import streamlit as st

st.set_page_config(layout="wide")
st.title("📊 Compare Models")
st.markdown("Compare the forecasting performance of two models.")

# --- Initialize session state ---
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []

models = ["SARIMAX", "Prophet", "Exponential Smoothing"]

# --- Add CSS styling ---
st.markdown("""
    <style>
    div[data-testid="stButton"] > button {
        width: 100%;
        padding: 14px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-3px);
        box-shadow: 0px 6px 14px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("### Select two models to compare:")

cols = st.columns(len(models))

for i, model in enumerate(models):
    with cols[i]:
        selected = model in st.session_state.selected_models
        
        # Apply dynamic styling based on selection
        button_style = f"""
            <style>
            div[data-testid="stButton"]:has(button[kind="primary"]):nth-of-type({i+1}) > button {{
                background-color: #28a745 !important;
                color: white !important;
                border: none !important;
            }}
            </style>
        """
        
        if selected:
            st.markdown(button_style, unsafe_allow_html=True)
        
        # Use different button types based on selection
        button_type = "primary" if selected else "secondary"
        
        if st.button(model, key=model, use_container_width=True, type=button_type):
            if selected:
                st.session_state.selected_models.remove(model)
            elif len(st.session_state.selected_models) < 2:
                st.session_state.selected_models.append(model)
            else:
                st.warning("⚠️ You can select up to 2 models only.")
            st.rerun()

# --- Display results ---
st.markdown("---")

selected_models = st.session_state.selected_models

if len(selected_models) == 2:
    st.success(f"✅ Selected: {selected_models[0]} and {selected_models[1]}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{selected_models[0]}")
        st.write("Metrics and forecasted graph here.")
    
    with col2:
        st.subheader(f"{selected_models[1]}")
        st.write("Metrics and forecasted graph here.")
else:
    st.info("Please select exactly two models to compare.")