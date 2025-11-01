import streamlit as st
import base64

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Airline Forecast Dashboard",
    page_icon="✈️",
    layout="wide"
)

# --- STYLES ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #f8f9fa, #e9ecef, #dee2e6);
            color: #212529;
            font-family: 'Poppins', sans-serif;
        }
        h1, h2, h3, h4 {
            color: #0d1b2a !important;
            font-weight: 600;
            text-align: center;
        }
        .airline-card {
            background: rgba(255, 255, 255, 0.97);
            border-radius: 18px;
            padding: 2rem 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0px 5px 15px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }
        .airline-card:hover {
            transform: translateY(-6px);
            box-shadow: 0px 10px 25px rgba(0,0,0,0.25);
        }
        .airline-logo {
            width: 150px;
            height: 150px;
            object-fit: contain;
            margin-bottom: 15px;
            border-radius: 14px;
            background-color: white;
            padding: 10px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
        }
        div[data-testid="stButton"] > button {
            background: linear-gradient(145deg, #212529, #343a40);
            color: #f8f9fa;
            border-radius: 14px;
            font-size: 17px;
            font-weight: 600;
            padding: 0.8em 2em;
            transition: all 0.3s ease-in-out;
            box-shadow: 0px 6px 14px rgba(0,0,0,0.25);
            border: none;
            text-align: center;
            margin-top: 15px;
        }
        div[data-testid="stButton"] > button:hover {
            background: linear-gradient(145deg, #000000, #343a40);
            color: #00b4d8;
            transform: scale(1.08);
            box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
        }
        [data-testid="column"] {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- FRONT PAGE ---
st.title("✈️ Airline Forecasting and Sensitivity Dashboard")
st.markdown("### Choose an airline to explore predictions and insights.")
st.markdown("<hr>", unsafe_allow_html=True)

# --- AIRLINE DATA ---
airlines = {
    "American Airlines": ("american.jpeg", "american"),
    "Delta Airlines": ("delta.png", "delta"),
    "United Airlines": ("united.jpeg", "united"),
    "Southwest Airlines": ("southwest.png", "southwest")
}

# --- GRID LAYOUT ---
cols = st.columns(4)

for i, (name, (img_path, page)) in enumerate(airlines.items()):
    with cols[i]:
        try:
            with open(img_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode()
            st.markdown(f"""
            <div class="airline-card">
                <img src="data:image/png;base64,{encoded}" class="airline-logo">
                <h4>{name}</h4>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.error(f"⚠️ Could not load {img_path}")

        # --- SELECT AIRLINE BUTTON ---
        if st.button(f"Select {name}", key=name):
            st.session_state["selected_airline"] = name
            st.success(f"{name} selected!")

# --- NEXT STEP BUTTONS ---
st.markdown("<hr>", unsafe_allow_html=True)

if "selected_airline" in st.session_state:
    st.subheader(f"Selected Airline: {st.session_state['selected_airline']}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Use one model"):
            st.switch_page("pages/one_model.py")
    with col2:
        if st.button("Compare models"):
            st.switch_page("pages/compare_models.py")
else:
    st.info("👆 Select an airline first to continue.")
