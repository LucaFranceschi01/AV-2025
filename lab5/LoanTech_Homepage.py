# ------------------------------------------------------------
# LoanTech Landing page
# Author: Luca Franceschi
# Email: luca.franceschi01@estudiant.upf.edu
# All contents are created in-house unless explicitly stated.

# UI Created with assistance from ChatGPT (GPT-5, OpenAI) (basically the entirety of this page)
# ------------------------------------------------------------
import streamlit as st

# Page setup
st.set_page_config(page_title="LoanTech | Smarter Loan Decisions", layout="wide")

# --- Custom CSS for styling ---
st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Global font + colors */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Hero section styling */
    .hero {
        text-align: center;
        padding: 5rem 1rem 3rem 1rem;
        background: linear-gradient(180deg, #2C7BE5 0%, #175CDA 100%);
        color: white;
        border-radius: 1rem;
        box-shadow: 0 4px 25px rgba(0,0,0,0.1);
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .hero p {
        font-size: 1.25rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }

    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    .footer {
        text-align: center;
        color: #888;
        margin-top: 4rem;
        font-size: 0.9rem;
    }

    .cta-button {
        background-color: white;
        color: #2C7BE5;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        text-decoration: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .cta-button:hover {
        background-color: #f3f6ff;
        text-decoration: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <h1>LoanTech</h1>
    <p>Empowering smarter, faster, and fairer loan decisions through intelligent prediction technology.</p>
    <a href="/Page_2" class="cta-button">🔮 Try the Loan Predictor</a>
</div>
""", unsafe_allow_html=True)

# --- About / Features Section ---
st.markdown("### Why Choose LoanTech?")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>💡 AI-Powered Insights</h3>
        <p>Leverage machine learning to assess creditworthiness with transparency and precision.</p>
        <a href="/Page_1" class="cta-button">Look at our data visualizations!</a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>⚙️ Scalable & Reliable</h3>
        <p>Built with modern data infrastructure to handle thousands of applications securely.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>🔐 Fair & Compliant</h3>
        <p>Aligned with responsible AI and lending regulations to ensure fairness and privacy.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    © 2025 LoanTech · UI Designed by ChatGPT
    <br>All rights reserved.
</div>
""", unsafe_allow_html=True)