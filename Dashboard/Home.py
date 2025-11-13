import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(page_title="📊 Student Dashboard", layout="wide")

# --- Main Page Title ---
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📚 Student Habits vs Academic Performance Dashboard</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 18px; padding: 10px 20px; background-color: #f0f2f6; border-radius: 10px;'>
    🎉 Welcome to the interactive dashboard! Explore how students' lifestyle habits influence academic performance. Use the navigation menu to dive into the analyses and visual insights.
</div>
""", unsafe_allow_html=True)

# --- Layout with image and questions ---
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4727/4727424.png", width=200)
with col2:
    st.subheader("💡 Key Questions")
    st.markdown("""
    - Which factors impact grades?
    - Can we cluster students based on habits?
    """)

st.markdown("---")

# --- File Uploader ---
st.subheader("📁 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file with student data", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df  
        st.success("✅ Dataset uploaded and saved successfully!")

        st.markdown("### 🔍 Preview of Uploaded Data")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

else:
    st.info("ℹ️ Please upload a dataset to begin.")

