import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="EDA | Student Dashboard", layout="wide")
st.title("📋 Exploratory Data Analysis (EDA)")

# === Use uploaded dataset ===
if "df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset on the Home page first.")
    st.stop()

df = st.session_state["df"]

# === Dataset Information ===
st.markdown("### ℹ️ Dataset Info")
col1, col2 = st.columns(2)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])

with st.expander("🔍 View Data Types"):
    st.dataframe(df.dtypes.rename("Data Type"))

# === Missing Values ===
st.markdown("### ❓ Missing Values")
missing = df.isnull().sum()
missing = missing[missing > 0]

if not missing.empty:
    st.dataframe(missing.rename("Missing Count"))
else:
    st.success("✅ No missing values found!")

# === Duplicate Check ===
st.markdown("### 🔁 Duplicate Check")
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    st.warning(f"⚠️ Found **{duplicate_count}** duplicate rows.")
    with st.expander("👁️ View Duplicate Rows"):
        st.dataframe(df[df.duplicated()].head(10))
    if st.checkbox("🧹 Remove duplicate rows?"):
        df = df.drop_duplicates()
        st.session_state["df"] = df  # Update dataset in session state
        st.success(f"✅ Duplicates removed. Dataset now has {df.shape[0]} rows.")
else:
    st.success("✅ No duplicate rows found.")

# === Summary Statistics ===
st.markdown("### 📊 Summary Statistics")
st.dataframe(df.describe(), use_container_width=True)
