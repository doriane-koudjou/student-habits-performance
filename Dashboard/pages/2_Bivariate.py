import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load from session
if "df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset on the Home page first.")
    st.stop()

df = st.session_state["df"]

# Set page config
st.set_page_config(page_title="Bivariate Analysis", layout="wide")
st.markdown("<h3 style='color:#4B8BBE;'>🔗 Bivariate Analysis</h3>", unsafe_allow_html=True)

# Select features
columns = df.columns.tolist()
if "student_id" in columns:
    columns.remove("student_id")

col1, col2 = st.columns(2)
x_var = col1.selectbox("📌 Select X-axis (independent variable)", columns)
y_var = col2.selectbox("🎯 Select Y-axis (dependent variable)", columns, index=1 if len(columns) > 1 else 0)

x_dtype = pd.api.types.infer_dtype(df[x_var])
y_dtype = pd.api.types.infer_dtype(df[y_var])

# === Numeric vs Numeric ===
if pd.api.types.is_numeric_dtype(df[x_var]) and pd.api.types.is_numeric_dtype(df[y_var]):
    st.markdown(f"### 📈 Scatter Plot: {x_var} vs {y_var}")
    fig = px.scatter(df, x=x_var, y=y_var, trendline="ols",
                     color_discrete_sequence=["#4B8BBE"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Correlation")
    corr_val = df[[x_var, y_var]].corr().iloc[0, 1]
    st.info(f"🔹 Pearson correlation between **{x_var}** and **{y_var}** is **{corr_val:.2f}**.")

# === Numeric vs Categorical ===
elif (pd.api.types.is_numeric_dtype(df[x_var]) and pd.api.types.is_categorical_dtype(df[y_var]) or
      pd.api.types.is_numeric_dtype(df[y_var]) and pd.api.types.is_categorical_dtype(df[x_var]) or
      (df[x_var].dtype == "object" and pd.api.types.is_numeric_dtype(df[y_var])) or
      (df[y_var].dtype == "object" and pd.api.types.is_numeric_dtype(df[x_var]))):

    st.markdown(f"### 📦 Box/Violin Plot: {x_var} vs {y_var}")

    if pd.api.types.is_numeric_dtype(df[x_var]):
        fig = px.violin(df, y=y_var, x=x_var, box=True, points="all",
                        color_discrete_sequence=["#4B8BBE"])
    else:
        fig = px.violin(df, x=x_var, y=y_var, box=True, points="all",
                        color_discrete_sequence=["#4B8BBE"])
    st.plotly_chart(fig, use_container_width=True)

# === Categorical vs Categorical ===
elif df[x_var].dtype == "object" and df[y_var].dtype == "object":
    st.markdown(f"### 🔥 Category Heatmap: {x_var} vs {y_var}")
    ct = pd.crosstab(df[x_var], df[y_var])
    fig = px.imshow(ct, text_auto=True, color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠️ Unsupported combination of data types.")

    # === Correlation Matrix (Numeric Only) ===
st.subheader("📈 Correlation Matrix (numeric features only)")

numeric_df = df.select_dtypes(include='number')

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='Blues',         
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={'label': 'Correlation'}
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig)



