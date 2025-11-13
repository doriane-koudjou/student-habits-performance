import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Check if dataset is loaded
if "df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset on the Home page first.")
    st.stop()

df = st.session_state["df"]

# Title
st.set_page_config(page_title="Univariate Analysis", layout="wide")
st.title("📈 Univariate Analysis")

# Select column (automatically exclude student_id if it exists)
columns = df.columns.tolist()
if "student_id" in columns:
    columns.remove("student_id")

selected_column = st.selectbox("🔍 Select a variable", columns)

# ============ Numeric Feature ============
if pd.api.types.is_numeric_dtype(df[selected_column]):


    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📊 Histogram of {selected_column}")
        hist = px.histogram(
            df,
            x=selected_column,
            nbins=20,
            title=None,
            histfunc="count",
            color_discrete_sequence=["#4B8BBE"]
        )
        st.plotly_chart(hist, use_container_width=True)

    with col2:
        st.subheader(f"📦 Box Plot of {selected_column}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[selected_column], ax=ax, color="#4B8BBE")
        ax.set_title("")
        st.pyplot(fig)

    st.subheader("🧮 Summary Statistics")
    st.dataframe(df[selected_column].describe().to_frame(), use_container_width=True)

# ============ Categorical Feature ============
else:
    
    value_counts = df[selected_column].value_counts()
    labels = value_counts.index.tolist()
    values = value_counts.values.tolist()

    color_palette = ["#4B8BBE", "#00b894", "#fdcb6e", "#e17055", "#6c5ce7"]
    color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(labels)}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📊 Bar Chart of {selected_column}")
        bar = px.bar(
            x=labels,
            y=values,
            labels={"x": selected_column, "y": "Count"},
            title=None,
            color=labels,
            color_discrete_map=color_map
        )
        bar.update_layout(showlegend=False)  
        st.plotly_chart(bar, use_container_width=True)

    with col2:
        st.subheader(f"🥧 Pie Chart of {selected_column}")
        pie = px.pie(
            names=labels,
            values=values,
            title=None,
            color=labels,
            color_discrete_map=color_map,
            hole=0
        )
        pie.update_traces(
            textinfo='value+percent',
            hovertemplate='%{label}<br>Count: %{value}<br>Percent: %{percent}',
            textposition='inside'
        )
        pie.update_layout(showlegend=True)
        st.plotly_chart(pie, use_container_width=True)

    st.subheader("📋 Value Counts")
    st.dataframe(value_counts.to_frame(name="Count"), use_container_width=True)
