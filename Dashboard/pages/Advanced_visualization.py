import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

# ---- Load & Preprocess Dataset ----
@st.cache_data

def load_data():
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset on the Home page first.")
        st.stop()
    df = st.session_state["df"].copy()

    mappings = {
        'diet_quality': {'Poor': 1, 'Average': 2, 'Good': 3},
        'exercise_frequency': {'Rarely': 1, 'Sometimes': 2, 'Often': 3},
        'internet_quality': {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

# ---- Streamlit Setup ----
st.set_page_config(page_title="Student Clustering Dashboard", layout="wide")
st.title("🎓 Clustering Student Habits & Exam Performance")

# ---- Load Data ----
df = load_data()

# ---- Sidebar ----
st.sidebar.header("Clustering Controls")

excluded_features = ['exercise_frequency']
all_features = [col for col in df.drop(columns=["exam_score"]).columns if col not in excluded_features]
user_selected_features = st.sidebar.multiselect(
    "Select features for clustering:",
    all_features,
    default=all_features[:5]
)

k = st.sidebar.slider("Number of Clusters (k):", 2, 6, value=3)

# Filter sliders
numeric_cols = df[user_selected_features].select_dtypes(include=np.number).columns.tolist()
st.sidebar.markdown("### Optional: Filter Data")
filters = {}
for col in numeric_cols:
    min_val, max_val = float(df[col].min()), float(df[col].max())
    selected_range = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val))
    filters[col] = selected_range

# ---- Clustering ----
if len(user_selected_features) < 1:
    st.warning("Please select at least one feature for clustering.")
    st.stop()

# Apply filtering
df_cluster = df[user_selected_features + ['exam_score']].dropna()
for col, (min_v, max_v) in filters.items():
    if col in df_cluster.columns:
        df_cluster = df_cluster[(df_cluster[col] >= min_v) & (df_cluster[col] <= max_v)]

# Improved check for usable features
selected_features = []
skipped_features = []
for col in user_selected_features:
    if col in df_cluster.columns:
        non_null_series = df_cluster[col].dropna()
        try:
            non_null_series = pd.to_numeric(non_null_series)
            if non_null_series.nunique() < 2 or non_null_series.std() == 0:
                skipped_features.append(col)
            else:
                selected_features.append(col)
        except ValueError:
            skipped_features.append(col)

if len(selected_features) < 1:
    st.error("No usable features remaining for clustering after removing low-variance ones.")
    st.stop()

if df_cluster.shape[0] < k:
    st.error(
        f"⚠️ Only {df_cluster.shape[0]} students left after filtering — cannot form {k} clusters. "
        "Try reducing the number of clusters (k) or broadening your filters."
    )
    st.stop()

# Clustering logic
X = df_cluster[selected_features].copy()
X = pd.get_dummies(X, drop_first=False)

if X.shape[1] == 0 or np.all(np.std(X, axis=0) == 0):
    st.error("Clustering requires at least one feature with variation across rows.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
df_cluster['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)


# ---- PCA and Box Plot Side-by-Side ----
if X_scaled.shape[1] >= 2:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_cluster['PCA1'] = pca_result[:, 0]
    df_cluster['PCA2'] = pca_result[:, 1]

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_pca, columns=['PCA1', 'PCA2'])
    centroids_df['Cluster'] = centroids_df.index.astype(str)

    st.subheader("📌 Cluster Visualization")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            df_cluster,
            x="PCA1",
            y="PCA2",
            color="Cluster",
            hover_data=['exam_score'] + selected_features,
            title="PCA Projection of Clusters",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        color_map = {trace.name: trace.marker.color for trace in fig.data if trace.name in df_cluster['Cluster'].unique()}
        for i, row in centroids_df.iterrows():
            fig.add_scatter(
                x=[row['PCA1']],
                y=[row['PCA2']],
                mode='markers',
                marker=dict(size=20, symbol='x', color=color_map.get(str(i), "black"), line=dict(width=2, color='black')),
                name=f"Centroid {i}",
                showlegend=True
            )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig_box, ax_box = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_cluster, x='Cluster', y='exam_score', palette='Set2')
        plt.title("Exam Scores by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Exam Score")
        st.pyplot(fig_box)

        # ---- Cluster Summary as Bar Chart ----
st.subheader("📊 Cluster Summary - Feature Averages")
numeric_selected = df_cluster[selected_features + ['exam_score']].select_dtypes(include=np.number).columns.tolist()
cluster_summary = df_cluster.groupby("Cluster")[numeric_selected].mean().round(2)
cluster_summary.index.name = "Cluster"
cluster_summary = cluster_summary.reset_index()

summary_melted = cluster_summary.melt(id_vars="Cluster", var_name="Feature", value_name="Average")
fig_summary = px.bar(
    summary_melted,
    x="Cluster",
    y="Average",
    color="Feature",
    barmode="group",
    text_auto=".2f",
    title="Average Feature Values per Cluster"
)
st.plotly_chart(fig_summary, use_container_width=True)


# ---- Data Preview ----
st.subheader("🔍 Clustered Data Preview")
st.dataframe(df_cluster.head(10))
