import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Feature Importance | Student Dashboard", layout="wide")
st.title("📌 Feature Importance for Exam Score Prediction")

# Load dataset
@st.cache_data
def load_data():
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset on the Home page first.")
        st.stop()
    return st.session_state["df"].copy()

df = load_data()

# Preprocessing (hidden from user)
df = df.dropna(subset=['exam_score'])
X = df.drop(columns=['exam_score'])

# Remove student_id if present
if 'student_id' in X.columns:
    X = X.drop(columns=['student_id'])

y = df['exam_score']

# Ordinal encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    encoder = OrdinalEncoder()
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- Model Evaluation ----
st.markdown("### 📊 Model Performance")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("R² Score", f"{r2:.2f}")

# ---- Feature Importance ----
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

# Assign distinct colors using Plotly palette
colors = px.colors.qualitative.Plotly
feat_imp_df['Color'] = [colors[i % len(colors)] for i in range(len(feat_imp_df))]

# ---- Plotly Bar Chart with Custom Colors ----
st.markdown("### 🔍 Top Feature Importances")
fig = go.Figure()
fig.add_trace(go.Bar(
    x=feat_imp_df['Importance'],
    y=feat_imp_df['Feature'],
    orientation='h',
    marker=dict(color=feat_imp_df['Color']),
    text=feat_imp_df['Importance'].round(3),
    textposition='auto'
))
fig.update_layout(
    title="Feature Importance for Predicting Exam Score",
    xaxis_title="Importance",
    yaxis_title="Feature",
    height=600,
    margin=dict(l=100, r=20, t=50, b=50)
)
st.plotly_chart(fig, use_container_width=True)

# ---- Preview Encoded Data ----
st.markdown("### 🧾 Sample of Encoded Dataset Used for Prediction")
st.dataframe(X.head())
