
🎯**Project Overview**

This project explores how students’ daily habits — including study hours, sleep patterns, social media use, diet, and mental health — influence their academic performance.
The dataset consists of 1,000 synthetic student records and was used to identify key behavioral factors correlated with high or low exam scores.

An interactive Streamlit dashboard was built to visualize insights, allowing users to explore trends, relationships, and clusters dynamically.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🧰 **Tools & Technologies**

- Programming:	Python
- Libraries:	pandas, numpy, matplotlib, seaborn, scikit-learn, plotly
- Dashboarding:	Streamlit
- Analysis:	KMeans Clustering, PCA, Correlation Heatmaps
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🧠 **Objectives**

- Identify lifestyle patterns influencing student performance

- Explore correlations between study habits and exam results

- Segment students using unsupervised learning (KMeans)

- Build an interactive dashboard for real-time data exploration
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🔍 **Data Analysis Workflow**

**1.Data Preparation**

- Cleaned and standardized all numerical and categorical variables

- Normalized numeric values for clustering

- Removed outliers to ensure balanced distributions

**2.Univariate Analysis**

Examined distributions of key variables (study hours, sleep duration, social media use)

Visualized mean exam scores by category

**3.Bivariate Analysis**

- Correlation heatmap between behavioral and performance indicators

- Scatter plots showing relationships such as:

    - Study Hours ⟷ Exam Scores

    - Sleep Hours ⟷ Mental Health Index

**4.Clustering (KMeans + PCA)**

- Grouped students into behavioral clusters

- Visualized clusters on a 2D PCA plot

**5.Dashboard Development**

- Built multi-page Streamlit app featuring:

- EDA dashboard (interactive histograms, heatmaps)

- Clustering dashboard (cluster explorer)

- KPI summary page
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 ## 📊 Feature Importance Insights

The feature importance chart reveals the following insights:

### Top Contributing Features
- **study_hours_per_day** – by far the most influential factor, with a normalized importance score of **0.708**.  
  This suggests that the amount of time a student dedicates to studying each day is the strongest predictor of academic performance.
- **mental_health_rating** – importance score of **0.106**.  
  A student’s mental well-being plays a key role in concentration, stress management, and exam readiness.  
  *This highlights the importance of psychological support programs in schools.*
- **Moderate importance features:**  
  - `social_media_hours`  
  - `sleep_hours`  
  - `netflix_hours`  
  These features likely influence exam performance indirectly by affecting focus, restfulness, and study time availability.

### Lower Importance Features
- Features with low predictive power include:  
  - `diet_quality`, `internet_quality` – minimal influence on exam scores, possibly due to weak relationships or dataset limitations.  
  - `part_time_job`, `extracurricular_participation` – each scoring **0.002**, suggesting these commitments do not significantly interfere with exam scores.  
    *This could be due to weak or non-linear relationships not captured by the model, or limited sample size for some categories.*
