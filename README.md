🏏 Predictive Modelling & Dashboard for IPL Cricket (2018–2025)

This repository contains the implementation for my MSc Data Analytics dissertation project:
“Predictive Modelling and Interactive Dashboard for Cricket Match and Player Performance Analysis in the Indian Premier League (IPL)” by Gowtham Kumar Mani Periyasamy (University of Strathclyde, 2025).

🎯 Project Aim

To analyse IPL ball-by-ball data (2018–2025) and develop:

Descriptive insights — Player, team, and venue-level performance trends.

Predictive models — Forecasting match outcomes (batting first win vs chasing win).

Interactive dashboard — A Streamlit web app for dynamic filtering, visualisation, and exporting insights.

📂 Repository Contents

ipl_dashboard.py → Streamlit dashboard app (team insights, players, venues, matches).

CMAPA.html → Jupyter Notebook export with full exploratory analysis, feature engineering, and model training.

report_MSCDA.docx → Full dissertation write-up (MSc Data Analytics, University of Strathclyde).

data/ (not included in repo, to be added locally) → Match & delivery datasets from Cricsheet.org.

🛠️ Tech Stack

Python: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, shap, optuna

Visualisation: matplotlib, plotly

Dashboard: Streamlit

Data Source: Cricsheet.org

🔍 Key Features

End-to-end pipeline: data cleaning → feature engineering → ML models → interpretability → dashboard.

Predictive modelling using CatBoost, XGBoost, LightGBM, Stacking & Blended Ensembles.

83% test accuracy, ROC-AUC ≈ 0.91, with ~95% real-world validation accuracy on unseen IPL matches.

SHAP analysis for interpretable AI — toss decision, venue, and powerplay performance emerged as top predictors.

Streamlit dashboard with tabs for: Team Insights, Overview, Batting, Bowling, Matches, Players, Venues, Team-Venue performance.

📊 Example Insights

Teams chasing generally outperform, confirming the dew-factor advantage.

Venue conditions (e.g., Bengaluru, Wankhede) heavily impact scoring rates.

Ensemble models outperform baselines, offering reliable win forecasts.

🚀 Future Work

Real-time analytics with streaming pipelines (Kafka/Spark).

Fusion of multimodal data (video, player fitness, social media).

Cloud deployment for franchise or fantasy-cricket use cases.
