# 🏏 IPL Match & Player Performance Analysis

![Dissertation Cover](https://i.imgur.com/7p4hJgN.png)

This repository contains the full implementation for the MSc Data Analytics dissertation project: **"Predictive Modelling and Interactive Dashboard for Cricket Match and Player Performance Analysis in the Indian Premier League (IPL)"**.

**Author:** Gowtham Kumar Mani Periyasamy  
**Institution:** University of Strathclyde (2025)  
**Supervisor:** Dr. Lindsey Corson

---

## 🎯 Project Overview

This project provides a comprehensive data-driven analysis of the Indian Premier League (IPL) from 2018 to 2025. It delivers two distinct, high-value outputs:

1.  **A High-Performance Predictive Model:** An ensemble machine learning model trained to forecast IPL match outcomes with high accuracy.
2.  **An Interactive Analytics Dashboard:** A user-friendly web application built with Streamlit for dynamic exploration of team, player, and venue statistics.

---

## ✨ Key Features

* **End-to-End Data Pipeline:** From raw JSON data parsing and cleaning to advanced feature engineering.
* **Advanced Predictive Modelling:** Implements and compares multiple classifiers, culminating in a blended ensemble of CatBoost and a Stacking model for optimal performance.
* **High Accuracy:** The final model achieved **~95% accuracy** on a real-world validation set of 215 unseen matches from the 2023–2025 IPL seasons.
* **Explainable AI (XAI):** Uses SHAP (SHapley Additive exPlanations) to interpret model predictions, identifying key drivers of match outcomes like toss decisions, venue advantage, and team form.
* **Interactive Dashboard:** A standalone Streamlit application for deep-dive analysis, featuring dynamic filters, auto-generated insights, and data export capabilities.

---

## 📊 Interactive Dashboard Preview

The Streamlit dashboard allows users to explore historical data without writing any code.



---

## 🛠️ Tech Stack

* **Data Analysis & Modelling:** Python, Pandas, NumPy, Scikit-learn
* **ML Frameworks:** XGBoost, LightGBM, CatBoost, Optuna (for hyperparameter tuning)
* **Interpretability:** SHAP
* **Dashboard:** Streamlit
* **Visualisation:** Matplotlib, Plotly, Seaborn
* **Data Source:** [Cricsheet.org](https://cricsheet.org/)

---

## 📂 Repository Structure

```
├── 📂 data/
│   └── ipl_json.zip          # Raw ball-by-ball data from Cricsheet (2018-2025)
├──  notebooks/
│   └── CMAPA.ipynb           # Jupyter Notebook with the full analysis pipeline
├── 📄 ipl_dashboard.py      # The Streamlit dashboard application script
├── 📄 requirements.txt      # Required Python packages for the project
└── 📄 README.md             # You are here!
```

---

## ⚙️ Installation & Setup

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Unzip the data:**
    * The analysis notebook assumes the `ipl_json.zip` file located in the `data/` directory is extracted.

---

## 🚀 How to Use

This project has two main components you can run:

### 1. The Jupyter Notebook

The full end-to-end analysis, from data cleaning and EDA to feature engineering and model training, is documented in the Jupyter Notebook.

* Navigate to the `notebooks/` directory and run `CMAPA.ipynb` using Jupyter Notebook or JupyterLab.

### 2. The Streamlit Dashboard

To launch the interactive web application:

1.  Make sure you are in the root directory of the project.
2.  Run the following command in your terminal:
    ```bash
    streamlit run ipl_dashboard.py
    ```
3.  Your web browser will open with the dashboard running locally.
