# 📊 AI-Powered Sales Forecasting & Automated Insight Reporting System

An end-to-end AI automation project that forecasts store-level sales demand, explains *why* the model predicts what it does, detects unusual sales patterns, and automatically generates plain-English business reports using a large language model — all wrapped in an interactive Streamlit dashboard.

This project combines **predictive machine learning**, **data analytics**, and **generative AI automation** into a single working pipeline, similar in spirit to modern BI tools (e.g., Tableau Pulse) that pair forecasting with AI-generated narrative insights.

---

## 🎯 Problem Statement

Retail businesses need to forecast demand accurately to manage inventory, staffing, and promotions. Raw forecasts alone aren't enough — stakeholders need to know:
1. **What** will sales look like next?
2. **Why** does the model predict that?
3. **What's unusual** that needs attention?
4. **What should we do about it** — explained in plain language, not statistics?

This project answers all four questions through one automated pipeline.

---

## 🏗️ Architecture

```
Raw Sales Data (Rossmann dataset)
        ↓
Data Cleaning & Feature Engineering
        ↓
XGBoost Forecasting Model
        ↓
SHAP Explainability  ──────┐
        ↓                  │
Anomaly Detection           │
        ↓                  │
AI Report Generation (Groq LLM) 
        ↓
Streamlit Dashboard / Automated Report Files
```

---

## 🧠 Key Features

- **Time-aware forecasting** — XGBoost regressor trained with proper time-based train/test splitting (never random split, to avoid data leakage common in time series tasks)
- **Feature engineering** — lag features (7/14-day), rolling averages (7/30-day), calendar features, holiday/promo flags, and competition context
- **Baseline benchmarking** — compares XGBoost against a naive rolling-average baseline to prove real predictive value
- **Explainability (SHAP)** — shows which factors drive each prediction, in a format understandable to non-technical stakeholders
- **Anomaly detection** — flags statistically unusual prediction errors (z-score method) to surface stores/days worth investigating
- **AI-generated business reports** — uses Groq's LLM API to turn raw numbers and anomalies into a short, plain-English summary for store managers
- **Automated pipeline script** — runs the entire process (train → predict → detect → report) independently and saves timestamped reports, simulating a production scheduling setup
- **Interactive dashboard** — Streamlit app for exploring forecasts, explainability, and reports per store

---

## 📈 Results

| Metric | Naive Baseline | XGBoost Model |
|---|---|---|
| MAE | 1,681.33 | 663.40 |
| RMSE | 2,437.52 | 940.39 |
| Improvement | — | **~60% reduction in error** |

**SHAP analysis** revealed that short-term promotional activity (`Promo`) is the strongest driver of daily sales, followed by medium-term historical trends (`Sales_Lag_14`, `Sales_RollingMean_30`). Long-term promotional programs (`Promo2`) and competitor proximity had comparatively minimal impact.

**Anomaly detection** flagged 2.23% of test predictions as statistically unusual, with a handful of high-volume stores (e.g., Store 909, 876, 292) recurring as outliers.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Data handling | pandas, numpy |
| Forecasting model | XGBoost |
| Explainability | SHAP |
| Anomaly detection | Statistical z-score thresholding |
| AI report generation | Groq API (Llama 3.3 70B) |
| Dashboard | Streamlit |
| Automation | Python `schedule` / standalone pipeline script |
| Environment management | python-dotenv, conda |

---

## 📁 Project Structure

```
sales-forecast-project/
├── data/
│   ├── train.csv
│   ├── store.csv
│   └── test.csv
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_prep.py
│   ├── train_model.py
│   ├── report_generator.py
│   └── xgboost_sales_model.pkl
├── reports/
├── app.py
├── automated_pipeline.py
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd sales-forecast-project
```

### 2. Create a virtual environment
```bash
conda create -n sales_forecast python=3.11
conda activate sales_forecast
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Add the dataset
Download the [Rossmann Store Sales dataset](https://www.kaggle.com/competitions/rossmann-store-sales/data) from Kaggle and place `train.csv` and `store.csv` inside the `data/` folder.

### 6. Run the dashboard
```bash
python -m streamlit run app.py
```

### 7. (Optional) Run the automated pipeline standalone
```bash
python automated_pipeline.py
```

---

## 🔍 Methodology Notes

- **Time-based validation**: Data was split chronologically rather than randomly, since random splitting would leak future information into training.
- **Handling missing data**: Missing values were filled logically based on *why* they were missing, not blindly imputed.
- **Closed-store filtering**: Days when a store was closed were excluded from training.
- **On-demand AI reporting**: Reports are generated on button-click to avoid unnecessary API calls.

---

## ⚠️ Limitations

- The AI-generated report occasionally has minor directional wording imprecision.
- Very high-volume outlier stores are harder for the model to predict accurately.
- The anomaly threshold (z-score > 2.5) could be tuned further based on business needs.

---

## 🚀 Future Improvements

- Add a "scenario simulator" for interactive what-if analysis
- Incorporate external factors (weather, local events) as features
- Deploy the automated pipeline on a real scheduler (cron, Airflow)
- Add store-tier segmentation for high-volume outlier stores

---

## 📄 Dataset Source

[Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) — Kaggle competition dataset containing historical sales data for 1,115 Rossmann drug stores.
