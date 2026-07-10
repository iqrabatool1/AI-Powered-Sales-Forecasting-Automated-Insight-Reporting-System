import streamlit as st
import sys
import os

sys.path.append('src')

from data_prep import prepare_full_dataset, get_feature_columns
from train_model import train_test_split_by_date
from report_generator import detect_anomalies, build_summary_text, generate_ai_report
import joblib

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

st.title("📊 AI-Powered Sales Forecasting Dashboard")
st.caption("Automated demand forecasting, anomaly detection, and AI-generated business insights")

# --- Load data and model (cached so it doesn't reload every interaction) ---
@st.cache_data
def load_pipeline_data():
    df = prepare_full_dataset(data_dir='data')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('src/xgboost_sales_model.pkl')
    return model

with st.spinner("Loading data and model..."):
    df = load_pipeline_data()
    model = load_model()

st.success(f"Loaded {len(df):,} rows of sales data and trained model successfully")

# --- Sidebar: store selector ---
st.sidebar.header("🔍 Filter")
store_list = sorted(df['Store'].unique())
selected_store = st.sidebar.selectbox("Select a store", store_list)

store_df = df[df['Store'] == selected_store].sort_values('Date')

# --- Overview section ---
st.header(f"📈 Store {selected_store} — Sales Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Average Daily Sales", f"${store_df['Sales'].mean():,.0f}")
col2.metric("Max Daily Sales", f"${store_df['Sales'].max():,.0f}")
col3.metric("Total Days Recorded", f"{len(store_df):,}")

st.line_chart(store_df.set_index('Date')['Sales'])

# --- Forecast section ---
st.header("🔮 Sales Forecast")

feature_cols = get_feature_columns()

# Use the most recent row of this store as the basis for showing recent predictions
recent_data = store_df.tail(30).copy()
X_recent = recent_data[feature_cols]

recent_data['Predicted_Sales'] = model.predict(X_recent)

st.write("Actual vs Predicted Sales — last 30 recorded days")

chart_data = recent_data.set_index('Date')[['Sales', 'Predicted_Sales']]
st.line_chart(chart_data)

st.dataframe(
    recent_data[['Date', 'Sales', 'Predicted_Sales']].sort_values('Date', ascending=False),
    use_container_width=True
)

import shap
import matplotlib.pyplot as plt

# --- SHAP Explainability ---
st.header("🧠 Why These Predictions? (Model Explainability)")

with st.spinner("Calculating feature importance..."):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_recent)

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_recent, show=False)
st.pyplot(fig)

st.caption("This chart shows which factors most influenced the sales predictions for this store's recent history. Red = high value of that feature, blue = low value.")


# --- AI-Generated Business Report ---
st.header("🤖 AI-Generated Insight Report")

if st.button("Generate Report"):
    with st.spinner("Analyzing patterns and writing report..."):
        # Build test set for anomaly detection across ALL stores (more meaningful than single store)
        from train_model import train_test_split_by_date
        _, test_data_full = train_test_split_by_date(df)
        X_test_full = test_data_full[feature_cols]
        y_test_full = test_data_full['Sales']
        y_pred_full = model.predict(X_test_full)

        baseline_pred = X_test_full['Sales_RollingMean_7']
        from sklearn.metrics import mean_absolute_error
        baseline_mae = mean_absolute_error(y_test_full, baseline_pred)
        model_mae = mean_absolute_error(y_test_full, y_pred_full)

        test_with_anomalies = detect_anomalies(test_data_full, y_pred_full)
        overall_summary, anomaly_text = build_summary_text(test_with_anomalies, baseline_mae, model_mae)
        report = generate_ai_report(overall_summary, anomaly_text, env_path='.env')

    st.markdown(report)