import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq


def detect_anomalies(test_data, y_pred, z_threshold=2.5):
    """Compare predicted vs actual sales and flag anomalies."""
    test_data = test_data.copy()
    test_data['Predicted_Sales'] = y_pred
    test_data['Error'] = test_data['Sales'] - test_data['Predicted_Sales']
    test_data['Abs_Error'] = test_data['Error'].abs()

    mean_error = test_data['Error'].mean()
    std_error = test_data['Error'].std()
    test_data['Error_Zscore'] = (test_data['Error'] - mean_error) / std_error
    test_data['Is_Anomaly'] = test_data['Error_Zscore'].abs() > z_threshold

    return test_data


def build_summary_text(test_data, baseline_mae, mae, top_n=5):
    """Build the text block fed into the LLM prompt."""
    overall_summary = f"""
Total predictions evaluated: {len(test_data)}
Baseline MAE: {baseline_mae:.2f}
XGBoost MAE: {mae:.2f}
Improvement over baseline: {round((1 - mae/baseline_mae) * 100, 1)}%

Anomalies detected: {test_data['Is_Anomaly'].sum()} ({round(test_data['Is_Anomaly'].mean()*100, 2)}% of test data)
"""
    anomalies = test_data[test_data['Is_Anomaly']].sort_values('Abs_Error', ascending=False)
    top_anomalies = anomalies[['Store', 'Date', 'Sales', 'Predicted_Sales', 'Error']].head(top_n)
    anomaly_text = top_anomalies.to_string(index=False)

    return overall_summary, anomaly_text


def generate_ai_report(overall_summary, anomaly_text, env_path='../.env'):
    """Send summary to Groq LLM and return the written report."""
    load_dotenv(dotenv_path=env_path, override=True)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
You are a data analyst writing a weekly sales performance summary for store managers.

Here is the model performance and anomaly data:

{overall_summary}

Top anomalies (biggest gaps between predicted and actual sales):
{anomaly_text}

Write a short, clear business report (150-200 words) that:
1. Summarizes how well the forecasting model performed
2. Highlights which stores had unusual sales behavior and whether they sold more or less than expected
3. Suggests 1-2 possible business reasons worth investigating
4. Uses plain English, no technical jargon (avoid words like MAE, RMSE, z-score)
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content