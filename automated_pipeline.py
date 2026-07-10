import sys
import os
from datetime import datetime

sys.path.append('src')

from data_prep import prepare_full_dataset, get_feature_columns
from train_model import train_test_split_by_date, train_xgboost_model, evaluate_model
from report_generator import detect_anomalies, build_summary_text, generate_ai_report


def run_pipeline():
    print(f"[{datetime.now()}] Starting automated pipeline run...")

    # 1. Prepare data
    print("Loading and preparing data...")
    df = prepare_full_dataset(data_dir='data')
    feature_cols = get_feature_columns()

    # 2. Split and train
    print("Training model...")
    train_data, test_data = train_test_split_by_date(df)
    X_train = train_data[feature_cols]
    y_train = train_data['Sales']
    X_test = test_data[feature_cols]
    y_test = test_data['Sales']

    model = train_xgboost_model(X_train, y_train)
    y_pred, mae, rmse = evaluate_model(model, X_test, y_test)
    print(f"Model trained. MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    # 3. Baseline comparison
    baseline_pred = X_test['Sales_RollingMean_7']
    from sklearn.metrics import mean_absolute_error
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    # 4. Detect anomalies
    print("Detecting anomalies...")
    test_data_with_anomalies = detect_anomalies(test_data, y_pred)
    anomaly_count = test_data_with_anomalies['Is_Anomaly'].sum()
    print(f"Anomalies detected: {anomaly_count}")

    # 5. Generate AI report
    print("Generating AI report...")
    overall_summary, anomaly_text = build_summary_text(test_data_with_anomalies, baseline_mae, mae)
    report = generate_ai_report(overall_summary, anomaly_text, env_path='.env')

    # 6. Save report to a timestamped file
    os.makedirs('reports', exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    report_path = f'reports/report_{timestamp}.txt'

    with open(report_path, 'w') as f:
        f.write(f"Sales Forecast Report - Generated {datetime.now()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    print(f"Report saved to {report_path}")
    print(f"[{datetime.now()}] Pipeline run complete.\n")

    return report


if __name__ == '__main__':
    run_pipeline()