import sys
sys.path.append('.')

import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_prep import prepare_full_dataset, get_feature_columns


def train_test_split_by_date(df, split_date='2015-06-01'):
    """Time-based split — never use random split for time series."""
    df = df.sort_values('Date')
    train_data = df[df['Date'] < split_date]
    test_data = df[df['Date'] >= split_date]
    return train_data, test_data


def train_xgboost_model(X_train, y_train):
    """Train the XGBoost regressor."""
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return predictions and error metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, mae, rmse


def run_full_training(data_dir='../data', model_out_path='xgboost_sales_model.pkl'):
    """Full pipeline: prep data -> split -> train -> evaluate -> save."""
    df = prepare_full_dataset(data_dir)
    feature_cols = get_feature_columns()

    train_data, test_data = train_test_split_by_date(df)

    X_train = train_data[feature_cols]
    y_train = train_data['Sales']
    X_test = test_data[feature_cols]
    y_test = test_data['Sales']

    model = train_xgboost_model(X_train, y_train)
    y_pred, mae, rmse = evaluate_model(model, X_test, y_test)

    joblib.dump(model, model_out_path)

    print(f"Model trained and saved to {model_out_path}")
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    return model, test_data, y_pred, mae, rmse


if __name__ == '__main__':
    run_full_training()