"""
This script performs a global evaluation of the trained LSTM model.
It calculates MAE, MAPE, and RMSPE on unseen data from 2015-06-01 onwards.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

def get_metrics(y_true, y_pred):
    mask = y_true != 0
    rmspe = np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mape, rmspe

def evaluate_global(model, df, scaler):
    df = df.copy()
    df['log_sales'] = np.log1p(df['sales'])

    df['store_type'] = LabelEncoder().fit_transform(df['store_type'].astype(str))
    df['assortment'] = LabelEncoder().fit_transform(df['assortment'].astype(str))
    df['state_holiday'] = LabelEncoder().fit_transform(df['state_holiday'].astype(str))

    cols = [
        'log_sales', 'promo', 'promo2', 'school_holiday', 'state_holiday',
        'day_of_week', 'month', 'rolling_avg_7', 'competition_distance',
        'store_type', 'assortment'
    ]

    test_date = '2015-06-01'

    # Scaling using the passed scaler object
    scaled_values = scaler.transform(df[cols])

    X_ts, X_store, y_true = [], [], []
    window = 7

    print(f"Building sequences for test set (post {test_date})...")
    for store_id in df['store_id'].unique():
        store_mask = df['store_id'] == store_id
        store_df = df[store_mask].sort_values('date')

        store_indices = np.where(store_mask.values)[0]
        store_vals = scaled_values[store_indices]
        store_dates = store_df['date'].values
        real_sales = store_df['sales'].values

        for i in range(window, len(store_vals)):
            if store_dates[i] >= np.datetime64(test_date):
                X_ts.append(store_vals[i-window:i, :])
                X_store.append(store_id)
                y_true.append(real_sales[i])

    X_ts, X_store = np.array(X_ts), np.array(X_store)
    y_true = np.array(y_true)

    if len(y_true) == 0:
        print("No test data available.")
        return 0, 0, 0

    print(f"Running inference on {len(y_true)} samples...")
    preds_scaled = model.predict([X_ts, X_store], batch_size=1024, verbose=1)

    dummy = np.zeros((len(preds_scaled), len(cols)))
    dummy[:, 0] = preds_scaled.flatten()
    inv_preds = np.expm1(scaler.inverse_transform(dummy)[:, 0])

    mae, mape, rmspe = get_metrics(y_true, inv_preds)

    return mae, mape, rmspe