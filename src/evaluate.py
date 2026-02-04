"""
Global evaluation script.
Tests the model on the 'future' data (post 2015-06-01) for all stores.
"""
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from database import get_training_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

def get_metrics(y_true, y_pred):
    # Standard retail metrics
    mask = y_true != 0
    rmspe = np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mape, rmspe

def evaluate_global():
    print("Loading model and assets...")
    model = load_model(os.path.join(MODELS_DIR, 'sales_model.keras'), compile=False)
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

    # Load and clean data
    df = get_training_data()

    # Feature engineering
    df['log_sales'] = np.log1p(df['sales'])
    df['store_type'] = LabelEncoder().fit_transform(df['store_type'])
    df['assortment'] = LabelEncoder().fit_transform(df['assortment'])
    df['state_holiday'] = LabelEncoder().fit_transform(df['state_holiday'])
    cols = [
        'log_sales',
        'promo',
        'promo2',
        'school_holiday',
        'state_holiday',
        'day_of_week',
        'month',
        'rolling_avg_7',
        'competition_distance',
        'store_type',
        'assortment'
    ]
    test_date = '2015-06-01'
    test_df = df[df['date'] >= test_date].copy()

    if test_df.empty:
        print("No test data found!")
        return

    # Use the scaler from training
    scaled_test = scaler.transform(test_df[cols])

    X_ts, X_store, y_true = [], [], []
    window = 7

    print(f"Building sequences for test set...")
    for store_id in test_df['store_id'].unique():
        # Mask data for current store
        mask = test_df['store_id'] == store_id
        store_vals = scaled_test[mask]
        store_ids = test_df[mask]['store_id'].values
        real_sales = test_df[mask]['sales'].values

        if len(store_vals) < window:
            continue

        for i in range(window, len(store_vals)):
            X_ts.append(store_vals[i-window:i, :])
            X_store.append(store_ids[i])
            y_true.append(real_sales[i])

    X_ts, X_store = np.array(X_ts), np.array(X_store)
    y_true = np.array(y_true)

    print(f"Predicting on {len(y_true)} samples...")
    preds_scaled = model.predict([X_ts, X_store], batch_size=1024, verbose=1)

    # Invert scaling and log transformation
    dummy = np.zeros((len(preds_scaled), len(cols)))
    dummy[:, 0] = preds_scaled.flatten()
    inv_preds = np.expm1(scaler.inverse_transform(dummy)[:, 0])

    # Final report
    mae, mape, rmspe = get_metrics(y_true, inv_preds)

    print("\n" + "="*40)
    print("      FINAL TEST REPORT (UNSEEN DATA)")
    print("="*40)
    print(f"Period: {test_date} to 2015-07-31")
    print(f"MAE:   {mae:.2f}€")
    print(f"MAPE:  {mape*100:.2f}%")
    print(f"RMSPE: {rmspe:.4f}")
    print("="*40)

    # Plot results for the first store in the set
    sample_id = X_store[0]
    sample_mask = X_store == sample_id
    #plot_results(y_true[sample_mask], inv_preds[sample_mask], sample_id)

def plot_results(real, pred, store_id):
    plt.figure(figsize=(12, 6))
    plt.plot(real, label='Real Sales', color='#1f77b4')
    plt.plot(pred, label='AI Prediction', color='#ff7f0e', linestyle='--')
    plt.title(f'Store {store_id} - Final Test Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    evaluate_global()