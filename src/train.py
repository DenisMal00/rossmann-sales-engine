"""
This script provides the training logic for the LSTM model.
Now it accepts a dataframe as input and returns the trained assets.
"""
import numpy as np
from alembic.command import history
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from model import build_lstm_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_global_model(df,batch_size, epochs, lstm_units):
    df = df.copy()
    # Target transformation
    df['log_sales'] = np.log1p(df['sales'])

    # Categorical encoding
    df['store_type'] = LabelEncoder().fit_transform(df['store_type'].astype(str))
    df['assortment'] = LabelEncoder().fit_transform(df['assortment'].astype(str))
    df['state_holiday'] = LabelEncoder().fit_transform(df['state_holiday'].astype(str))

    cols = [
        'log_sales', 'promo', 'promo2', 'school_holiday', 'state_holiday',
        'day_of_week', 'month', 'rolling_avg_7', 'competition_distance',
        'store_type', 'assortment'
    ]

    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])

    # Date split points
    val_date = '2015-04-01'
    test_date = '2015-06-01'

    X_ts_train, X_store_train, y_train = [], [], []
    X_ts_val, X_store_val, y_val = [], [], []
    # Test lists are kept for sequence building but evaluation is handled separately
    X_ts_test, X_store_test, y_test = [], [], []

    window = 7

    print("Building sequences with 3-way split...")
    for store_id in df['store_id'].unique():
        store_df = df[df['store_id'] == store_id].sort_values('date')
        values = store_df[cols].values
        ids = store_df['store_id'].values
        dates = store_df['date'].values

        for i in range(window, len(store_df)):
            curr_date = dates[i]
            seq = values[i - window:i, :]
            store_id_val = ids[i]
            target = values[i, 0]  # log_sales

            if curr_date >= np.datetime64(test_date):
                X_ts_test.append(seq)
                X_store_test.append(store_id_val)
                y_test.append(target)
            elif curr_date >= np.datetime64(val_date):
                X_ts_val.append(seq)
                X_store_val.append(store_id_val)
                y_val.append(target)
            else:
                X_ts_train.append(seq)
                X_store_train.append(store_id_val)
                y_train.append(target)

    X_ts_train, X_store_train, y_train = np.array(X_ts_train), np.array(X_store_train), np.array(y_train)
    X_ts_val, X_store_val, y_val = np.array(X_ts_val), np.array(X_store_val), np.array(y_val)

    print(f"Dataset Split: Train {len(y_train)} | Val {len(y_val)}")

    # Training
    model = build_lstm_model(input_shape=(window, len(cols)), units=lstm_units)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

    history=model.fit(
        [X_ts_train, X_store_train], y_train,
        validation_data=([X_ts_val, X_store_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True,
        callbacks=[early_stop, reduce_lr]
    )

    # Return assets for the pipeline to manage
    return model, scaler, history