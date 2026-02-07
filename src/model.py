from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding, Reshape, Concatenate
from tensorflow.keras.models import Model

def build_lstm_model(input_shape, units, num_stores=1116):
    """
    Multi-input model: LSTM for time series trends and Entity Embeddings for store identity.
    """
    # Time series branch
    ts_input = Input(shape=input_shape, name='ts_input')
    x_ts = LSTM(units=units, return_sequences=False)(ts_input)
    x_ts = Dropout(0.25)(x_ts)

    # Store ID branch (Embedding)
    store_input = Input(shape=(1,), name='store_input')
    x_store = Embedding(input_dim=num_stores, output_dim=10)(store_input)
    x_store = Reshape(target_shape=(10,))(x_store)

    # Merge branches
    merged = Concatenate()([x_ts, x_store])

    # Prediction head
    x = Dense(32, activation='relu')(merged)
    output = Dense(1, name='output')(x)

    model = Model(inputs=[ts_input, store_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model