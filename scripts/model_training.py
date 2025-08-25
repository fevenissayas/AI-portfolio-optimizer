import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

def run_model_training(processed_data_path='data/processed/adj_close.csv',
                       model_save_path='models/lstm_tsla_forecast_model.keras',
                       forecast_output_path='data/processed/tsla_12month_forecast.csv',
                       target_ticker='TSLA',
                       train_end_date='2023-12-31',
                       test_start_date='2024-01-01',
                       sequence_length=60,
                       future_forecast_days=252):
    
    print(f"--- Starting Model Training for {target_ticker} ---")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(forecast_output_path), exist_ok=True)

    try:
        data = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
        tsla_data = data[target_ticker].to_frame()
        print(f"Data loaded successfully for {target_ticker} from {processed_data_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    train_data = tsla_data[:train_end_date]
    test_data = tsla_data[test_start_date:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    full_scaled_data = scaler.fit_transform(tsla_data)

    X_full, y_full = create_sequences(full_scaled_data, sequence_length)
    X_full = np.reshape(X_full, (X_full.shape[0], X_full.shape[1], 1))

    lstm_model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_full.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    print("Training LSTM model on full historical data...")
    history = lstm_model.fit(X_full, y_full,
                             epochs=100,
                             batch_size=32,
                             callbacks=[early_stopping],
                             verbose=0)

    print("\nLSTM Model Training Complete.")
    lstm_model.save(model_save_path)
    print(f"Trained LSTM model saved to {model_save_path}")

if __name__ == "__main__":
    run_model_training()