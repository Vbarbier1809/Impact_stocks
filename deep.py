import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def load_and_prepare_data_deep(folder_path):
    """
    Load all CSV files in folder_path, add 'Company' column, and return a DataFrame sorted by Company and Date.
    Raises:
        FileNotFoundError: if no CSV files are found in the specified folder.
    """
    # Support any CSV filenames
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{folder_path}'. "
                                f"Check that the path is correct and files end in .csv.")

    df_list = []
    for file in csv_files:
        # Attempt to infer company name from filename (before first underscore)
        basename = os.path.basename(file)
        company_name = basename.split('_')[0]
        df_temp = pd.read_csv(file, parse_dates=["Date"]) if 'Date' in pd.read_csv(file, nrows=0).columns else pd.read_csv(file)
        df_temp["Company"] = company_name
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    if 'Date' in df.columns:
        df.sort_values(["Company", "Date"], inplace=True)
    else:
        df.sort_values(["Company"], inplace=True)

    if 'Close' not in df.columns:
        raise KeyError("Column 'Close' not found in loaded CSV files.")

    return df[['Date', 'Close', 'Company']] if 'Date' in df.columns else df[['Close', 'Company']]


def create_dataset(series: np.ndarray, window_size: int = 30) -> (np.ndarray, np.ndarray):
    # unchanged...
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y


def build_mlp_model(input_shape, hidden_dims=[64, 32], dropout_rate=0.2,
                    activation='relu', optimizer='adam', learning_rate=0.001):
    """
    Build a simple MLP for regression.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    for units in hidden_dims:
        x = tf.keras.layers.Dense(units, activation=activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, name='output')(x)
    model = tf.keras.Model(inputs, outputs, name='MLP')

    # Configure optimizer
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate)
    else:
        opt = optimizer

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model


def build_rnn_model(input_shape, hidden_dims=[50], dropout_rate=0.2,
                    activation='tanh', optimizer='adam', learning_rate=0.001):
    """
    Build a SimpleRNN model for regression.
    """
    model = tf.keras.Sequential(name='RNN')
    for i, units in enumerate(hidden_dims):
        return_seq = (i < len(hidden_dims)-1)
        model.add(tf.keras.layers.SimpleRNN(units,
                                            activation=activation,
                                            return_sequences=return_seq))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, name='output'))

    # Configure optimizer
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate)
    else:
        opt = optimizer

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model


def build_lstm_model(input_shape, hidden_dims=[50], dropout_rate=0.2,
                     activation='tanh', optimizer='adam', learning_rate=0.001):
    """
    Build an LSTM model for regression.
    """
    model = tf.keras.Sequential(name='LSTM')
    for i, units in enumerate(hidden_dims):
        return_seq = (i < len(hidden_dims)-1)
        model.add(tf.keras.layers.LSTM(units,
                                       activation=activation,
                                       return_sequences=return_seq))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, name='output'))

    # Configure optimizer
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate)
    else:
        opt = optimizer

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model


def train_model(model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                input_shape, hidden_dims, dropout_rate, activation,
                optimizer, learning_rate, epochs=50, batch_size=32,
                validation_data=None, verbose=1):
    """
    Build and train a model of type 'MLP', 'RNN', or 'LSTM'.
    Returns the trained model and training history.

    validation_data: tuple (X_val, y_val) or None
    """
    # Build model
    if model_type.upper() == 'MLP':
        model = build_mlp_model(input_shape, hidden_dims,
                                dropout_rate, activation,
                                optimizer, learning_rate)
    elif model_type.upper() == 'RNN':
        model = build_rnn_model(input_shape, hidden_dims,
                                dropout_rate, activation,
                                optimizer, learning_rate)
    elif model_type.upper() == 'LSTM':
        model = build_lstm_model(input_shape, hidden_dims,
                                  dropout_rate, activation,
                                  optimizer, learning_rate)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=verbose
    )
    return model, history


def predict_and_evaluate(model, X_test: np.ndarray, y_test: np.ndarray,
                         scaler: MinMaxScaler or StandardScaler = None,
                         print_results: bool = True):
    """
    Predict on X_test, inverse-scale predictions, compute MAE and RMSE.
    Prints first 10 true vs predicted if print_results.
    Returns dict with 'y_true', 'y_pred', 'mae', 'rmse'.
    """
    y_pred = model.predict(X_test).flatten()
    if scaler is not None:
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_test_inv, y_pred_inv = y_test, y_pred

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    if print_results:
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        print("First 10 predictions vs actual:")
        for i in range(min(10, len(y_test_inv))):
            print(f"Predicted: {y_pred_inv[i]:.4f}, Actual: {y_test_inv[i]:.4f}")

    return {
        'y_true': y_test_inv,
        'y_pred': y_pred_inv,
        'mae': mae,
        'rmse': rmse
    }


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = 'Predicted vs True'):
    """
    Plot true vs predicted values on the same figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def deep():
    df = load_and_prepare_data_deep("/Users/victorbarbier/Documents/Dauphine/Python for Data Science/Companies_historical_data")
    company = 'Pfizer'
    series = df[df['Company']==company]['Close'].values
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1,1)).flatten()
    X, y = create_dataset(series_scaled, window_size=30)
    split = int(len(X)*0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    model, history = train_model('LSTM', X_train, y_train, X_train.shape[1:],
                             hidden_dims=[50], dropout_rate=0.2,activation='tanh', optimizer='adam',
                              learning_rate=0.001, epochs=100,  batch_size=32,
                              validation_data=(X_test, y_test))
    results = predict_and_evaluate(model, X_test, y_test, scaler)
    plot_predictions(results['y_true'], results['y_pred'], title=f"{company} LSTM")

if __name__ == "__main__":
    deep()
