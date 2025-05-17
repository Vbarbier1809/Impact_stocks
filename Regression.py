import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def load_and_prepare_data(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    for file in csv_files:
        company_name = os.path.basename(file).replace("_historical_data.csv", "")
        df_temp = pd.read_csv(file)
        df_temp["Company"] = company_name
        df_list.append(df_temp)
    df = pd.concat(df_list, ignore_index=True)
    df_close = df[['Close', 'Company']]
    return df_close

def preprocess_and_create_datasets(df_close, n_days=30):
    company_datasets = {}
    for company in df_close['Company'].unique():
        df_company = df_close[df_close['Company'] == company]
        close_values = df_company[['Close']].values
        company_returns_close = close_values.flatten()
        split_idx = int(len(close_values) * 0.8)
        train_data = close_values[:split_idx]
        test_data = close_values[split_idx - n_days:]
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_data)
        scaled_test = scaler.transform(test_data)

        def create_target_features(data, n=n_days):
            x, y = [], []
            for i in range(n, len(data)):
                x.append(data[i - n:i, 0])
                y.append(data[i, 0])
            return np.array(x), np.array(y)

        X_train, y_train = create_target_features(scaled_train, n_days)
        X_test, y_test = create_target_features(scaled_test, n_days)

        company_datasets[company] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'company_returns_close': company_returns_close
        }
    return company_datasets

def train_and_evaluate_model(name, model, param_grid, X_train, y_train, X_test, y_test, scaler, company_returns_close, y_train_len):
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(company_returns_close)), company_returns_close, color='red', label='Valeurs réelles')
    start = y_train_len
    plt.plot(range(start + 30, start + len(y_pred_inv) + 30), y_pred_inv, color='blue', label=f'{name} Predictions')
    plt.title(f'{name} - Prédictions vs Réel')
    plt.legend()
    plt.show()

    return {
        'Modèle': name,
        'MSE': mse,
        'RMSE': rmse,
        'Best Params': grid.best_params_,
        'Model': best_model
    }

def evaluate_company(datasets, company_name):
    if company_name not in datasets:
        raise KeyError(f"Entreprise '{company_name}' non trouvée dans les données disponibles: {list(datasets.keys())}")

    data = datasets[company_name]
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    scaler = data['scaler']
    returns = data['company_returns_close']
    y_train_len = len(y_train)

    experiments = [
        ('LinearRegression', LinearRegression(), {}),
        ('RandomForest', RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [3, 5, 10]}),
        ('KNN', KNeighborsRegressor(), {'n_neighbors': [3, 5, 10]}),
        ('XGBoost', XGBRegressor(verbosity=0), {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]})
    ]

    results = []
    for name, model, grid in experiments:
        print(f"\n>>> Entraînement: {name}")
        res = train_and_evaluate_model(name, model, grid, X_train, y_train, X_test, y_test, scaler, returns, y_train_len)
        results.append(res)
    return results

def main():
    folder_path = "/Users/victorbarbier/Documents/Dauphine/Python for Data Science/Companies_historical_data"
    df_close = load_and_prepare_data(folder_path)
    datasets = preprocess_and_create_datasets(df_close)

    print("Entreprises disponibles :", list(datasets.keys()))
    company = input("Entrez le nom de l'entreprise à évaluer : ").strip()
    results = evaluate_company(datasets, company)

    print("\n=== Résultats pour", company, "===")
    for r in results:
        print(f"{r['Modèle']}: MSE={r['MSE']:.4f}, RMSE={r['RMSE']:.4f}, Params={r['Best Params']}")

if __name__ == '__main__':
    main()