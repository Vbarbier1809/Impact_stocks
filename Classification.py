import pandas as pd
import numpy as np
import glob
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import shap
import os
import optuna
import joblib
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands
from sklearn.model_selection import train_test_split

def get_returns_df(folder="Historique", close_col="Close", horizon=20):

    csv_pattern = os.path.join(folder, "*.csv")
    filepaths = glob.glob(csv_pattern)
    print(f"Fichiers trouvés : {filepaths}")

    returns_dict = {}

    for filepath in filepaths:
        company_name = os.path.splitext(os.path.basename(filepath))[0]
        
        # 1) Lecture et filtrage
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.dropna(subset=[close_col], inplace=True)   # on ne garde que les lignes ayant un Close valide
        df = df[[close_col]]                          # on ne conserve que cette colonne

        # Vérification si la colonne existe
        if close_col not in df.columns:
            print(f"Attention : la colonne '{close_col}' est absente dans {filepath}.")
            continue

        # 2) Créer "Close_Horizon" (shift de -horizon)
        df["Close_Horizon"] = df[close_col].shift(-horizon)

        # 3) Calculer "horizon_return"
        df["horizon_return"] = (df["Close_Horizon"] - df[close_col]) / df[close_col]

        # 4) Créer la colonne "label" (exemple de logique : buy=2, hold=1, sell=0)
        #    Ici, on choisit un critère fictif : 
        #    - Si horizon_return > +5% => label=2 (buy)
        #    - Si horizon_return entre -2% et +5% => label=1 (hold)
        #    - Sinon => label=0 (sell)
        def assign_label(r):
            if pd.isna(r):
                return np.nan
            elif r > 0.05:
                return 2
            elif r > -0.02:
                return 1
            else:
                return 0

        df["label"] = df["horizon_return"].apply(assign_label)

        # (Optionnel) on peut dropper les dernières lignes qui n'ont pas de Close_Horizon
        df.dropna(subset=["Close_Horizon"], inplace=True)

        # 5) Stockage dans le dictionnaire
        returns_dict[company_name] = df

    return returns_dict


def add_technical_features(df, close_col='Close'):
    if close_col not in df.columns:
        raise ValueError(f"La colonne '{close_col}' est absente du DataFrame.")

    # S'assurer qu'il n'y ait pas de NaN gênants avant
    df = df.copy()  # pour ne pas modifier l'original
    df.dropna(subset=[close_col], inplace=True)
    
    sma_20 = SMAIndicator(close=df[close_col], window=20, fillna=False).sma_indicator()
    df['SMA_20'] = sma_20
    ema_20 = EMAIndicator(close=df[close_col], window=20, fillna=False).ema_indicator()
    df['EMA_20'] = ema_20
    rsi_14 = RSIIndicator(close=df[close_col], window=14, fillna=False).rsi()
    df['RSI_14'] = rsi_14
    macd_indicator = MACD(close=df[close_col], fillna=False)
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    boll = BollingerBands(close=df[close_col], window=20, window_dev=2, fillna=False)
    df['Bollinger_High'] = boll.bollinger_hband()
    df['Bollinger_Low'] = boll.bollinger_lband()
    
    # 6) Rolling Volatility 20 jours (on peut utiliser std sur 'Close')
    #    ou plus spécifiquement sur les rendements journaliers
    df['Rolling_Volatility_20'] = df[close_col].rolling(window=20).std()
    
    # 7) ROC (Rate Of Change), ex. période=10
    roc_10 = ROCIndicator(close=df[close_col], window=10, fillna=False).roc()
    df['ROC_10'] = roc_10
    
    # Nettoyer ensuite les premières lignes qui sont NaN
    df.dropna(inplace=True)
    
    return df

def prepare_classification_data(labeled_data, label_col='label',
                               cols_to_drop=['Close', 'Close_Horizon',
                                             'Weekly_return', 'Next Day Close'],
                               test_size=0.2, random_state=42):
    
    # 1) Concaténer tous les DF
    #    On va faire un seul gros DataFrame, on peut ajouter un "ignore_index=True" 
    #    si on ne veut plus distinguer les dates / entreprises.
    
    df_full = pd.concat(labeled_data.values(), axis=0, ignore_index=True)
    # Optionnel: vous pouvez ajouter une colonne "Entreprise" si vous voulez conserver l'info
    # Ex: df_comp['Entreprise'] = comp_name, et concat(...

    # Vérification que la colonne label_col existe
    if label_col not in df_full.columns:
        raise ValueError(f"La colonne de label '{label_col}' est introuvable dans df_full.")
    
    # 2) Séparer Y et X
    y = df_full[label_col].copy()
    X = df_full.drop(columns=[label_col], axis=1)
    feature_names = X.columns.to_list()
    
    # 3) Supprimer de X les colonnes indésirables
    #    (ex. 'Close', 'Close_Horizon', 'Weekly_return', 'Next Day Close')
    existing_cols_to_drop = [c for c in cols_to_drop if c in X.columns]
    X.drop(columns=existing_cols_to_drop, axis=1, inplace=True)
    
    # 4) Standardiser X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_names

def train_xgboost(X_train, y_train, X_test, y_test, feature_names=None):

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    grid = GridSearchCV(
        xgb, 
        param_grid=param_grid, 
        scoring='accuracy', 
        cv=3, 
        n_jobs=-1, 
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("=== XGBoost ===")
    print("Meilleurs paramètres :", grid.best_params_)
    print("Meilleur score (CV)  :", grid.best_score_)

    # Évaluation sur le test set
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    # SHAP
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    # Par exemple, si vous avez 3 classes: shap_values est une liste
    # On peut afficher la summary plot pour une classe spécifique
    # Disons la classe "buy" = 2
    # -> shap_values[:, :, 2] pour la 3e classe.
    # S'il n'y a que 2 classes, shap_values.shape = (n_samples, n_features).
    # On suppose ici 2 classes pour l'exemple:
    if len(shap_values.shape) == 3:
        # multi-class => on choisit la classe index=2 si "buy"
        class_index = 2  # adapter selon votre code
        shap.summary_plot(shap_values[:, :, class_index], 
                          X_test, feature_names=feature_names)
    else:
        # binaire => shap_values (n_samples, n_features)
        shap.summary_plot(shap_values, X_test, feature_names=feature_names)

    return {
        'best_model': best_model,
        'classification_report': report,
        'best_params': grid.best_params_,
        'best_score': grid.best_score_
    }


def train_random_forest(X_train, y_train, X_test, y_test, feature_names=None):
    """
    GridSearchCV pour un RandomForestClassifier,
    puis classification_report + SHAP summary plot (TreeExplainer).
    """

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        rf, 
        param_grid=param_grid, 
        scoring='accuracy', 
        cv=3, 
        n_jobs=-1, 
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("=== Random Forest ===")
    print("Meilleurs paramètres :", grid.best_params_)
    print("Meilleur score (CV)  :", grid.best_score_)

    # Évaluation
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    # SHAP
    # Comme c'est un modèle d'arbres, TreeExplainer fonctionne bien
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    # Si classification binaire => shap_values[0] est la contribution classe=0, shap_values[1] pour classe=1, etc.
    # Pour multiclass => shap_values = list ou array de dimension 3
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # on suppose 3 classes => shap_values[2] si vous voulez la classe "buy"
        class_index = 0  # adapter
        shap.summary_plot(shap_values[class_index], X_test, feature_names=feature_names)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names)

    return {
        'best_model': best_model,
        'classification_report': report,
        'best_params': grid.best_params_,
        'best_score': grid.best_score_
    }

def train_knn(X_train, y_train, X_test, y_test, feature_names=None):
    """
    GridSearchCV pour un KNeighborsClassifier,
    affichage classification_report,
    et SHAP via KernelExplainer (optionnel, plus lent).
    """

    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn = KNeighborsClassifier()

    grid = GridSearchCV(
        knn, 
        param_grid=param_grid, 
        scoring='accuracy', 
        cv=3, 
        n_jobs=-1, 
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("=== k-Nearest Neighbors ===")
    print("Meilleurs paramètres :", grid.best_params_)
    print("Meilleur score (CV)  :", grid.best_score_)

    # Évaluation
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    # SHAP sur un modèle non basé sur des arbres => KernelExplainer
    # ATTENTION: KernelExplainer peut être très lent selon la taille de X_test.
    # On peut prendre un sous-échantillon
    sample_for_shap = X_test[:50]  # ex. 50 points

    explainer = shap.KernelExplainer(best_model.predict_proba, sample_for_shap)
    # On calcule ensuite les shap_values sur le même échantillon ou un petit subset
    shap_values = explainer.shap_values(sample_for_shap, nsamples=100)

    # Si multi-class => shap_values est une liste
    # On peut plot pour la classe "buy" (ex. index=2) ou binaire => shap_values[1].
    # Ex pour binaire => shap_values[1].
    if isinstance(shap_values, list) and len(shap_values) > 1:
        class_index = 1
        shap.summary_plot(shap_values[class_index], sample_for_shap, feature_names=feature_names)
    else:
        shap.summary_plot(shap_values, sample_for_shap, feature_names=feature_names)

    return {
        'best_model': best_model,
        'classification_report': report,
        'best_params': grid.best_params_,
        'best_score': grid.best_score_
    }


def train_logistic_regression(X_train, y_train, X_test, y_test, feature_names=None):
    """
    GridSearchCV pour LogisticRegression,
    classification_report + SHAP (KernelExplainer)
    """

    param_grid = {
        'C': [0.01, 0.1, 1.0, 10],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2']
        # 'elasticnet' + 'l1_ratio' si vous voulez tester du l1/l2 mix
    }

    logreg = LogisticRegression(random_state=42, max_iter=1000)

    grid = GridSearchCV(
        logreg, 
        param_grid=param_grid, 
        scoring='accuracy', 
        cv=3, 
        n_jobs=-1, 
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("=== Logistic Regression ===")
    print("Meilleurs paramètres :", grid.best_params_)
    print("Meilleur score (CV)  :", grid.best_score_)

    # Évaluation
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    # SHAP (KernelExplainer)
    sample_for_shap = X_test[:50]
    explainer = shap.KernelExplainer(best_model.predict_proba, sample_for_shap)
    shap_values = explainer.shap_values(sample_for_shap, nsamples=100)

    # binaire => shap_values[1], multi-class => shap_values[2], ...
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # par ex. la classe index=1
        shap.summary_plot(shap_values[1], sample_for_shap, feature_names=feature_names)
    else:
        shap.summary_plot(shap_values, sample_for_shap, feature_names=feature_names)

    return {
        'best_model': best_model,
        'classification_report': report,
        'best_params': grid.best_params_,
        'best_score': grid.best_score_
    }


def train_svm(X_train, y_train, X_test, y_test, feature_names=None):
    """
    GridSearchCV pour un SVC,
    classification_report + SHAP (KernelExplainer).
    """

    param_grid = {
        'C': [0.1, 1.0, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    svc = SVC(probability=True, random_state=42)

    grid = GridSearchCV(
        svc, 
        param_grid=param_grid, 
        scoring='accuracy', 
        cv=3, 
        n_jobs=-1, 
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("=== SVM (SVC) ===")
    print("Meilleurs paramètres :", grid.best_params_)
    print("Meilleur score (CV)  :", grid.best_score_)

    # Évaluation
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    # SHAP (KernelExplainer)
    sample_for_shap = X_test[:50]
    explainer = shap.KernelExplainer(best_model.predict_proba, sample_for_shap)
    shap_values = explainer.shap_values(sample_for_shap, nsamples=100)

    # Affichage
    if isinstance(shap_values, list) and len(shap_values) > 1:
        class_index = 1
        shap.summary_plot(shap_values[class_index], sample_for_shap, feature_names=feature_names)
    else:
        shap.summary_plot(shap_values, sample_for_shap, feature_names=feature_names)

    return {
        'best_model': best_model,
        'classification_report': report,
        'best_params': grid.best_params_,
        'best_score': grid.best_score_
    }



