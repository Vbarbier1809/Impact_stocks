import pandas as pd
import yfinance as yf
import pandas as pd
import datetime
from datetime import timedelta
import os
from Company import companies
from Ratios import ratios
from scrap import *
from Clustering import *
from Classification import *
from Regression import *
from deep import *
from Scrapping_news import *
from Finetuning_BERT import *
from sentiment import * 

def main():
    # --- TP1 -------
    # collect_and_export_ratios(companies, ratios, csv_filename="ratios_companies.csv")
    # fetch_and_save_historical_data(companies_dict=companies, years=5, output_folder="Historique")
    # print('TP1 done!')

    # ----TP2------
        # Paths and column setups
    csv_path = "ratios_companies.csv"
    columns_financial = [
        'forwardPE', 'beta', 'priceToBook', 'operatingMargins',
        'returnOnEquity', 'profitMargins'
    ]

    # Financial clustering (KMeans)
    data_financial, data_scaled_financial, scaler_fin = \
        preprocess_for_financial_clustering(
            csv_path=csv_path,
            columns_to_keep=columns_financial
        )

    print("Data original (5 premières lignes) :")
    print(data_financial.head())
    print("\nData scaled (5 premières lignes) :")
    print(pd.DataFrame(data_scaled_financial, 
                         columns=columns_financial).head())

    ks, inertias = find_optimal_k(data_scaled_financial, k_range=range(1, 11))
    df_km = cluster_and_tsne(data_scaled_financial, data_financial, k=5)

    # Risk clustering (Agglomerative)
    columns_risk = [
        'debtToEquity', 'beta', 'operatingMargins',
        'profitMargins', 'returnOnAssets', 'trailingEps'
    ]
    data_risk, data_scaled_risk, scaler_risk = \
        preprocess_for_financial_clustering(
            csv_path=csv_path,
            columns_to_keep=columns_risk
        )

    print("Data original (5 premières lignes) risk :")
    print(data_risk.head())
    print("\nData scaled (5 premières lignes) risk :")
    print(pd.DataFrame(data_scaled_risk, 
                         columns=columns_risk).head())

    plot_dendrogram(data_scaled_risk, method='ward')
    df_hier = agglomerative_cluster_and_summary(
        data_scaled=data_scaled_risk,
        original_df=data_risk,
        n_clusters=3
    )
    for c in sorted(df_hier['Cluster'].unique()):
        print(f"\nCluster {c} :", df_hier[df_hier['Cluster']==c].index.tolist())

    # Rendement correlation and DBSCAN
    folder_path = "/Users/victorbarbier/Documents/Dauphine/Python for Data Science/Historique"
    corr_matrix = plot_rendement_corr_and_dendrogram(folder_path)

    plot_dbscan_clusters(
        corr_df=corr_matrix,
        eps=1.2,
        min_samples=4
    )

    plot_k_distance(
        StandardScaler().fit_transform(corr_matrix.values),
        k=4
    )

    # Silhouette scores
    scores = silhouette_scores_all(
        folder_path=folder_path,
        kmeans_k=3,
        hier_k=3,
        dbscan_eps=1.0,
        dbscan_min_samples=4
    )
    print("Silhouette scores summary:", scores)

    # -----TP3--------
        
    # 1) Collecte des historiques et calcul des labels
    returns = get_returns_df(folder="Historique", horizon=20)
    # 2) Ajout des features techniques
    for name, df in returns.items():
        returns[name] = add_technical_features(df, close_col='Close')
    # 3) Préparation X/Y
    X_train, X_test, y_train, y_test, feats = prepare_classification_data(returns)
    # 4) Entraînement et évaluation du XGBoost
    model = train_xgboost(X_train, y_train, X_test, y_test, feature_names=feats)
    # 5) Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_best_model.joblib")
    print("Modèle XGBoost enregistré dans models/xgboost_best_model.joblib")

    # -------TP4-----------
    folder_path = "/Users/victorbarbier/Documents/Dauphine/Python for Data Science/Companies_historical_data"
    df_close = load_and_prepare_data(folder_path)
    datasets = preprocess_and_create_datasets(df_close)

    print("Entreprises disponibles :", list(datasets.keys()))
    company = input("Entrez le nom de l'entreprise à évaluer : ").strip()
    results = evaluate_company(datasets, company)

    print("\n=== Résultats pour", company, "===")
    for r in results:
        print(f"{r['Modèle']}: MSE={r['MSE']:.4f}, RMSE={r['RMSE']:.4f}, Params={r['Best Params']}")


    # -------TP5----------
    deep()

    # ------TP6--------
    all_news = {}
    for company in companies:
        raw_data = fetch_raw_news(company)
        articles_by_date = parse_news_by_date(company, raw_data)
        all_news[company] = articles_by_date

    # Sauvegarde dans un fichier JSON
    with open("news_data2.json", "w", encoding="utf-8") as f:
        json.dump(all_news, f, indent=4, ensure_ascii=False)

    print("Extraction et organisation terminées. Résultats dans 'news_data2.json'.")


    # -------TP7-------
    ### TOUT EST COMMENTE CAR SI ON LANCE LE FINETUNING, CELA PREND BEAUCOUP DE TEMPS
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset = load_and_prepare_datasets()
    # text_column = None
    # train_columns = dataset["train"].column_names

    # if "text" in train_columns:
    #     text_column = "text"
    # elif "sentence" in train_columns:
    #     text_column = "sentence"
    # else:
    #     for col in train_columns:
    #         if "text" in col.lower() or "sentence" in col.lower() or "tweet" in col.lower() or "content" in col.lower():
    #             text_column = col
    #             break

    # if text_column is None:
    #     raise ValueError("Impossible de déterminer la colonne contenant le texte dans le dataset")

    # print(f"Utilisation de la colonne '{text_column}' pour le texte")

    # # 2. Finetuning du modèle BERT
    # bert_tokenizer, bert_model, bert_trainer = train_model(
    #     "bert-base-uncased",
    #     dataset,
    #     batch_size=16,
    #     num_epochs=3
    # )

    # # 3. Finetuning du modèle FinBERT
    # finbert_tokenizer, finbert_model, finbert_trainer = train_model(
    #     "yiyanghkust/finbert-tone",  # FinBERT pré-entraîné sur des données financières
    #     dataset,
    #     batch_size=16,
    #     num_epochs=3
    # )

    # # 4. Évaluation détaillée des modèles
    # print("\n--- Évaluation détaillée du modèle BERT ---")
    # evaluate_detailed(bert_model, bert_tokenizer, dataset, text_column)

    # print("\n--- Évaluation détaillée du modèle FinBERT ---")
    # evaluate_detailed(finbert_model, finbert_tokenizer, dataset, text_column)

    # # 5. Comparaison des performances
    # print("\n--- Comparaison des performances ---")
    # bert_eval = bert_trainer.evaluate()
    # finbert_eval = finbert_trainer.evaluate()

    # print("Performances BERT:")
    # for metric, value in bert_eval.items():
    #     print(f"{metric}: {value:.4f}")

    # print("\nPerformances FinBERT:")
    # for metric, value in finbert_eval.items():
    #     print(f"{metric}: {value:.4f}")

 #------TP8------
    sentiment()

if __name__ == "__main__":
    main()