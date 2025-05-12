# Étape 0 : Importer les bibliothèques
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour les tracés
# %matplotlib inline
# plt.style.use('seaborn-whitegrid')
# sns.set()


# Étape 1 : Définir la fonction de prétraitement

def preprocess_for_financial_clustering(csv_path, columns_to_keep=None):
    # 1. Charger le CSV ; on suppose que la première colonne est l'index (noms d'entreprises)
    df = pd.read_csv(csv_path, index_col=0)

    # 2. Filtrer les colonnes pertinentes si columns_to_keep n'est pas None
    if columns_to_keep is not None:
        df = df[columns_to_keep]

    # 3. Supprimer les lignes avec des valeurs manquantes
    df.dropna(axis=0, inplace=True)

    # 4. Standardisation
    scaler = StandardScaler()
    data_scaled_np = scaler.fit_transform(df.values)
    data_scaled = pd.DataFrame(data_scaled_np, index=df.index, columns=df.columns)

    return df, data_scaled, scaler

# Étape 2 : Définir la fonction elbow_method

def elbow_method(data_scaled, max_k=10):
    inertias = []
    K_range = range(1, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(data_scaled)
        inertias.append(kmeans.inertia_)

    # Tracé de la courbe
    plt.figure(figsize=(7, 5))
    plt.plot(K_range, inertias, 'o--')
    plt.title("Méthode du coude (Elbow Method)")
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertie (SSE)")
    plt.show()

# Étape 3 : Définir la fonction do_kmeans_clustering

def do_kmeans_clustering(data_scaled, n_clusters=4, perplexity=30, learning_rate=200):
    # 1. Clustering KMeans
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(data_scaled)
    labels = kmeans.labels_

    # On crée un DataFrame pour stocker les résultats
    df_clusters = data_scaled.copy()
    df_clusters['cluster'] = labels

    # 2. Visualisation TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    tsne_results = tsne.fit_transform(data_scaled)  # data_scaled est un DataFrame, mais TSNE attend un array

    df_clusters['TSNE_1'] = tsne_results[:, 0]
    df_clusters['TSNE_2'] = tsne_results[:, 1]

    # 3. Plot t-SNE coloré par cluster
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='TSNE_1',
        y='TSNE_2',
        hue='cluster',
        data=df_clusters,
        palette='Set2'
    )
    plt.title(f"K-Means (k={n_clusters}) - Visualisation t-SNE")
    plt.legend()
    plt.show()

    return df_clusters