import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import glob
from sklearn.neighbors import NearestNeighbors

def preprocess_for_financial_clustering(csv_path, columns_to_keep=None):
    df = pd.read_csv(csv_path, index_col=0)
    if columns_to_keep is not None:
        df = df[columns_to_keep]
    df.dropna(axis=0, inplace=True)
    scaler = StandardScaler()
    data_scaled_np = scaler.fit_transform(df.values)
    data_scaled = pd.DataFrame(data_scaled_np, index=df.index, columns=df.columns)

    return df, data_scaled, scaler

def find_optimal_k(data, k_range=range(1, 11), random_state=42):
    inertias = []
    ks = list(k_range)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state)
        km.fit(data)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(ks, inertias, marker='o')
    plt.xlabel('Nombre de clusters k')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude pour choix de k')
    plt.show()
    return ks, inertias

def cluster_and_tsne(data_scaled, original_df, k, random_state=42):
    # KMeans et ajout de la colonne Cluster
    km = KMeans(n_clusters=k, random_state=random_state)
    clusters = km.fit_predict(data_scaled)
    df_with_clusters = original_df.copy()
    df_with_clusters['Cluster'] = clusters
    
    # Caractéristiques moyennes par cluster
    print("Caractéristiques moyennes des clusters :")
    cluster_summary = df_with_clusters.groupby('Cluster').mean()
    print(cluster_summary)
    
    # t-SNE pour visualisation
    tsne = TSNE(n_components=2, random_state=random_state)
    tsne_results = tsne.fit_transform(data_scaled)
    plt.figure()
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters)
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.title(f'Visualisation t-SNE des clusters (k={k})')
    plt.show()
    
    return df_with_clusters
def agglomerative_cluster_and_summary(data_scaled, original_df, n_clusters=3):
    # Instanciation et fit
    agglom = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agglom.fit_predict(data_scaled)
    
    # Ajout de la colonne au DataFrame
    df_with_clusters = original_df.copy()
    df_with_clusters['Cluster'] = labels
    
    # Résumé des métriques par cluster
    print(f"Résumé des moyennes par cluster (n_clusters={n_clusters}) :")
    summary = df_with_clusters.groupby('Cluster').mean()
    print(summary)
    
    return df_with_clusters

def plot_dendrogram(data_scaled, method='ward'):
    # Calcul de la matrice de linkage
    Z = linkage(data_scaled, method=method)
    
    # Tracé du dendrogramme
    plt.figure(figsize=(10, 6))
    dendrogram(Z, truncate_mode='level', p=5, leaf_rotation=90.)
    plt.title(f"Dendrogramme (linkage method = '{method}')")
    plt.xlabel('Échantillons')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

def plot_rendement_corr_and_dendrogram(folder_path):
    # 1) Chargement et assemblage
    rendement_dict = {}
    for file in glob.glob(f"{folder_path}/*.csv"):
        df = pd.read_csv(file)
        company = file.split("/")[-1].replace("_historical_data.csv", "")
        rendement_dict[company] = df["Rendement"]
    rendement_df = pd.DataFrame(rendement_dict)
    rendement_df.fillna(rendement_df.mean(), inplace=True)

    # 2) Calcul de la matrice de corrélation
    corr_df = rendement_df.corr()

    # 3) Heatmap de la corrélation
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr_df.values, interpolation='nearest', aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(corr_df)))
    ax.set_xticklabels(corr_df.columns, rotation=90)
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df.index)
    fig.colorbar(cax, ax=ax, label='Coefficient de corrélation')
    ax.set_title("Heatmap de la corrélation des rendements")
    plt.tight_layout()
    plt.show()

    # 4) Dendrogramme hiérarchique
    Z = linkage(corr_df, method="ward")
    plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        labels=corr_df.columns,
        leaf_rotation=90,
        color_threshold=None
    )
    plt.title("Dendrogramme basé sur les corrélations de rendement")
    plt.xlabel("Entreprises")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return corr_df

def plot_k_distance(X, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # on prend la distance à la k-ième NN pour chaque point
    k_dist = np.sort(distances[:, -1])
    plt.plot(k_dist)
    plt.ylabel(f"Distance au {k}ᵉ NN")
    plt.xlabel("Points triés par distance")
    plt.title("k-distance plot pour choisir eps")
    plt.show()

def plot_dbscan_clusters(corr_df, eps=0.5, min_samples=5, random_state=42):
    # 1) Standardisation
    scaler = StandardScaler()
    X = scaler.fit_transform(corr_df.values)

    # 2) DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    unique_labels = sorted(set(labels))
    n_clusters = len([lab for lab in unique_labels if lab != -1])

    print(f"DBSCAN trouvé {n_clusters} clusters (+ outliers: {labels.tolist().count(-1)})")

    # 3) t-SNE pour visualisation
    tsne = TSNE(n_components=2, random_state=random_state)
    X2 = tsne.fit_transform(X)
    plt.figure(figsize=(8,6))
    for lab in unique_labels:
        mask = (labels == lab)
        plt.scatter(
            X2[mask, 0],
            X2[mask, 1],
            label = f"Cluster {lab}" if lab != -1 else "Outliers",
            s=50,
            alpha=0.7
        )
    plt.legend()
    plt.title("t-SNE des entreprises (DBSCAN)")
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.tight_layout()
    plt.show()

def silhouette_scores_all(folder_path, kmeans_k=3, hier_k=3, dbscan_eps=1.0, dbscan_min_samples=4):
    # 1) Chargement et assemblage des rendements
    rendement_dict = {}
    for file in glob.glob(f"{folder_path}/*.csv"):
        df = pd.read_csv(file)
        company = file.split("/")[-1].replace("_historical_data.csv", "")
        rendement_dict[company] = df["Rendement"]
    rendement_df = pd.DataFrame(rendement_dict)
    rendement_df.fillna(rendement_df.mean(), inplace=True)

    # 2) Matrice de corrélation et mise à l’échelle
    corr_df = rendement_df.corr()
    X = StandardScaler().fit_transform(corr_df.values)

    # 3) KMeans
    km = KMeans(n_clusters=kmeans_k, random_state=42)
    labels_km = km.fit_predict(X)
    score_km = silhouette_score(X, labels_km)

    # 4) Agglomerative
    ag = AgglomerativeClustering(n_clusters=hier_k, linkage='ward')
    labels_ag = ag.fit_predict(X)
    score_ag = silhouette_score(X, labels_ag)

    # 5) DBSCAN
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels_db = db.fit_predict(X)
    # exclure les outliers pour le calcul
    mask = labels_db != -1
    if len(set(labels_db[mask])) > 1:
        score_db = silhouette_score(X[mask], labels_db[mask])
    else:
        score_db = float('nan')

    # 6) Affichage
    print("Silhouette scores :")
    print(f"  KMeans (k={kmeans_k})        : {score_km:.3f}")
    print(f"  Hierarchical (k={hier_k})   : {score_ag:.3f}")
    print(f"  DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples}) : {score_db:.3f}")

    return {
        'KMeans': score_km,
        'Hierarchical': score_ag,
        'DBSCAN': score_db
    }


def main():
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


if __name__ == "__main__":
    main()
