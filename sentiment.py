import glob
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

import pytz
import torch
import pandas as pd
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


def get_texts_timestamps(news_data, company="Tesla"):
    news_texts = []
    news_timestamps = []
    ny_tz = pytz.timezone("America/New_York")

    for date in news_data.get(company, {}):
        for article in news_data[company][date]:
            dt_utc = datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00"))
            dt_ny = dt_utc.astimezone(ny_tz).replace(minute=0, second=0, microsecond=0)
            full_text = f"{article['title']} {article['description']}"
            news_texts.append(full_text)
            news_timestamps.append(dt_ny)

    return news_texts, news_timestamps


def get_sentiments(model_path, texts, device=None, batch_size=16):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) charger tokenizer + modèle
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model     = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    sentiments = []
    id2label = model.config.id2label  # dictionnaire mapping id->label

    # 2) inférence par batch
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            outputs = model(**enc)
            preds   = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

            # on convertit en label textuel
            sentiments.extend([id2label[p] for p in preds])

    return sentiments


def align_timestamps(timestamps):
    aligned = []
    for ts in timestamps:
        h, m = ts.hour, ts.minute
        # 1) 9h30 ≤ ts < 15h
        if (h > 9 or (h == 9 and m >= 30)) and h < 15:
            aligned.append(ts.replace(minute=0, second=0, microsecond=0))
        # 2) 15h ≤ ts < 24h
        elif h >= 15:
            aligned.append(ts.replace(hour=15, minute=0, second=0, microsecond=0))
        # 3) 0h ≤ ts < 9h30
        else:
            veille = ts - timedelta(days=1)
            aligned.append(veille.replace(hour=15, minute=0, second=0, microsecond=0))
    return aligned

def sentiment():
    # --- chargement des news ---
    with open("news_data2.json", "r") as f:
        news_data = json.load(f)

    news_texts, news_timestamps = get_texts_timestamps(news_data, company="Tesla")

    # --- prédiction des sentiments ---
    model_folder = "/Users/victorbarbier/Documents/Dauphine/Python for Data Science/Projet_final/bert-base-uncased_finetuned"
    sentiments = get_sentiments(model_folder, news_texts)

    # --- construction d'un DataFrame pour visualiser tout ça ---
    df_news = pd.DataFrame({
        "timestamp": news_timestamps,
        "text":      news_texts,
        "sentiment": sentiments
    })

    df_news['aligned_timestamp'] = align_timestamps(df_news['timestamp'])

    print("Aperçu avec les timestamps alignés :")
    print(df_news.head(10))
    pivot = (
        df_news
        .groupby(['aligned_timestamp', 'sentiment'])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print("\nNombre d'articles par sentiment et par tranche :")
    print(pivot)

    print("\nRépartition des sentiments :")
    print(df_news['sentiment'].value_counts())

    return df_news


if __name__ == "__main__":
    # {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

    df_results = sentiment()
