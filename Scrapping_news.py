import requests
import json
from datetime import datetime, timedelta
import os
import pandas as pd

from Company import companies
from api import api_key

def fetch_raw_news(company_name):
    url = "https://newsapi.org/v2/everything"
    now = datetime.now()
    first_day = (now - timedelta(days=10)).strftime('%Y-%m-%d')
    last_day = now.strftime('%Y-%m-%d')

    params = {
        "apiKey": api_key,
        "q": company_name,
        "language": "en",
        "pageSize": 100,
        "from": first_day,
        "to": last_day,
        "sources": "financial-post,the-wall-street-journal,bloomberg,the-washington-post"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erreur lors de la requête pour {company_name} : code {response.status_code}")
        return {"articles": []}

def parse_news_by_date(company_name, data):
    news_by_date = {}
    for article in data.get("articles", []):
        title = article.get("title", "")
        description = article.get("description", "")
        source_name = article.get("source", {}).get("name", "")
        published_at = article.get("publishedAt", "")
        if company_name.lower() in title.lower() or company_name.lower() in description.lower():
            date_str = published_at.split("T")[0] if "T" in published_at else published_at
            article_dict = {
                "title": title,
                "description": description,
                "publishedAt": published_at,
                "source": source_name
            }
            if date_str not in news_by_date:
                news_by_date[date_str] = []
            news_by_date[date_str].append(article_dict)

    return news_by_date

if __name__ == "__main__":
    all_news = {}
    for company in companies:
        raw_data = fetch_raw_news(company)
        articles_by_date = parse_news_by_date(company, raw_data)
        all_news[company] = articles_by_date

    # Sauvegarde dans un fichier JSON
    with open("news_data2.json", "w", encoding="utf-8") as f:
        json.dump(all_news, f, indent=4, ensure_ascii=False)

    print("Extraction et organisation terminées. Résultats dans 'news_data2.json'.")