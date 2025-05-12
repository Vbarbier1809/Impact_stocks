import os
import json

#MAJ

def update_existing_news(company_name, new_articles_by_date, json_file="news_data.json"):
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    if company_name not in existing_data:
        existing_data[company_name] = {}

    for date_str, articles in new_articles_by_date.items():
        if date_str not in existing_data[company_name]:
            existing_data[company_name][date_str] = []

        existing_articles = existing_data[company_name][date_str]
        existing_titles = {art.get("title", "") for art in existing_articles}

        for art in articles:
            title = art.get("title", "")
            if title not in existing_titles:
                existing_articles.append(art)
        else:
            return 'FOU'

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print(f"Les actualités de '{company_name}' ont été mises à jour dans '{json_file}'.")


if __name__ == "__main__":
    company = "Apple"

    new_data = {
        "2025-05-05": [
            {
                "title": "Apple releases new device",
                "description": "Details about the device...",
                "publishedAt": "2025-04-04T10:00:00Z",
                "source": "Bloomberg"
            }
        ]
    }
    update_existing_news(company, new_data, "news_data2.json")
