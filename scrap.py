import yfinance as yf
import pandas as pd
import datetime
from datetime import timedelta
import os
from Company import companies
from Ratios import ratios

def collect_and_export_ratios(companies, ratios_list, csv_filename="ratios_companies.csv"):

    ratios_data = {ratio: [] for ratio in ratios_list}
    company_names = []

    for company, symbol in companies.items():
        ticker = yf.Ticker(symbol)
        info = ticker.info

        company_names.append(company)

        for ratio in ratios_list:
            value = info.get(ratio, None) 
            ratios_data[ratio].append(value)

    df = pd.DataFrame(ratios_data, index=company_names)

    df.to_csv(csv_filename, index=True)
    print('collect_and_export_ratios done!')

    return df

def fetch_and_save_historical_data(companies_dict, years=5, output_folder="Historique"):
    end_date = datetime.datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for company_name, symbol in companies_dict.items():
        print(f"\n=== Traitement de {company_name} ({symbol}) ===")
        print("Téléchargement de l’historique boursier sur 5 ans...")
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            print(f"Aucune donnée trouvée pour {symbol}.")
            continue
        df = pd.DataFrame()
        df['Close'] = data['Close']
        df['Next Day Close'] = df['Close'].shift(-1)
        df['Rendement'] = (df['Next Day Close'] - df['Close']) / df['Close']
        csv_filename = f"{company_name}_historique.csv"
        full_path = os.path.join(output_folder, csv_filename)
        df.to_csv(full_path, index=True)
        print(f"Fichier enregistré : {full_path}")
        print('fetch_and_save_historical_data done!')

def main():
    # 1. Collecte et export des ratios
    collect_and_export_ratios(companies, ratios, csv_filename="ratios_companies.csv")
    
    # 2. Fetch et sauvegarde de l'historique boursier
    fetch_and_save_historical_data(companies_dict=companies, years=5, output_folder="Historique")

if __name__ == "__main__":
    main()