import numpy as np
import pandas as pd

# Daten für das Jahr 2023 generieren
days = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Produkte für Frühstück und andere Mahlzeiten
breakfast_products = ['Croissant']
other_products = ['Burger', 'Salat', 'Cola']

# Feiertage in Österreich 2023
feiertage = {
    '2023-01-01': 'Neujahr',
    '2023-01-06': 'Heilige Drei Könige',
    '2023-04-09': 'Ostersonntag',
    '2023-04-10': 'Ostermontag',
    '2023-05-01': 'Staatsfeiertag',
    '2023-05-18': 'Christi Himmelfahrt',
    '2023-05-28': 'Pfingstsonntag',
    '2023-05-29': 'Pfingstmontag',
    '2023-06-08': 'Fronleichnam',
    '2023-08-15': 'Mariä Himmelfahrt',
    '2023-10-26': 'Nationalfeiertag',
    '2023-11-01': 'Allerheiligen',
    '2023-12-08': 'Mariä Empfängnis',
    '2023-12-25': 'Weihnachten',
    '2023-12-26': 'Stefanitag'
}

# Simuliere tägliche Verkaufszahlen für Frühstücksprodukte
breakfast_data = {product: np.random.poisson(lam=100, size=len(days)) for product in breakfast_products}

# Simuliere tägliche Verkaufszahlen für andere Produkte (Mittag- und Abendessen)
other_data = {product: np.random.poisson(lam=150, size=len(days)) for product in other_products}

# Erstelle einen DataFrame
sales_df = pd.DataFrame(breakfast_data, index=days)
sales_df.index.name = 'Date'
other_sales_df = pd.DataFrame(other_data, index=days)

# Füge die Jahreszeit hinzu
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Fruehling'
    elif month in [6, 7, 8]:
        return 'Sommer'
    else:
        return 'Herbst'

# Füge die Jahreszeit zu jedem Tag hinzu
sales_df.insert(1, 'Jahreszeit', [get_season(day.month) for day in days])

# Wetterbedingungen anpassen (Schnee nur im Winter möglich)
def get_weather(season):
    if season == 'Winter':
        return np.random.choice(['Sonnig', 'Regnerisch', 'Bewoelkt', 'Schnee'])
    else:
        return np.random.choice(['Sonnig', 'Regnerisch', 'Bewoelkt'])

sales_df['Wetter'] = sales_df['Jahreszeit'].apply(get_weather)

# Füge den Wochentag hinzu
sales_df['Wochentag'] = days.day_name()

# Feiertage zuweisen
sales_df['Feiertag'] = sales_df.index.strftime('%Y-%m-%d').map(feiertage).notna()

# Füge die Feiertagtyp-Spalte hinzu
sales_df['Feiertagtyp'] = sales_df.index.strftime('%Y-%m-%d').map(feiertage).fillna('Kein Feiertag')

# Definiere verschiedene Event-Typen (inklusive 'Kein Event' für Tage ohne besondere Events)
event_types = ['Kein Event', 'Konzert', 'Fussballspiel', 'Festival', 'Messe', 'Strassenfest']
sales_df['Event'] = np.random.choice(event_types, size=len(days), p=[0.9, 0.025, 0.025, 0.02, 0.015, 0.015]) != 'Kein Event'

# Füge die Eventtyp-Spalte hinzu
event_labels = ['Konzert', 'Fussballspiel', 'Messe']
sales_df['Eventtyp'] = sales_df['Event'].apply(lambda x: np.random.choice(event_labels) if x else 'Kein Event')

# Trends einfügen

# 2. Bei Feiertagen immer viel mehr Verkäufe (100 % mehr für alle Produkte)
for i, day in enumerate(days):
    feiertag = sales_df.iloc[i]['Feiertag']
    if feiertag:
        for product in other_products:
            other_sales_df.loc[day, product] = np.round(other_sales_df.loc[day, product] * 2).astype(int)
        for product in breakfast_products:
            sales_df.loc[day, product] = np.round(sales_df.loc[day, product] * 2).astype(int)

# 3. Im Sommer mehr Cola (200 % mehr)
for i, day in enumerate(days):
    if sales_df.iloc[i]['Jahreszeit'] == 'Sommer':
        other_sales_df.loc[day, 'Cola'] = np.round(other_sales_df.loc[day, 'Cola'] * 2.0).astype(int)

# 4. Bei Events fast kein Salat und nur Burger (Salat um 95 % reduzieren, Burger um 90 % steigern)
for i, day in enumerate(days):
    if sales_df.iloc[i]['Event']:
        other_sales_df.loc[day, 'Salat'] = np.round(other_sales_df.loc[day, 'Salat'] * 0.05).astype(int)
        other_sales_df.loc[day, 'Burger'] = np.round(other_sales_df.loc[day, 'Burger'] * 1.9).astype(int)

# 5. Am Wochenende viel mehr Frühstück als sonst (90 % mehr)
for i, day in enumerate(days):
    weekday = sales_df.iloc[i]['Wochentag']
    if weekday in ['Saturday', 'Sunday']:
        for product in breakfast_products:
            sales_df.loc[day, product] = np.round(sales_df.loc[day, product] * 1.9).astype(int)

# Aufteilung der Verkaufszahlen in Tageszeiten (Frühstück, Mittagessen, Abendessen)
time_slots = ['Fruehstueck', 'Mittagessen', 'Abendessen']

# Frühstücksprodukte werden nur während des Frühstücks konsumiert (100% der Verkäufe im Frühstücks-Slot)
for product in breakfast_products:
    sales_df[f'{product}_Fruehstueck'] = sales_df[product].astype(int)

# Andere Produkte werden für Mittagessen und Abendessen aufgeteilt (z.B. 40% Mittagessen, 60% Abendessen)
for product in other_products:
    other_sales_df[f'{product}_Mittagessen'] = np.round(other_sales_df[product] * 0.4).astype(int)
    other_sales_df[f'{product}_Abendessen'] = np.round(other_sales_df[product] * 0.6).astype(int)

# Lösche die ursprünglichen Summenspalten
sales_df.drop(columns=breakfast_products, inplace=True)
other_sales_df.drop(columns=other_products, inplace=True)

# Füge die anderen Produkte in den Haupt-DataFrame hinzu
sales_df = pd.concat([sales_df, other_sales_df], axis=1)
sales_df.rename(columns={'Datum': 'Date'}, inplace=True)

# Daten als CSV-Datei speichern mit UTF-8-Encoding
sales_df.to_csv('sales_data_with_feiertage_v7.csv', encoding='utf-8', index=True)

print("Daten erfolgreich generiert und in 'sales_data_with_feiertage_v7.csv' gespeichert.")
