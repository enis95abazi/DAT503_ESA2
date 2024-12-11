import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from tkinter import simpledialog
import tkinter as tk
import os
import numpy as np

# Pfad zur CSV-Datei korrigieren
file_path = os.path.join(os.getcwd(), 'sales_data_with_feiertage_v7.csv')

# CSV-Daten importieren
sales_df = pd.read_csv(file_path, parse_dates=['Date'])

# Feiertage in Österreich 2024
feiertage_2024 = pd.DataFrame({
    'holiday': ['Neujahr', 'Heilige Drei Könige', 'Ostersonntag', 'Ostermontag', 'Staatsfeiertag',
                'Christi Himmelfahrt', 'Pfingstsonntag', 'Pfingstmontag', 'Fronleichnam',
                'Mariä Himmelfahrt', 'Nationalfeiertag', 'Allerheiligen', 'Mariä Empfängnis', 'Weihnachten', 'Stefanitag'],
    'ds': pd.to_datetime([
        '2024-01-01', '2024-01-06', '2024-03-31', '2024-04-01', '2024-05-01',
        '2024-05-09', '2024-05-19', '2024-05-20', '2024-05-30',
        '2024-08-15', '2024-10-26', '2024-11-01', '2024-12-08', '2024-12-25', '2024-12-26'
    ]),
    'lower_window': 0,
    'upper_window': 1,
})

# Popup-Eingabefenster für den zukünftigen Zeitraum
root = tk.Tk()
root.withdraw()  # Verstecke das Hauptfenster

start_date = simpledialog.askstring('Startdatum', 'Geben Sie das Startdatum für die Prognose ein (YYYY-MM-DD):')
end_date = simpledialog.askstring('Enddatum', 'Geben Sie das Enddatum für die Prognose ein (YYYY-MM-DD):')

if not start_date or not end_date:
    raise ValueError('Start- und Enddatum müssen angegeben werden.')

try:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
except ValueError:
    raise ValueError('Ungültiges Datumsformat. Bitte geben Sie das Datum im Format YYYY-MM-DD ein.')

if start_date >= end_date:
    raise ValueError('Das Enddatum muss nach dem Startdatum liegen.')

# Liste der Produktspalten, unterteilt nach Tageszeit
products = [
    'Croissant_Fruehstueck', 'Burger_Mittagessen', 'Burger_Abendessen',
    'Salat_Mittagessen', 'Salat_Abendessen', 'Cola_Mittagessen', 'Cola_Abendessen'
]

# Kodierung der Wochenend-Spalte
sales_df['Wochenende'] = sales_df['Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

# Dummy-Kodierung des Wetters
wetter_dummies = pd.get_dummies(sales_df['Wetter'], prefix='Wetter')
sales_df = pd.concat([sales_df, wetter_dummies], axis=1)

# Berechnung der Wetterwahrscheinlichkeiten basierend auf historischen Daten
wetter_probs = sales_df[wetter_dummies.columns].mean()

# Konvertiere die Feiertagsspalte in numerische Werte, falls vorhanden
if 'Feiertag' in sales_df.columns:
    sales_df['Feiertag'] = sales_df['Feiertag'].apply(lambda x: 1 if x else 0)

# Konvertiere die Event-Spalte in numerische Werte, falls vorhanden
if 'Event' in sales_df.columns:
    sales_df['Event'] = sales_df['Event'].apply(lambda x: 1 if x else 0)

# Berechnung der Event-Wahrscheinlichkeit basierend auf historischen Daten
event_prob = sales_df['Event'].mean()

# Dummy-Kodierung der Jahreszeiten
season_dummies = pd.get_dummies(sales_df['Jahreszeit'], prefix='Jahreszeit')
sales_df = pd.concat([sales_df, season_dummies], axis=1)

# Durchlaufe jede Produktsparte und erstelle ein Prophet-Modell für jede
for product in products:
    # Erstelle eine neue DataFrame für das jeweilige Produkt
    columns = ['Date', product, 'Wochenende', 'Feiertag', 'Event'] + list(wetter_dummies.columns) + list(season_dummies.columns)
    product_df = sales_df[columns].rename(columns={'Date': 'ds', product: 'y'})
    
    # Initialisiere und trainiere das Prophet-Modell mit Feiertagen und jährlicher Saisonalität
    model = Prophet(holidays=feiertage_2024, yearly_seasonality=False, weekly_seasonality=False)
    
    # Füge spezifische jährliche und wöchentliche Saisonalitäten mit Fourier-Terms hinzu
    model.add_seasonality(name='yearly', period=365.25, fourier_order=10)
    model.add_seasonality(name='weekly', period=7, fourier_order=5)
    
    # Füge die Regressoren hinzu
    model.add_regressor('Wochenende')
    model.add_regressor('Feiertag')
    model.add_regressor('Event')
    for wetter in wetter_dummies.columns:
        model.add_regressor(wetter)
    for season in season_dummies.columns:
        model.add_regressor(season)
    
    # Trainiere das Modell
    model.fit(product_df)
    
    # Zukunftsdaten basierend auf dem angegebenen Zeitraum erstellen
    future = pd.date_range(start=start_date, end=end_date, freq='D').to_frame(index=False, name='ds')
    future = future[future['ds'] > sales_df['Date'].max()]
    
    # Füge die Regressoren zu den zukünftigen Daten hinzu
    future['Wochenende'] = future['ds'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    future['Feiertag'] = future['ds'].isin(feiertage_2024['ds']).astype(int)
    future['Event'] = np.random.binomial(1, event_prob, size=len(future))  # Simuliere zukünftige Events basierend auf historischen Daten
    for wetter in wetter_dummies.columns:
        future[wetter] = np.random.binomial(1, wetter_probs[wetter], size=len(future))
    for season in season_dummies.columns:
        # Berechne die Jahreszeit basierend auf dem Monat (Winter: 12, 1, 2; Fruehling: 3, 4, 5; Sommer: 6, 7, 8; Herbst: 9, 10, 11)
        future[season] = future['ds'].dt.month.map({12: 'Jahreszeit_Winter', 1: 'Jahreszeit_Winter', 2: 'Jahreszeit_Winter',
                                                    3: 'Jahreszeit_Fruehling', 4: 'Jahreszeit_Fruehling', 5: 'Jahreszeit_Fruehling',
                                                    6: 'Jahreszeit_Sommer', 7: 'Jahreszeit_Sommer', 8: 'Jahreszeit_Sommer',
                                                    9: 'Jahreszeit_Herbst', 10: 'Jahreszeit_Herbst', 11: 'Jahreszeit_Herbst'}).apply(lambda x: 1 if x == season else 0)
    
    # Überprüfen, ob Zukunftsdaten vorhanden sind
    if future.empty:
        print(f"Keine zukünftigen Daten für {product} verfügbar.")
        continue
    
    # Prognose erstellen
    forecast = model.predict(future)
    
    # Ergebnisse anzeigen
    print(f"Prognose für {product}:")
    forecast_rounded = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_rounded['ds'] = forecast_rounded['ds'].dt.strftime('%Y-%m-%d')
    forecast_rounded[['yhat', 'yhat_lower', 'yhat_upper']] = forecast_rounded[['yhat', 'yhat_lower', 'yhat_upper']].round(0).astype(int)
    print(forecast_rounded.tail())
    
    # Speichern der Prognose als CSV-Datei
    forecast_file_name = f"forecast_{product}.csv"
    forecast_rounded.to_csv(forecast_file_name, index=False)
    print(f"Prognose für {product} wurde in '{forecast_file_name}' gespeichert.")
    
    # Plot der Prognose
    plt.figure()
    plt.plot(forecast['ds'], forecast['yhat'], label='Vorhersage')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='skyblue', alpha=0.4)
    plt.xlim([forecast['ds'].min(), forecast['ds'].max()])  # Nur die zukünftigen Daten anzeigen
    plt.xlabel('Datum')
    plt.ylabel('Verkaufszahlen')
    plt.legend()
    plt.grid()
    plt.title(f"Prognose für {product} (nur Zukunft)")
    plt.savefig(f"forecast_plot_{product}.png")
    plt.show()
