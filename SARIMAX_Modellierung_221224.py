import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import numpy as np

# Daten laden
file_path = "sales_data_with_feiertage_v6.csv"
data = pd.read_csv(file_path, delimiter=";")

# Datum als Index setzen und in datetime-Format konvertieren
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
data.set_index('Date', inplace=True)

# Zielwert auswählen (z. B. Burger_Mittagessen)
time_series = data['Burger_Mittagessen']

# Stationarität prüfen
result = adfuller(time_series)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] <= 0.05:
    print("Zeitreihe ist stationär.")
else:
    print("Zeitreihe ist nicht stationär.")

# SARIMA-Modell für Januar: Feinplanung
model_january = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7))
model_january_fit = model_january.fit(disp=False)

# Prognose für Januar 2024 (31 Tage)
forecast_january_steps = 31
forecast_january = model_january_fit.get_forecast(steps=forecast_january_steps)
forecast_january_ci = forecast_january.conf_int(alpha=0.05)

forecast_january_dates = pd.date_range(start='2024-01-01', periods=forecast_january_steps, freq='D')
forecast_january_df = pd.DataFrame({
    'Date': forecast_january_dates,
    'Predicted_Burger_Mittagessen': forecast_january.predicted_mean,
    'Lower_CI': forecast_january_ci.iloc[:, 0].values,
    'Upper_CI': forecast_january_ci.iloc[:, 1].values
})
forecast_january_df.set_index('Date', inplace=True)

# SARIMA-Modell für Februar: Grobplanung
model_february = SARIMAX(time_series, order=(1, 1, 0), seasonal_order=(0, 1, 1, 30))
model_february_fit = model_february.fit(disp=False)

# Prognose für Februar 2024 (28 Tage)
forecast_february_steps = 28
forecast_february = model_february_fit.get_forecast(steps=forecast_february_steps)
forecast_february_ci = forecast_february.conf_int(alpha=0.05)

forecast_february_dates = pd.date_range(start='2024-02-01', periods=forecast_february_steps, freq='D')
forecast_february_df = pd.DataFrame({
    'Date': forecast_february_dates,
    'Predicted_Burger_Mittagessen': forecast_february.predicted_mean,
    'Lower_CI': forecast_february_ci.iloc[:, 0].values,
    'Upper_CI': forecast_february_ci.iloc[:, 1].values
})
forecast_february_df.set_index('Date', inplace=True)

# Prognosen kombinieren
forecast_combined = pd.concat([forecast_january_df, forecast_february_df])

# Visualisierung
plt.figure(figsize=(14, 7))
plt.plot(time_series, label='Original Data', color='blue')
plt.plot(forecast_january_df.index, forecast_january_df['Predicted_Burger_Mittagessen'], label='Forecast Januar 2024', color='red')
plt.fill_between(forecast_january_df.index,
                 forecast_january_df['Lower_CI'],
                 forecast_january_df['Upper_CI'],
                 color='pink', alpha=0.3, label='Confidence Interval (Januar)')

plt.plot(forecast_february_df.index, forecast_february_df['Predicted_Burger_Mittagessen'], label='Forecast Februar 2024', color='green')
plt.fill_between(forecast_february_df.index,
                 forecast_february_df['Lower_CI'],
                 forecast_february_df['Upper_CI'],
                 color='lightgreen', alpha=0.3, label='Confidence Interval (Februar)')

# Vertikale Linie für Startdatum der Prognose
plt.axvline(x=pd.Timestamp('2024-01-01'), color='grey', linestyle='--', label='Start Prognose')

plt.title('Optimiertes SARIMA-Modell: Fein- und Grobplanung (2024)')
plt.xlabel('Datum')
plt.ylabel('Verkäufe')
plt.legend()
plt.grid()
plt.show()

# Prognosen anzeigen
print("Prognose Januar 2024:")
print(forecast_january_df)
print("\nPrognose Februar 2024:")
print(forecast_february_df)
