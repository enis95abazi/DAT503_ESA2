# DAT503_ESA2
# SARIMA-Modellierung zur Fein- und Grobplanung

## Überblick
Dieses Projekt verwendet ein SARIMA-Modell zur Vorhersage der Verkaufszahlen eines Fast-Food-Restaurants. Es teilt die Prognose in zwei Abschnitte:

- **Feinplanung (Jänner 2024)**
- **Grobplanung (Februar 2024)**

## Hauptmerkmale
- **Datenanalyse**: Verwendet historische Verkaufsdaten aus `sales_data_with_feiertage_v6.csv`.
- **Modellierung**: SARIMA-Modelle für zwei Planungszeiträume.
- **Visualisierung**: Darstellung von Originaldaten, Prognosen und Konfidenzintervallen.

## Genutzte Technologien
- **Pandas**: Datenmanagement und -verarbeitung.
- **Matplotlib**: Visualisierung der Zeitreihen und Prognosen.
- **Statsmodels**: SARIMA-Modellierung und statistische Tests.

## Dateien
- **`SARIMAX_Modellierung_221224.py`**: Hauptskript zur Durchführung der Analyse und Prognose.
- **`sales_data_with_feiertage_v6.csv`**: Datensatz mit historischen Verkaufsdaten.

## Nutzung
1. **Vorbereitung**: alle Pakete installieren:
   ```bash
   pip install pandas matplotlib statsmodels
   ```
2. **Ausführung**: Führe das Skript aus:
   ```bash
   python SARIMAX_Modellierung_221224.py
   ```
3. **Ergebnisse**: Die Visualisierung zeigt die Verkaufsprognosen für die beiden Monate Jänner und Februar. 

## Hinweis
Das Projekt unterscheidet explizit zwischen Fein- und Grobplanung, da in der Praxis mit verderblichen Materialen ebenfalls kurzfristig genauer geplant werden muss. 
