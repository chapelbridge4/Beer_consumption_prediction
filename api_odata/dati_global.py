import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from api_odata.dati_nazioni import descrizioni
# import numpy as np
# from sklearn.impute import SimpleImputer

# Caricare il file ogni volta
consumption_file = 'file_csv/dati_global.csv'
consumption_file_0 = '../file_csv/dati_global.csv'

# Pulizia dati dentro il df
df_cons = pd.read_csv(consumption_file, sep=',"', encoding='utf-8', skiprows=1, engine='python')
df_cons = df_cons.rename(columns= lambda x: x.replace('"', ''))
df_cons = df_cons.rename(columns= lambda x: x.replace(' ', '_'))
df_cons = df_cons.rename(columns= lambda x: x.replace('_2', '2'))
df_cons = df_cons.rename(columns= lambda x: x.replace(',_', '_'))
df_cons = df_cons.rename(columns= lambda x: x.replace('""', ''))
df_cons = df_cons.replace('"', '', regex=True)

anni_columns = df_cons.columns[3:]  # Seleziona solo le colonne degli anni
non_anni_columns = df_cons.columns[:3]  # Seleziona tutte le colonne tranne quelle degli anni
df_cons[anni_columns] = df_cons[anni_columns].apply(pd.to_numeric, errors='coerce')

df_consumi = df_cons[non_anni_columns.tolist() + anni_columns.tolist()[::-1]]
# df_consumi = df_consumi[df_consumi['2019'] != 0]
df_consumi = df_consumi[df_consumi['Beverage_Types'].str.contains('Beer')]

# Imputo i valori mancanti utilizzando la mediana degli stessi
imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')

# Itera attraverso ogni colonna del DataFrame df_consumi
for column in df_consumi:
    # Controlla se ci sono valori nulli nella colonna corrente
    if df_consumi[column].isnull().values.any():
        # Se ci sono valori nulli, crea un array numpy dalla colonna corrente
        tmp = df_consumi[column].to_numpy()

        # Utilizza un oggetto imp_mean per sostituire i valori nulli con il valore medio della colonna
        df_consumi[column] = imp_mean.fit_transform(tmp)

# Per fare un check veloce
df_consumi.info()

print(df_consumi)

# Ristruttura il dataframe in modo che ogni riga rappresenti un anno specifico per un paese specifico
df_melt = df_consumi.melt(id_vars=['Countries_territories_and_areas', 'Data_Source', 'Beverage_Types'],
                  var_name='Year', value_name='Alcohol_Consumption')

# Converte la colonna 'Year' in un intero
df_melt['Year'] = df_melt['Year'].astype(int)
df_melt['Countries_territories_and_areas'] = df_melt['Countries_territories_and_areas'].str.upper()
df_melt.sort_values( by=['Countries_territories_and_areas','Year'],  inplace=True)

print(df_melt)

# Filtra il DataFrame df_country mantenendo solo i nomi presenti in descrizioni
df_melt= df_melt[df_melt['Countries_territories_and_areas'].isin(descrizioni)]

# Converte i nomi delle colonne in minuscolo
df_melt.columns = [col.lower() for col in df_melt.columns]


print(df_melt)
