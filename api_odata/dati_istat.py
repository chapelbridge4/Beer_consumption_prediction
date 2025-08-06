import pandas as pd

# Caricare il file ogni volta
consumption_file_ita = 'file_csv/dati_ita.csv'
consumption_file_ita_0 = '../file_csv/dati_ita.csv'

# Carica il file CSV in un DataFrame
df_cons_ita = pd.read_csv(consumption_file_ita, sep='|', encoding='utf-8', engine='python')

# Rinomina le colonne nel DataFrame
df_cons_ita = df_cons_ita.rename(columns= lambda x: x.replace(' ', '_'))
df_cons_ita = df_cons_ita.rename(columns= lambda x: x.replace('TIPO_DATO_AVQ', 'tipo_bevanda'))

# Filtra le righe che contengono 'BIRRA' nella colonna 'tipo_bevanda'
df_cons_ita = df_cons_ita[df_cons_ita['tipo_bevanda'].str.contains('BIRRA')]

# Crea una copia del DataFrame originale
df_ita = df_cons_ita

# Rimuovi il prefisso '11_' dalla colonna 'tipo_bevanda'
df_ita['tipo_bevanda'] = df_ita['tipo_bevanda'].str.replace('11_', '')
df_ita['Classe_di_età'] = df_ita['Classe_di_età'].str.replace(' anni', '')
df_ita['Classe_di_età'] = df_ita['Classe_di_età'].str.replace(' e più', '')


# Elimina alcune colonne non necessarie
df_ita = df_ita.drop(columns=['Flag_Codes', 'Flags','Seleziona_periodo'])

# Mapping dei nomi delle colonne nel DataFrame rispetto a quelli della tabella SQL
column_mapping = {
    'SEXISTAT1': 'sexstat',
    'ETA1': 'eta',
    'TIME': 'anno'
}

# Rinomina le colonne nel DataFrame utilizzando il mapping
df_ita.rename(columns=column_mapping, inplace=True)

# Stampa informazioni sul DataFrame
df_ita.info()

# Converte i nomi delle colonne in minuscolo
df_ita.columns = [col.lower() for col in df_ita.columns]

# Ordina il DataFrame in base a 'tipo_bevanda' e 'anno'
df_ita.sort_values( by=['tipo_bevanda', 'anno'],  inplace=True)

# Stampa i nomi delle colonne e il DataFrame finale
print(df_ita.columns)
# print(df_ita['classe_di_età'])
