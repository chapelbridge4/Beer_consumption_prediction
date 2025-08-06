import pandas as pd

# Caricare il file ogni volta
consumption_file_ita = 'file_csv/dati_istat_regioni.csv'
consumption_file_ita_0 = '../file_csv/dati_istat_regioni.csv'

# Carica il file CSV in un DataFrame
df_cons_ita = pd.read_csv(consumption_file_ita, sep=';', encoding='utf-8', engine='python')

# print(df_cons_ita.columns)
# print(df_cons_ita)

# Rinomina le colonne nel DataFrame
df_cons_ita = df_cons_ita.rename(columns= lambda x: x.replace(' ', '_'))
df_cons_ita = df_cons_ita.rename(columns= lambda x: x.replace('TIPO_DATO_AVQ', 'tipo_bevanda'))
df_cons_ita = df_cons_ita.drop(columns=['ITTER107'])
# print(df_cons_ita)

# Rimuovi il prefisso '11_' dalla colonna 'tipo_bevanda'
df_cons_ita['tipo_bevanda'] = df_cons_ita['tipo_bevanda'].str.replace('11_', '')
# Filtra le righe che contengono 'BIRRA' nella colonna 'tipo_bevanda'
df_cons_ita = df_cons_ita[df_cons_ita['tipo_bevanda'].str.contains('BIRRA')]
# Elimina alcune colonne non necessarie
df_ita = df_cons_ita.drop(columns=['Flag_Codes', 'Flags','Seleziona_periodo'])
# print(df_ita)

# Mapping dei nomi delle colonne nel DataFrame rispetto a quelli della tabella SQL
column_mapping = {
    'TIME': 'anno'
}

# Rinomina le colonne nel DataFrame utilizzando il mapping
df_ita.rename(columns=column_mapping, inplace=True)
# print(df_ita)

# Converte i nomi delle colonne in minuscolo
df_ita.columns = [col.lower() for col in df_ita.columns]
# print(df_ita)

# Ordina il DataFrame in base a 'tipo_bevanda' e 'anno'
df_ita.sort_values( by=['tipo_bevanda', 'anno'],  inplace=True)
df_ita_regioni = df_ita

print(df_ita_regioni)
print(df_ita_regioni.columns)
