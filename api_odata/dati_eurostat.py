import pandas as pd

# Sold production, exports and imports [DS-056120__custom_3278652]
# prendere impqnt, expqnt, prodqnt, impval, expval, prodval dal file excel

# il prodotto è in litri

# Percorso del tuo file Excel
excel_file_path = "../file_csv/ds-056120__custom_3278652_spreadsheet.xlsx"
excel_file_path_0 = "file_csv/ds-056120__custom_3278652_spreadsheet.xlsx"

# Carica il file Excel in un DataFrame
xl = pd.ExcelFile(excel_file_path_0)

# Ottieni una lista dei nomi dei fogli nel file Excel
sheet_names = xl.sheet_names
print("Nomi dei fogli nel file Excel:", sheet_names)

# Mappatura tra le descrizioni e i nomi delle locations nel DataFrame
mapping = {
    'Netherlands': 'NETHERLANDS (KINGDOM OF THE)',
    'United Kingdom' : 'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND'
}

def process_sheet(excel_file_path_0, sheet_name, columns_name):
    # Leggi il foglio Excel
    df = pd.read_excel(excel_file_path_0, sheet_name=sheet_name, header=None, names=columns, skiprows=9)
    df = df.drop(0)
    df = df[df['location'] != 'EU27TOTALS_2020']
    df = df[df['location'] != 'Bosnia and Herzegovina']
    df = df[df['location'] != 'Cyprus']
    df = df[df['location'] != 'Special value']

    # Applica la mappatura al DataFrame
    df['location'] = df['location'].replace(mapping)
    df['location'] = df['location'].str.upper()

    # Seleziona colonne degli anni e non degli anni
    anni_columns = df.columns[1:]
    non_anni_columns = df.columns[:1]

    # Trasforma i valori delle colonne degli anni in numerici
    df[anni_columns] = df[anni_columns].apply(pd.to_numeric, errors='coerce')

    # Ristruttura il DataFrame
    df_trat = df.melt(id_vars=['location'], var_name='year', value_name=columns_name)
    df_trat.sort_values(by=['location', 'year'], inplace=True)

    # Elimina le righe indesiderate
    df_masked = df_trat.drop(df_trat[(df_trat['location'] == ':') | (df_trat['location'].isnull()) | (
                df_trat['location'] == 'Special value')].index)

    return df_masked


# Definisci i nomi delle colonne degli anni e del foglio Excel
columns = ['location', 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

sheet_names = {
    "Sheet 7": "qta_prod",
    "Sheet 5": "val_prod",
    "Sheet 4": "qta_exp",
    "Sheet 6": "val_exp",
    "Sheet 9": "qta_imp",
    "Sheet 1": "val_imp"
}

dfs_masked = {}

# Itera attraverso ogni foglio del file Excel e il rispettivo nome della colonna
for sheet_name, col_name in sheet_names.items():
    df_masked = process_sheet(excel_file_path_0, sheet_name, col_name)
    dfs_masked[sheet_name] = df_masked
    print(f"Dati elaborati per {col_name}:")
    print(df_masked)

# Concatena tutti i DataFrame in uno unico
df_combined = pd.concat(dfs_masked.values(), axis=1)

# Rimuove eventuali colonne duplicate
df_prod_exp_imp = df_combined.loc[:, ~df_combined.columns.duplicated()]

# Visualizza il DataFrame combinato
print("DataFrame Sold production, exports and imports:")
print(df_prod_exp_imp)


# unique_locations = [
#     'AUSTRIA', 'BELGIUM', 'BOSNIA AND HERZEGOVINA', 'BULGARIA', 'CROATIA',
#     'CYPRUS', 'CZECHIA', 'DENMARK', 'EU27TOTALS_2020', 'ESTONIA', 'FINLAND',
#     'FRANCE', 'GERMANY', 'GREECE', 'HUNGARY', 'ICELAND', 'IRELAND', 'ITALY',
#     'LATVIA', 'LITHUANIA', 'LUXEMBOURG', 'MALTA', 'MONTENEGRO', 'NETHERLANDS',
#     'NORTH MACEDONIA', 'NORWAY', 'POLAND', 'PORTUGAL', 'ROMANIA', 'SERBIA',
#     'SLOVAKIA', 'SLOVENIA', 'SPAIN', 'SWEDEN', 'UNITED KINGDOM'
# ]
#
# descrizioni = ['UNITED ARAB EMIRATES', 'ALBANIA', 'ARGENTINA', 'AUSTRIA', 'AUSTRALIA', 'BELGIUM', 'BULGARIA', 'BAHRAIN', 'BERMUDA', 'BRAZIL', 'CANADA', 'SWITZERLAND', 'CHILE', 'CHINA',  'COSTA RICA',
#                 'CAPE VERDE', 'CZECHIA', 'GERMANY', 'DENMARK', 'ESTONIA',  'GREECE', 'SPAIN', 'FINLAND', 'FRANCE', 'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND', 'HONG KONG', 'CROATIA',
#                 'HUNGARY', 'IRELAND', 'ISRAEL', 'ICELAND', 'ITALY', 'JAPAN', 'CAMBODIA', 'REPUBLIC OF KOREA', 'LITHUANIA', 'LUXEMBOURG', 'LATVIA', 'MALDIVES', 'MEXICO', 'MALAYSIA', 'NETHERLANDS (KINGDOM OF THE)',
#                 'NORWAY', 'NEW ZEALAND', 'PANAMA', 'PHILIPPINES', 'POLAND', 'PORTUGAL', 'PARAGUAY', 'ROMANIA', 'RUSSIAN FEDERATION', 'SEYCHELLES', 'SWEDEN', 'SINGAPORE', 'REPUBLIC OF SAN MARINO', 'SINT MAARTEN DUTCH PART',
#                'THAILANDIA', 'TRINIDAD AND TOBAGO', 'TAIWAN', 'UKRAINE', 'UNITED STATES OF AMERICA', 'URUGUAY', 'SOUTH AFRICA']