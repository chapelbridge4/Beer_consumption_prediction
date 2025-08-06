import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sql_server import connessione_postgres_p
from sql_server.connessione_postgres_p import filepath_0, filepath
from prediction import funzioni
from sqlalchemy import create_engine
from sklearn.preprocessing import OneHotEncoder

# pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

# Esplorazione grafica
import seaborn as sns
import matplotlib.pyplot as plt

def test_model(x, y):
    """
    Questa funzione testa il modello utilizzando il set di dati fornito.

    Parameters:
    x (DataFrame): Il DataFrame delle caratteristiche (features).
    y (Series): La Serie del target.

    Returns:
    None
    """
    # [1] split
    x_train, x_test, y_train, y_test = funzioni.train_test_split(x, y, test_size=0.2)

    # [2] pipeline
    pipe = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=poly_degree),
                      Ridge(alpha=ridge_alpha, fit_intercept=ridge_intercept))

    # [3] training
    pipe.fit(x_train, y_train)


    # [3] test
    y_pred = pipe.predict(x_test)


    # [4] error
    mse = funzioni.mean_squared_error(y_test, y_pred)
    r2 = funzioni.r2_score(y_test, y_pred)

    scores_df = pd.DataFrame({'rmse': [np.sqrt(mse)], 'r2': [r2]})
    print(scores_df)
    print("\n")

    funzioni.test_plot(y_test, y_pred)


# Query per selezionare tutti i dati relativi ai consumi dalla tabella 'dati_predizione'
query_dati_consumo = """
SELECT distinct
    anno_oper as anno,
    cod_naz, 
    country as nazione_area,
    consumo_litri_alcohol_puro as consumo_alcohol
from 
    dati_predizione
where
    anno_oper >= '2010' 
"""

try:
    # Creazione del motore per la connessione al database PostgreSQL
    engine_pos = create_engine(connessione_postgres_p.create_connection_string(filepath))

    # Utilizza un contesto with per garantire la chiusura della connessione
    with engine_pos.connect() as conn_pos:
        if conn_pos is None:
            print("Connessione a PostgreSQL non stabilita.")
        else:
            # Esegui la query per ottenere i dati delle consumi dall'azienda dal database
            df_consumi_alcohol = pd.read_sql(query_dati_consumo, conn_pos)

except Exception as e:
    # Gestione delle eccezioni nel caso si verifichi un errore durante la connessione o l'esecuzione della query
    print(f"Si è verificato un errore: {e}")

else:
    # Stampare le colonne solo se il blocco try è stato eseguito con successo
    print(df_consumi_alcohol.columns)

print(df_consumi_alcohol)

def consumo_fillna(df):

    """
    Questa funzione prende in input un DataFrame contenente dati relativi al consumo di alcol per diversi paesi
    in diversi anni. Gestisce i valori mancanti nella colonna 'consumo_alcohol' prevedendo tali valori mancanti
    utilizzando un modello di regressione lineare per ciascun paese. I passaggi principali includono la conversione
    della colonna 'anno' in formato numerico, l'esclusione di specifici Paesi dal DataFrame, il controllo dei valori
    nulli, la divisione dei dati noti e mancanti, l'addestramento del modello di regressione lineare e la previsione
    dei valori mancanti per ciascun paese. Infine, i valori previsti vengono aggiunti al DataFrame originale.

    Args:
    - df (DataFrame): Il DataFrame contenente i dati relativi al consumo di alcol per diversi paesi e anni.

    Returns:
    - DataFrame: Il DataFrame aggiornato con i valori mancanti nella colonna 'consumo_alcohol' previsti e aggiunti.
    """

    df['anno'] = pd.to_numeric(df['anno'], errors='coerce')

    # Escludi il Paese della Polonia dal DataFrame
    df = df[df['nazione_area'] != 'POLAND']

    # Controllo dei valori nulli per df_consumi_selezionato
    rows_with_nan = df[df.isnull().any(axis=1)]
    print('righe con valori nan: ', rows_with_nan)

    # Filtra le righe con valori noti e valori mancanti
    df_known = df.dropna(subset=['consumo_alcohol'])
    df_unknown = df[df['consumo_alcohol'].isna()]

    for nazione_area in df['nazione_area'].unique():  # Itera su tutti i paesi presenti nel DataFrame

        # Filtra il DataFrame per il paese corrente
        df_nazione_area_known = df_known[df_known['nazione_area'] == nazione_area]
        df_nazione_area_unknown = df_unknown[df_unknown['nazione_area'] == nazione_area]

        # Se non ci sono dati noti per questo paese, passa al prossimo
        if df_nazione_area_known.empty or df_nazione_area_unknown.empty:
            continue

        # Addestra il modello di regressione lineare per il paese corrente
        model = LinearRegression()
        X_train = df_nazione_area_known[['anno']]
        y_train = df_nazione_area_known['consumo_alcohol']

        # Assicura che ci siano dati noti nella variabile target
        if y_train.isna().any():
            continue

        model.fit(X_train, y_train)

        # Prevedi i valori mancanti per il paese corrente
        X_pred = df_nazione_area_unknown[['anno']]
        predicted_consumption = model.predict(X_pred)

        # Aggiungi i valori previsti al DataFrame originale per il paese corrente
        df.loc[(df['nazione_area'] == nazione_area) & df[
            'consumo_alcohol'].isna(), 'consumo_alcohol'] = predicted_consumption

    return df

# Chiamata alla funzione per il DataFrame delle consumi dell'azienda
df_consumi_alcohol = consumo_fillna(df_consumi_alcohol)

# Controllo dei valori nulli per df_consumi_selezionato
rows_with_nan = df_consumi_alcohol[df_consumi_alcohol.isnull().any(axis=1)]
print('righe con valori nan: ', rows_with_nan)

# Controllo valori unici nella colonna nazione_area
print(rows_with_nan['nazione_area'].unique())

# Inizializzazione di un dizionario che associa valori di consumo stimati ai paesi senza dati
nazione_area_nan = rows_with_nan['nazione_area'].unique()
valori_cons_nazione_area_nan = {
    'CAMBODIA' : 2,
    'UKRAINE': 3,
    'SEYCHELLES' : 1.5,
    'ALBANIA' : 2.5,
    'PANAMA' : 3,
    'LATVIA' : 4,
    'URUGUAY' : 2.5,
    'LUXEMBOURG' : 5
}

# Ciclo attraverso i paesi senza dati di consumo per assegnare i valori stimati
for nazione_area in nazione_area_nan:
    consumo = valori_cons_nazione_area_nan.get(nazione_area)
    # Aggiornamento del DataFrame dei consumi selezionate con i valori di consumo stimati per il paese corrente
    df_consumi_alcohol.loc[(df_consumi_alcohol['nazione_area'] == nazione_area), 'consumo_alcohol'] = consumo

# Controllo per verificare se sono stati inseriti correttamente i valori di consumo per i paesi senza dati
controllo_record_no_cons = df_consumi_alcohol.loc[df_consumi_alcohol['nazione_area'].isin(['PANAMA', 'UKRAINE', 'CAMBODIA', 'ALBANIA', 'LATVIA', 'URUGUAY', 'SEYCHELLES'])]
print('SUPER CONTROLLO ', controllo_record_no_cons)

# Controllo dei valori nulli per df_consumi_selezionato
rows_with_nan_1 = df_consumi_alcohol[df_consumi_alcohol.isnull().any(axis=1)]
print('righe con valori nan dopo aver aggiunto i consumi: ', rows_with_nan_1)

print(df_consumi_alcohol.sort_values('anno', ascending=True).to_string())

# Genera istogramma per il DataFrame df_consumi_selezionato
df_consumi_alcohol.hist(figsize=(20, 15))
plt.show()

df_consumi_alcohol_target = df_consumi_alcohol['consumo_alcohol']

# Forma della distribuzione consumi
sns.displot(df_consumi_alcohol_target)
plt.title('Distribuzione consumi', fontsize=16)  # Aggiungo il titolo
plt.show()
print(df_consumi_alcohol_target.describe())
data_alc = df_consumi_alcohol_target  # Accedi alla colonna specifica
# Calcola i limiti superiore e inferiore dello scarto interquartile per i dati della colonna 'consumo_alcohol'
up_limit, low_limit = funzioni.interquartile_range_limits(data_alc)
print(up_limit, low_limit)
# Calcola i limiti dello scarto interquartile per i dati di quella colonna
print(f"Creazione del grafico per la colonna di df_consumi_alcohol: {'consumo_alcohol'}")
# Crea un grafico a dispersione per i dati di quella colonna, con i limiti dello scarto interquartile
funzioni.plot_generic_distribution(data_alc, up_limit, low_limit)
# Sostituisci tutti i valori nell'array tmp_alc che sono superiori al limite superiore o inferiori al limite inferiore con il valore del limite corrispondente
tmp_alc = data_alc.to_numpy()
# Conta quanti valori superano il limite superiore (outliers)
tmp_try = tmp_alc[tmp_alc > up_limit]
tmpContato = tmp_try.shape[0]
print('Valori che superano up_limit: ', tmpContato)
# Sostituisci gli outliers con i limiti superiore e inferiore
tmp_alc[tmp_alc > up_limit]  = up_limit
tmp_alc[tmp_alc < low_limit] = low_limit
# Visualizza l'istogramma dei dati dopo la rimozione degli outliers
sns.displot(tmp_alc)
plt.show()

# Seleziona solo le colonne numeriche dal DataFrame dei consumi
df_numerico_consumi_alcohol = df_consumi_alcohol.select_dtypes(include='number')
print(df_numerico_consumi_alcohol)

corr_matrix_consumi_alcohol = df_numerico_consumi_alcohol.corr()

# Aggiungi un titolo per la visualizzazione di consumo_alcohol
print("Correlazione delle colonne rispetto a consumo")
# plot consumi
plt.figure(figsize=(16, 16))
sns.heatmap(corr_matrix_consumi_alcohol, annot=True)
plt.title('Matrice di Correlazione', fontsize=16)  # Aggiungo il titolo
plt.show()
# Calcola la correlazione delle colonne rispetto alla colonna 'consumo_alcohol' e ordina in modo decrescente
corr_consumi = corr_matrix_consumi_alcohol['consumo_alcohol'].sort_values(ascending=False)
# Aggiungi un titolo per la visualizzazione dei consumi
print("Correlazione delle colonne rispetto consumi")
# Visualizza la serie di correlazioni
print(corr_consumi)
df_consumi = corr_matrix_consumi_alcohol.copy()  # Copia il DataFrame per non modificare quello originale
print(df_consumi.columns)

# Verifica le colonne disponibili in df_consumi
print('Colonne in df_consumi:', df_consumi.columns)

# Identifica i paesi in modo univoco
unique_countries = df_consumi_alcohol['nazione_area'].unique()

# Codifica i valori usando OneHotEncoder
hot_encoder = OneHotEncoder(sparse_output=False)
hot_encoded = hot_encoder.fit_transform(df_consumi_alcohol['nazione_area'].values.reshape(-1, 1))

# Creazione del nuovo DataFrame con i nomi dei paesi come nomi delle colonne
country_df_encoded = pd.DataFrame(hot_encoded, columns=unique_countries)

# Aggiungi il nuovo DataFrame al DataFrame originale
country_df_encoded = pd.concat([df_consumi_alcohol.reset_index(drop=True), country_df_encoded], axis=1)

print('CATEGORIZZATO consumi mln: ', country_df_encoded)

df_uiu = country_df_encoded.sort_values('anno', ascending=True)

print(df_uiu)

# Seleziona le features X rimuovendo la colonna 'consumo_alcohol' e 'nazione_area'
X = df_uiu.drop(['consumo_alcohol', 'nazione_area'], axis=1)
print(X.columns)
# Seleziona il target Y dalla colonna 'consumo_alcohol'
Y = df_uiu['consumo_alcohol']

# test 1
funzioni.pipeline_validation(make_pipeline(StandardScaler(), LinearRegression()), X, Y)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), LinearRegression()), X, Y)

# test 2
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X, Y)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X, Y)

# Valida un modello di regressione lineare polinomiale con cross-validation k-fold
funzioni.pipeline_poly_validation(LinearRegression(), X, Y, 3)

# Valida un modello di regressione Ridge polinomiale con cross-validation k-fold
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 2), Ridge(alpha=10, fit_intercept=True)), X, Y)
# Plotta la curva di apprendimento per il modello di regressione Ridge polinomiale
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 2), Ridge(alpha=10, fit_intercept=True)), X, Y)

# Utilizza la funzione pipeline_poly_validation per validare un modello di regressione Ridge polinomiale
funzioni.pipeline_poly_validation(Ridge(), X, Y, 4)

# Testa un modello di regressione Ridge
funzioni.test_pipeline_ridge(X, Y, 3, 100)

# Ottimizza i parametri del modello di regressione Ridge
poly_degree, ridge_alpha, ridge_intercept = funzioni.ottimization(X, Y)

# Testa il modello utilizzando i dati di input X e il target Y
test_model(X, Y)

df_consumi_s = df_uiu.drop(columns=unique_countries)

# Controlla se la colonna 'anno' esiste nel DataFrame
if 'anno' in df_consumi_s.columns:
    # Converti la colonna 'anno_oper' in tipo intero
    df_consumi_s['anno'] = df_consumi_s['anno'].astype(int)
    # Filtra le righe in cui 'anno' è minore o uguale al 2019
    df_consumi_s = df_consumi_s[df_consumi_s['anno'] <= 2019]
else:
    print("Colonna 'anno_oper' non trovata nel DataFrame.")


df_consumi_s = df_consumi_s[df_consumi_s['nazione_area'] != 'ALTRI']

def save_model_with_pickle(df):
    """
    Salva il modello addestrato utilizzando pickle.

    Args:
        df (DataFrame): Il DataFrame contenente i dati di addestramento.

    Returns:
        None
    """
    # Creazione delle features e target
    X = df.drop(["consumo_alcohol", "nazione_area"], axis=1)
    Y = df["consumo_alcohol"]

    # Divisione del dataset in train e test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Creazione del modello
    slc = StandardScaler()
    poly = PolynomialFeatures(degree=poly_degree)
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=ridge_intercept)

    model_pipeline = make_pipeline(slc, poly, ridge)

    # Addestramento del modello
    model_pipeline.fit(x_train, y_train)

    # Salvataggio del modello utilizzando pickle
    with open('prediction/model_consumi.pkl', "wb") as file:
        pickle.dump(model_pipeline, file)

# Chiamo la funzione per salvare il modello
# save_model_with_pickle(df_uiu)

def load_model_with_pickle(filename):
    """
    Carica il modello addestrato utilizzando pickle.

    Args:
        filename (str): Il nome del file contenente il modello salvato.

    Returns:
        model: Il modello addestrato.
    """
    with open('prediction/' + filename, 'rb') as f:
        model = pickle.load(f)
    return model

df_test = df_uiu.copy()

# Esempio di caricamento del modello
loaded_model = load_model_with_pickle('model_consumi.pkl')

# Fai previsioni utilizzando il modello caricato
Y_pred = loaded_model.predict(df_uiu.drop(['consumo_alcohol', 'nazione_area'], axis=1))

# print(Y_pred)

# Aggiungi la colonna 'consumo_alcohol_pred' contenente i valori predetti al DataFrame df_uiu
df_uiu['consumo_alcohol_pred'] = Y_pred

# Convertire la colonna 'consumo_alcohol_pred' in numerico
df_uiu['consumo_alcohol_pred'] = pd.to_numeric(df_uiu['consumo_alcohol_pred'], errors='coerce')

# Filtrare il DataFrame per escludere i record con consumo_alcohol_pred < 0
df_uiu_filtered = df_uiu[df_uiu['consumo_alcohol_pred'] >= 0]

df_uiu_filtered.loc[:, 'accuratezza_percentuale'] = (df_uiu_filtered['consumo_alcohol_pred']  * 100) / df_uiu_filtered['consumo_alcohol']

# Ora possiamo eliminare le colonne originali delle nazioni
df_uiu_filtered = df_uiu_filtered.drop(columns=unique_countries)

print(df_uiu_filtered.to_string())

# Genera una lista di anni da 2020 a 2027
anni_da_aggiungere = list(range(2020, 2028))

# Crea una copia del DataFrame originale per aggiungere le righe espandendo gli anni
df_espanso = df_test.copy()

# Lista per mantenere i DataFrame espansi per ogni anno
expanded_dfs = []

# Itera attraverso i nuovi anni e aggiungi una riga per ciascun anno per ogni nazione
for anno in anni_da_aggiungere:
    # Crea una copia profonda del DataFrame originale per l'anno corrente
    df_anno = df_espanso.copy()
    # Imposta l'anno appropriato
    df_anno['anno'] = anno
    # Aggiungi il DataFrame espanso alla lista
    expanded_dfs.append(df_anno)

# Concatena tutti i DataFrame espansi in uno unico
df_espanso = pd.concat(expanded_dfs, ignore_index=True)

# Mantieni solo le colonne utilizzate durante l'addestramento del modello
df_espanso = df_espanso[['anno', 'cod_naz','nazione_area', 'SWITZERLAND', 'AUSTRALIA', 'ITALY', 'CROATIA',
                         'ESTONIA', 'JAPAN', 'CANADA', 'SINGAPORE', 'GERMANY', 'ARGENTINA',
                         'BELGIUM', 'CHILE', 'CHINA', 'HUNGARY', 'ALTRI',
                         'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND', 'SOUTH AFRICA',
                         'SWEDEN', 'NORWAY', 'MALAYSIA', 'FRANCE', 'GREECE', 'NEW ZEALAND',
                         'REPUBLIC OF KOREA', 'SPAIN', 'TRINIDAD AND TOBAGO', 'IRELAND',
                         'PANAMA', 'NETHERLANDS (KINGDOM OF THE)', 'UKRAINE', 'AUSTRIA',
                         'PARAGUAY', 'ICELAND', 'PHILIPPINES', 'UNITED ARAB EMIRATES',
                         'UNITED STATES OF AMERICA', 'ROMANIA', 'DENMARK', 'BAHRAIN', 'FINLAND',
                         'COSTA RICA', 'RUSSIAN FEDERATION', 'MEXICO', 'MALDIVES', 'CAMBODIA',
                         'PORTUGAL', 'ISRAEL', 'BRAZIL', 'LITHUANIA', 'ALBANIA', 'LATVIA',
                         'URUGUAY', 'SEYCHELLES']]

# Ordina il DataFrame per 'anno'
df_espanso = df_espanso.sort_values(by=['anno'])
df_test_y = df_espanso

df_test_y['anno'] = df_test_y['anno'].astype(int)

print('TF?',df_test_y)

predizioni = loaded_model.predict(df_test_y.drop('nazione_area', axis=1))

# Aggiungiamo predizioni array come colonna al DataFrame
df_test_y.loc[:, 'consumo_alcohol'] = predizioni

# Convertire la colonna 'consumo_alcohol' in numerico
df_test_y['consumo_alcohol'] = pd.to_numeric(df_test_y['consumo_alcohol'], errors='coerce')

print('dopo predizione',df_test_y)

# Filtrare il DataFrame per escludere i record con consumo_alcohol < 0
df_uiu_filtered = df_test_y[(df_test_y['consumo_alcohol'] >= 0) & (df_test_y['anno'] > 2019)]

# Ora possiamo eliminare le colonne originali delle nazioni
df_uiu_filtered = df_uiu_filtered.drop(columns=unique_countries)

print('df filtrato',df_uiu_filtered)

df_predizione_consumi_nodup = df_uiu_filtered.drop_duplicates()

df_predizione_consumi_1 = df_predizione_consumi_nodup[(df_predizione_consumi_nodup['anno'] > 2019) & (df_predizione_consumi_nodup['nazione_area'] != 'ALTRI')]

print('andrà?', df_predizione_consumi_1)

df_pred_consumi = pd.concat([df_consumi_s, df_predizione_consumi_1], ignore_index= True)

df_pred_consumi = df_pred_consumi.drop_duplicates(subset=['anno', 'nazione_area'])

df_pred_consumi['consumo_alcohol'] = df_pred_consumi['consumo_alcohol'].round(2)

print(df_pred_consumi.sort_values(['anno']).to_string())