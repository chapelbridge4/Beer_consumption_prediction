import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sql_server.connessione_postgres_p import filepath_0, filepath

from sql_server import connessione_postgres_p
from prediction import funzioni
from sqlalchemy import create_engine

# pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, LabelEncoder
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

# Query per selezionare tutti i dati relativi agli acquisti/produzione dalla vista 'dati_predizione'
query_dati_acquisti_0 = """
SELECT
    *
from 
    dati_predizione
where
    tipo_causale = 'Acquisto'
"""

# Query per selezionare tutti i dati relativi agli acquisti/produzione dalla vista 'dati_predizione' compreso il mese
query_dati_acquisti = """
select distinct
anno_oper, data_oper as data,EXTRACT(MONTH FROM data_oper) AS mese, recipiente , litri_contenitore , sum(num_contenitore*qta) as num_cont_tot, litri_tot
from dati_predizione dp 
where tipo_causale = 'Acquisto' and anno_oper >= '2010' 
group by anno_oper, recipiente, litri_contenitore, data_oper, EXTRACT(MONTH FROM data_oper), litri_tot 
"""

try:
    # Creazione del motore per la connessione al database PostgreSQL
    engine_pos = create_engine(connessione_postgres_p.create_connection_string(filepath))

    # Utilizza un contesto with per garantire la chiusura della connessione
    with engine_pos.connect() as conn_pos:
        if conn_pos is None:
            print("Connessione a PostgreSQL non stabilita.")
        else:
            # Esegui la query per ottenere i dati delle acquisti dall'azienda dal database
            df_azienda_acquisti = pd.read_sql(query_dati_acquisti, conn_pos)

except Exception as e:
    # Gestione delle eccezioni nel caso si verifichi un errore durante la connessione o l'esecuzione della query
    print(f"Si è verificato un errore: {e}")

else:
    # Stampare le colonne solo se il blocco try è stato eseguito con successo
    print(df_azienda_acquisti.columns)
    print(df_azienda_acquisti)


# Crea un DataFrame con le colonne selezionate da df_azienda_acquisti
df_acquisti_selezionato = pd.DataFrame(df_azienda_acquisti, columns=['anno_oper', 'mese', 'data', 'recipiente', 'litri_contenitore', 'litri_tot'])

print(df_acquisti_selezionato)
print(df_acquisti_selezionato.describe())

# Controllo dei valori nulli per df_acquisti_selezionato
rows_with_nan = df_acquisti_selezionato[df_acquisti_selezionato.isnull().any(axis=1)]
print('righe con valori nan: ', rows_with_nan)

df_acquisti_selezionato.hist(figsize=(20, 15))
plt.show()

# Estrarre la colonna 'litri_tot' dal DataFrame df_acquisti_selezionato
df_acquisti = df_acquisti_selezionato["litri_tot"]

# Forma della distribuzione acquisti
sns.displot(df_acquisti)
plt.title('Distribuzione acquisti/produzione', fontsize=16)  # Aggiungo il titolo
plt.show()

# Eseguo la somma dei litri_tot per recipiente e litri_contenitore
trai = df_acquisti_selezionato.groupby(['recipiente', 'litri_contenitore'])['litri_tot'].apply(lambda x : x.sum()).reset_index()
trai['litri_tot'] = trai['litri_tot'].apply(lambda x: '{:.0f}'.format(x))
print(trai)

# Concateno le colonne recipiente e litri_contenitore per poi inserirle in una colonna categoria
df_acquisti_selezionato['categoria'] = df_acquisti_selezionato['recipiente'] + '_' + df_acquisti_selezionato['litri_contenitore'].astype(str)

# Seleziono i valori di categoria che sono meno presenti e che hanno valori molto più bassi della media nel dataframe
df_selezionato = df_acquisti_selezionato[df_acquisti_selezionato['categoria'].isin(['Cisterne_500.0', 'Cisterne_999.99', 'Bottiglie_2.0', 'Fusti_24.0', 'Bottiglie_0.2', 'Bottiglie_0.35', 'Bottiglie_0.5'])]
# Rimuovo i valori che risultano molto più bassi della media
df_acquisti_selezionato.drop(df_selezionato.index, inplace=True)

print(df_acquisti_selezionato)

# Istanzia l'oggetto LabelEncoder
label_encoder = LabelEncoder()

# Codifica le etichette della colonna 'categoria'
df_acquisti_selezionato['categoria_encoded'] = label_encoder.fit_transform(df_acquisti_selezionato['categoria'])

# Stampare le etichette originali e codificate
print("Etichette originali:")
print(df_acquisti_selezionato['categoria'].unique())

print("\nEtichette codificate:")
print(df_acquisti_selezionato['categoria_encoded'].unique())

data_acq = df_acquisti_selezionato['litri_tot']

print(df_acquisti_selezionato)

# Calcola i limiti superiore e inferiore dello scarto interquartile per i dati della colonna 'litri_tot'
up_limit, low_limit = funzioni.interquartile_range_limits(data_acq)
print(up_limit, low_limit)

# Calcola i limiti dello scarto interquartile per i dati di quella colonna
print(f"Creazione del grafico per la colonna di df_acquisti: {'litri_tot'}")

# Crea un grafico a dispersione per i dati di quella colonna, con i limiti dello scarto interquartile
funzioni.plot_generic_distribution(data_acq, up_limit, low_limit)

# Sostituisci tutti i valori nell'array tmp_acq che sono superiori al limite superiore o inferiori con il valore del limite corrispondente
tmp_acq = data_acq.to_numpy()

# Conta quanti valori superano il limite superiore (outliers)
tmp_try = tmp_acq[tmp_acq > up_limit]
tmpContato = tmp_try.shape[0]
print('Valori che superano up_limit: ', tmpContato)

# Sostituisci gli outliers con i limiti superiore e inferiore
tmp_acq[tmp_acq > up_limit]  = up_limit
tmp_acq[tmp_acq < low_limit] = low_limit

# Visualizza l'istogramma dei dati dopo la rimozione degli outliers
sns.displot(tmp_acq)
plt.show()

# Seleziona solo le colonne numeriche dal DataFrame delle acquisti
df_numerico_acquisti = df_acquisti_selezionato.select_dtypes(include='number')
print(df_numerico_acquisti)

# Matrice di correlazione
corr_matrix_acquisti = df_numerico_acquisti.corr()

# Aggiungi un titolo per la visualizzazione di litri_tot
print("Correlazione delle colonne rispetto a 'litri_tot' - Acquisti/Produzione")

# plot acquisti
plt.figure(figsize=(16, 16))
sns.heatmap(corr_matrix_acquisti, annot=True)
plt.title('Matrice di Correlazione - Acquisti/Produzione', fontsize=16)  # Aggiungo il titolo
plt.show()

# Calcola la correlazione delle colonne rispetto alla colonna 'litri_tot' e ordina in modo decrescente
corr_acquisti = corr_matrix_acquisti['litri_tot'].sort_values(ascending=False)

# Aggiungi un titolo per la visualizzazione
print("Correlazione delle colonne rispetto a 'litri_tot' - Acquisti/Produzione")

# Visualizza la serie di correlazioni
print(corr_acquisti)

print(df_acquisti_selezionato)

# Sommo i litri_tot per categoria, categoria_encoded, anno_oper e mese
df_grouped = df_acquisti_selezionato.groupby(['categoria', 'categoria_encoded', 'anno_oper', 'mese'])[['litri_tot']].apply(lambda x : x.astype(int).sum()).reset_index()
print(df_grouped)

# Ottieni i nomi unici delle categorie presenti nel DataFrame df_grouped prende entrambi il df
unique_categories = df_grouped['categoria'].unique()

# Codifica i valori usando OneHotEncoder
hot_encoder = OneHotEncoder(sparse_output=False)
hot_encoded = hot_encoder.fit_transform(df_grouped['categoria'].values.reshape(-1, 1))

# Creazione del nuovo DataFrame con i nomi delle categorie come nomi delle colonne
df_encoded = pd.DataFrame(hot_encoded, columns=unique_categories)

# Aggiungi il nuovo DataFrame al DataFrame originale
df_encoded = pd.concat([df_grouped.reset_index(drop=True), df_encoded], axis=1)

print('CATEGORIZZATO ACQUISTI/PRODUZIONE', df_encoded)

# Controllo dei valori nulli per df_acquisti_selezionato
rows_with_nan_2 = df_encoded[df_encoded.isnull().any(axis=1)]
print('Righe con valori nan: ', rows_with_nan_2)

# Esegui il merge dei DataFrame con specifica del suffisso per le colonne duplicate
df_uiu = df_encoded.select_dtypes(include=['number']).join(df_encoded[['anno_oper', 'categoria']], rsuffix='_encoded')

df_test = df_uiu.copy()

# Stampa il DataFrame df_uiu
# print(df_uiu.to_string())
print(df_uiu)

# Seleziona le features X rimuovendo la colonna 'litri_tot'
X = df_uiu.drop(['litri_tot', 'categoria'], axis=1)

# Seleziona il target Y dalla colonna 'litri_tot'
Y = df_uiu['litri_tot']

# test 1
funzioni.pipeline_validation(make_pipeline(StandardScaler(), LinearRegression()), X, Y)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), LinearRegression()), X, Y)

# test 2
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X, Y)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X, Y)

# Valida un modello di regressione lineare polinomiale con cross-validation k-fold
funzioni.pipeline_poly_validation(LinearRegression(), X, Y, 5)

# Valida un modello di regressione Ridge polinomiale con cross-validation k-fold
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 3), Ridge(alpha=10, fit_intercept=True)), X, Y)
# Plotta la curva di apprendimento per il modello di regressione Ridge polinomiale
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 3), Ridge(alpha=10, fit_intercept=True)), X, Y)

# Utilizza la funzione pipeline_poly_validation per validare un modello di regressione Ridge polinomiale
funzioni.pipeline_poly_validation(Ridge(), X, Y, 8)

# Testa un modello di regressione Ridge
funzioni.test_pipeline_ridge(X, Y, 3, 100)

# Ottimizza i parametri del modello di regressione Ridge
poly_degree, ridge_alpha, ridge_intercept = funzioni.ottimization(X, Y)

# Testa il modello utilizzando i dati di input X e il target Y
test_model(X, Y)

df_acquisti_s = df_uiu.drop(columns=unique_categories)

# Controlla che 'anno_oper' esista nel DataFrame
if 'anno_oper' in df_acquisti_s.columns:
    # Cast di 'anno_oper'
    df_acquisti_s['anno_oper'] = df_acquisti_s['anno_oper'].astype(int)
    df_acquisti_s = df_acquisti_s[df_acquisti_s['anno_oper'] <= 2019]
else:
    print("Colonna 'anno_oper' non trovata nel DataFrame.")

def save_model_with_pickle(df):
    """
    Salva il modello addestrato utilizzando pickle.

    Args:
        df (DataFrame): Il DataFrame contenente i dati di addestramento.

    Returns:
        None
    """
    # Creazione delle features e target
    X = df.drop(['litri_tot', 'categoria'], axis=1)
    Y = df['litri_tot']

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
    with open("prediction/model_acquisti.pkl", "wb") as file:
        pickle.dump(model_pipeline, file)


# Chiamata alla funzione per salvare il modello
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

# Esempio di caricamento del modello
loaded_model = load_model_with_pickle('model_acquisti.pkl')

# Fai previsioni utilizzando il modello caricato
Y_pred = loaded_model.predict(df_uiu.drop(['litri_tot', 'categoria'], axis=1))

# print(Y_pred)

# Aggiungi la colonna 'litri_tot_pred' contenente i valori predetti al DataFrame df_uiu
df_uiu['litri_tot_pred'] = Y_pred

# Formatta i valori nella colonna 'litri_tot_pred' senza notazione scientifica
df_uiu['litri_tot_pred'] = df_uiu['litri_tot_pred'].apply(lambda x: '{:.0f}'.format(x))

# Convertire la colonna 'litri_tot_pred' in numerico
df_uiu['litri_tot_pred'] = pd.to_numeric(df_uiu['litri_tot_pred'], errors='coerce')

# Filtrare il DataFrame per escludere i record con litri_tot_pred < 0
df_uiu_filtered = df_uiu[df_uiu['litri_tot_pred'] >= 0]

df_uiu_filtered.loc[:, 'diff_pred_perc'] = (df_uiu_filtered['litri_tot_pred'] / df_uiu_filtered['litri_tot']) * 100

print(df_uiu_filtered)

# Crea una copia del DataFrame originale per aggiungere le righe espandendo gli anni
df = df_test[['categoria_encoded','categoria', 'Bottiglie_0.33', 'Bottiglie_0.66', 'Bottiglie_0.75', 'Cisterne_10000.0', 'Cisterne_250.0', 'Fusti_15.0', 'Fusti_20.0', 'Fusti_30.0', 'Fusti_5.0', 'Lattine_0.33']].copy()

df = df.drop_duplicates()

print(df)

cat = df['categoria'].unique()


df_espanso = pd.DataFrame(columns=df.columns)  # DataFrame in cui verranno aggiunti i dati espansi

for categoria in cat:
    for anno in range(2020, 2028):
        for mese in range(1, 13):
            # Crea una copia del DataFrame originale
            df_anno = df[df['categoria'] == categoria].copy()

            # Imposta l'anno e il mese appropriati
            df_anno['anno_oper'] = anno
            df_anno['mese'] = mese

            # Aggiungi la riga al DataFrame espanso
            df_espanso = pd.concat([df_espanso, df_anno], ignore_index=True)


# VEDI COME FIXARE STA ROBA è SNERVANTE va ma toglie le cose negative sono tonto lol

print(df_espanso.to_string())

df_espanso['anno_oper'] = df_espanso['anno_oper'].astype(int)
df_espanso = df_espanso.drop_duplicates()
# df_espanso = df_espanso[df_espanso['anno_oper'] >= 2020]

# Mantieni solo le colonne utilizzate durante l'addestramento del modello
df_espanso = df_espanso[['categoria_encoded', 'mese', 'Bottiglie_0.33', 'Bottiglie_0.66', 'Bottiglie_0.75',
       'Cisterne_10000.0', 'Cisterne_250.0', 'Fusti_15.0', 'Fusti_20.0',
       'Fusti_30.0', 'Fusti_5.0', 'Lattine_0.33', 'anno_oper', 'categoria']]

# Ordina il DataFrame per 'anno'
df_espanso = df_espanso.sort_values(by=['anno_oper'])

# print(df_espanso.to_string())

# Visualizza il DataFrame df_test dopo le modifiche
print(df_espanso.columns)

predizioni = loaded_model.predict(df_espanso.drop('categoria', axis=1))

# Aggiungiamo predizioni array come colonna al DataFrame
df_espanso['litri_tot'] = predizioni

df_espanso['litri_tot'] = df_espanso['litri_tot'].apply(lambda x: '{:.0f}'.format(x))

df_espanso = df_espanso.drop(columns=unique_categories)

# Convertire la colonna 'predizione' in numerico
df_espanso['litri_tot'] = pd.to_numeric(df_espanso['litri_tot'], errors='coerce')

# print(df_espanso.to_string())

df_deduplicato = df_espanso.drop_duplicates(subset=['categoria_encoded', 'mese', 'anno_oper', 'categoria'])

# print(df_deduplicato.to_string())

# Filtrare il DataFrame per escludere i record con litri_tot_pred < 0 dato che ci sono predizioni negative
df_acquisti_predizione = df_deduplicato[df_deduplicato['litri_tot'] >= 0]

df_acquisti_pred = pd.concat([df_acquisti_s, df_acquisti_predizione], ignore_index= True)

df_acquisti_pred = df_acquisti_pred.drop_duplicates(subset=['categoria_encoded', 'mese', 'anno_oper', 'categoria'])

# Dovrebbe essere pronto per essere caricato sul db con un'accuratezza del 80%
print(df_acquisti_pred.columns)
print(df_acquisti_pred.sort_values(['anno_oper']).to_string())
