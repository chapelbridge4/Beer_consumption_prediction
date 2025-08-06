import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from prediction.funzioni import interquartile_range_limits, plot_generic_distribution
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

# Query per selezionare tutti i dati relativi al numero di persone che consumano birra dalla tabella 'odata_regita'
query_dati_persone = """
SELECT distinct
    anno,
    territorio,
    tipo_bevanda, 
    tipo_dato, 
    value*1000 as num_persone
from 
    odata_regita
where
    misura_avq = 'THV' 
"""

try:
    # Creazione del motore per la connessione al database PostgreSQL
    engine_pos = create_engine(connessione_postgres_p.create_connection_string(filepath))

    # Utilizza un contesto with per garantire la chiusura della connessione
    with engine_pos.connect() as conn_pos:
        if conn_pos is None:
            print("Connessione a PostgreSQL non stabilita.")
        else:
            # Esegui la query per ottenere i dati degli open data dal database
            df_persone_alcohol = pd.read_sql(query_dati_persone, conn_pos)

except Exception as e:
    # Gestione delle eccezioni nel caso si verifichi un errore durante la connessione o l'esecuzione della query
    print(f"Si è verificato un errore: {e}")

else:
    # Stampare le colonne solo se il blocco try è stato eseguito con successo
    print(df_persone_alcohol.columns)

print(df_persone_alcohol.sort_values('anno', ascending=True).to_string())

# Creo una lista regioni per identificare tutte le regioni nel df
regioni_1 = ['Emilia-Romagna',
                   'Provincia Autonoma Bolzano / Bozen',
                   'Marche',
                   'Sicilia',
                   'Valle d\'Aosta / Vallée d\'Aoste',
                   'Basilicata',
                   'Abruzzo',
                   'Piemonte',
                   'Toscana',
                   'Lazio',
                   'Sardegna',
                   'Liguria',
                   'Lombardia',
                   'Campania',
                   'Puglia',
                   'Friuli-Venezia Giulia',
                   'Molise',
                   'Umbria',
                   'Provincia Autonoma Trento',
                   'Veneto',
                   'Trentino Alto Adige / Südtirol',
                   'Calabria',
                   ]

regioni = ['Emilia-Romagna',
                   'Provincia Autonoma Bolzano / Bozen',
                   'Marche',
                   'Valle d\'Aosta / Vallée d\'Aoste',
                   'Abruzzo',
                   'Piemonte',
                   'Toscana',
                   'Liguria',
                   'Lombardia',
                   'Campania',
                   'Friuli-Venezia Giulia',
                   'Provincia Autonoma Trento',
                   'Trentino Alto Adige / Südtirol',
                   ]

# Creo una lista regioni per identificare tutte le zone nel df
zone_1 = ['Nord',
        'Nord-ovest',
        'Isole',
        'Sud',
        'Centro',
        'Mezzogiorno',
        ]

# Creo una lista regioni per identificare tutte le zone nel df
zone = ['Lazio',
        'Veneto',
        'Sicilia',
        'Puglia',
        'Calabria',
        'Sardegna',
        'Umbria',
        'Basilicata',
        'Molise',
        ]

# Controllo dei valori nulli per df_persone_alcohol
rows_with_nan = df_persone_alcohol[df_persone_alcohol.isnull().any(axis=1)]
print('righe con valori nan: ', rows_with_nan)

df_persone_alcohol['anno'] = pd.to_numeric(df_persone_alcohol['anno'], errors='coerce')

# Seleziona i record nel df che corrispondono alle regioni
df_regioni = df_persone_alcohol[df_persone_alcohol['territorio'].isin(regioni)]
print(df_regioni)

# Seleziona i record nel df che corrispondono alle zone
df_zone = df_persone_alcohol[df_persone_alcohol['territorio'].isin(zone)]
print(df_zone)

sns.displot(df_regioni['num_persone'])
plt.title('Distribuzione numero di persone - Regioni', fontsize=16)
plt.show()

# Analisi statistica per df_regioni
print(df_regioni['num_persone'].describe())

# Calcolo dei limiti superiore e inferiore dello scarto interquartile per la colonna 'num_persone'
data_regioni_persone = df_regioni['num_persone']
up_limit_regioni, low_limit_regioni = interquartile_range_limits(data_regioni_persone)
print("Limiti per df_regioni['num_persone']:", up_limit_regioni, low_limit_regioni)

# Creazione del grafico per visualizzare la distribuzione della colonna 'num_persone'
print(f"Creazione del grafico per la colonna di df_regioni: {'num_persone'}")
plot_generic_distribution(data_regioni_persone, up_limit_regioni, low_limit_regioni)

# Identificazione dei valori che superano il limite superiore
tmp_regioni = data_regioni_persone.to_numpy()
tmp_try_regioni = tmp_regioni[tmp_regioni > up_limit_regioni]
tmpContato_regioni = tmp_try_regioni.shape[0]
print('Valori che superano up_limit: ', tmpContato_regioni)

# Sostituzione dei valori che superano i limiti con i limiti stessi
tmp_regioni[tmp_regioni > up_limit_regioni]  = up_limit_regioni
tmp_regioni[tmp_regioni < low_limit_regioni] = low_limit_regioni

# Visualizzazione della distribuzione dei valori dopo la rimozione degli outliers
sns.displot(tmp_regioni)
plt.title('Distribuzione numero di persone - Regioni (senza outliers)')
plt.show()

# Analisi statistica per df_zone
sns.displot(df_zone['num_persone'])
plt.title('Distribuzione numero di persone - Zone', fontsize=16)
plt.show()

print(df_zone['num_persone'].describe())

# Calcolo dei limiti superiore e inferiore dello scarto interquartile per la colonna 'num_persone'
data_zone_persone = df_zone['num_persone']
up_limit_zone, low_limit_zone = interquartile_range_limits(data_zone_persone)
print("Limiti per df_zone['num_persone']:", up_limit_zone, low_limit_zone)

# Creazione del grafico per visualizzare la distribuzione della colonna 'num_persone'
print(f"Creazione del grafico per la colonna di df_zone: {'num_persone'}")
plot_generic_distribution(data_zone_persone, up_limit_zone, low_limit_zone)

# Identificazione dei valori che superano il limite superiore
tmp_zone = data_zone_persone.to_numpy()
tmp_try_zone = tmp_zone[tmp_zone > up_limit_zone]
tmpContato_zone = tmp_try_zone.shape[0]
print('Valori che superano up_limit: ', tmpContato_zone)

# Sostituzione dei valori che superano i limiti con i limiti stessi
tmp_zone[tmp_zone > up_limit_zone]  = up_limit_zone
tmp_zone[tmp_zone < low_limit_zone] = low_limit_zone

# Visualizzazione della distribuzione dei valori dopo la rimozione degli outliers
sns.displot(tmp_zone)
plt.title('Distribuzione numero di persone - Zone (senza outliers)')
plt.show()

# Creazione di nuove colonne categoriche per territorio e tipo_bevanda per df_regioni
df_regioni_encoded = df_regioni.copy()
df_regioni_encoded['territorio_cat'] = pd.Categorical(df_regioni_encoded['territorio']).codes
df_regioni_encoded['tipo_bevanda_cat'] = pd.Categorical(df_regioni_encoded['tipo_bevanda']).codes
print(df_regioni_encoded)

# Creazione di nuove colonne categoriche per territorio e tipo_bevanda per df_zone
df_zone_encoded = df_zone.copy()
df_zone_encoded['territorio_cat'] = pd.Categorical(df_zone_encoded['territorio']).codes
df_zone_encoded['tipo_bevanda_cat'] = pd.Categorical(df_zone_encoded['tipo_bevanda']).codes
print(df_zone_encoded)

# Seleziona solo le colonne numeriche dal DataFrame degli open data
df_numerico_zone = df_zone_encoded.select_dtypes(include='number')
print(df_numerico_zone)

# Seleziona solo le colonne numeriche dal DataFrame degli open data
df_numerico_regioni = df_regioni_encoded.select_dtypes(include='number')
print(df_numerico_regioni)

# Matrice di correlazione
corr_matrix_zone = df_numerico_zone.corr()
corr_matrix_regioni = df_numerico_regioni.corr()

# Aggiungi un titolo per la visualizzazione di num_persone
print("Correlazione delle colonne rispetto a 'num_persone' - zone")
# plot zone
plt.figure(figsize=(16, 16))
sns.heatmap(corr_matrix_zone, annot=True)
plt.title('Matrice di Correlazione - zone', fontsize=16)  # Aggiungo il titolo
plt.show()
# Calcola la correlazione delle colonne rispetto alla colonna 'num_persone' e ordina in modo decrescente
corr_zone = corr_matrix_zone['num_persone'].sort_values(ascending=False)
# Aggiungi un titolo per la visualizzazione del vino rosso
print("Correlazione delle colonne rispetto a 'num_persone' - zone")
# Visualizza la serie di correlazioni
print(corr_zone)
df_zone_2 = corr_matrix_zone.copy()  # Copia il DataFrame per non modificare quello originale
print(df_zone_2.columns)

# Aggiungi un titolo per la visualizzazione di num_persone
print("Correlazione delle colonne rispetto a 'num_persone' - regioni")
# plot regioni
plt.figure(figsize=(16, 16))
sns.heatmap(corr_matrix_regioni, annot=True)
plt.title('Matrice di Correlazione - regioni', fontsize=16)  # Aggiungo il titolo
plt.show()
# Calcola la correlazione delle colonne rispetto alla colonna 'num_persone' e ordina in modo decrescente
corr_regioni = corr_matrix_regioni['num_persone'].sort_values(ascending=False)
# Aggiungi un titolo per la visualizzazione del vino rosso
print("Correlazione delle colonne rispetto a 'num_persone' - regioni")
# Visualizza la serie di correlazioni
print(corr_regioni)
df_regioni_2 = df_numerico_regioni.copy()  # Copia il DataFrame per non modificare quello originale
print(df_regioni_2.columns)

# Ottieni i nomi unici dei paesi presenti nel DataFrame df_country prende entrambi i df
unique_regioni = df_regioni_encoded['territorio'].unique()
unique_zone = df_zone_encoded['territorio'].unique()

# Codifica i valori usando OneHotEncoder
hot_encoder = OneHotEncoder(sparse_output=False)
hot_encoded_reg = hot_encoder.fit_transform(df_regioni_encoded['territorio'].values.reshape(-1, 1))
hot_encoded_zon = hot_encoder.fit_transform(df_zone_encoded['territorio'].values.reshape(-1, 1))

# Creazione del nuovo DataFrame con i nomi dei paesi come nomi delle colonne
regioni_df_encoded = pd.DataFrame(hot_encoded_reg, columns=unique_regioni)
zone_df_encoded_mgl = pd.DataFrame(hot_encoded_zon, columns=unique_zone)

# Aggiungi il nuovo DataFrame al DataFrame originale
regioni_df_encoded = pd.concat([df_regioni_encoded.reset_index(drop=True), regioni_df_encoded], axis=1)
zone_df_encoded = pd.concat([df_zone_encoded.reset_index(drop=True), zone_df_encoded_mgl], axis=1)

print('CATEGORIZZATO regioni: ', regioni_df_encoded)
print('CATEGORIZZATO zone: ', zone_df_encoded)

df_uiu = regioni_df_encoded.sort_values('anno', ascending=True)

# Seleziona le features X rimuovendo le colonne 'num_persone', 'territorio', 'tipo_bevanda', 'tipo_dato'
X = df_uiu.drop(['num_persone', 'territorio', 'tipo_bevanda', 'tipo_dato'], axis=1)
print(X.columns)
# Seleziona il target Y dalla colonna 'num_persone'
Y = df_uiu['num_persone']

# test 1
funzioni.pipeline_validation(make_pipeline(StandardScaler(), LinearRegression()), X, Y)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), LinearRegression()), X, Y)

# test 2
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X, Y)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X, Y)

# Valida un modello di regressione lineare polinomiale con cross-validation k-fold
funzioni.pipeline_poly_validation(LinearRegression(), X, Y, 3)

# Valida un modello di regressione Ridge polinomiale con cross-validation k-fold
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 3), Ridge(alpha=10, fit_intercept=False)), X, Y)
# Plotta la curva di apprendimento per il modello di regressione Ridge polinomiale
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 3), Ridge(alpha=10, fit_intercept=False)), X, Y)

# Utilizza la funzione pipeline_poly_validation per validare un modello di regressione Ridge polinomiale
funzioni.pipeline_poly_validation(Ridge(), X, Y, 4)

# Testa un modello di regressione Ridge
funzioni.test_pipeline_ridge(X, Y, 3, 100)

# Ottimizza i parametri del modello di regressione Ridge
poly_degree, ridge_alpha, ridge_intercept = funzioni.ottimization(X, Y)

# Testa il modello utilizzando i dati di input X e il target Y
test_model(X, Y)

df_uiu_zone = zone_df_encoded.sort_values('anno', ascending=True)

# Seleziona le features X rimuovendo la colonna 'num_persone', 'territorio', 'tipo_bevanda', 'tipo_dato'
X_zon = df_uiu_zone.drop(['num_persone', 'territorio', 'tipo_bevanda', 'tipo_dato'], axis=1)
print(X_zon.columns)
# Seleziona il target Y dalla colonna 'num_persone'
Y_zon = df_uiu_zone['num_persone']

# test 1
funzioni.pipeline_validation(make_pipeline(StandardScaler(), LinearRegression()), X_zon, Y_zon)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), LinearRegression()), X_zon, Y_zon)

# test 2
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X_zon, Y_zon)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X_zon, Y_zon)

# Valida un modello di regressione lineare polinomiale con cross-validation k-fold
funzioni.pipeline_poly_validation(LinearRegression(), X_zon, Y_zon, 3)

# Valida un modello di regressione Ridge polinomiale con cross-validation k-fold
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 4), Ridge(alpha=1, fit_intercept=True)), X_zon, Y_zon)
# Plotta la curva di apprendimento per il modello di regressione Ridge polinomiale
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 4), Ridge(alpha=1, fit_intercept=True)), X_zon, Y_zon)

# Utilizza la funzione pipeline_poly_validation per validare un modello di regressione Ridge polinomiale
funzioni.pipeline_poly_validation(Ridge(), X_zon, Y_zon, 4)

# Testa un modello di regressione Ridge
funzioni.test_pipeline_ridge(X_zon, Y_zon, 3, 10)

# Ottimizza i parametri del modello di regressione Ridge
poly_degree, ridge_alpha, ridge_intercept = funzioni.ottimization(X_zon, Y_zon)

# Testa il modello utilizzando i dati di input X e il target Y
test_model(X_zon, Y_zon)

df_territori_s = pd.concat([df_uiu, df_uiu_zone], ignore_index= True)

colonne_da_rimuovere = np.concatenate((unique_regioni, unique_zone))
df_territori_s = df_territori_s.drop(columns=colonne_da_rimuovere)

# Check if 'anno' column exists in the DataFrame
if 'anno' in df_territori_s.columns:
    # Cast 'anno_oper' to integer type
    df_territori_s['anno'] = df_territori_s['anno'].astype(int)
    # Filter rows where 'anno' is less than or equal to 2020
    df_territori_s = df_territori_s[df_territori_s['anno'] <= 2022]
else:
    print("Column 'anno' not found in DataFrame.")

df_territori_s['num_persone'] = df_territori_s['num_persone'].astype(int)

df_territori_s['num_persone'] = df_territori_s['num_persone'].apply(lambda x: '{:.0f}'.format(x))

def save_model_with_pickle(df, nome_modello):
    """
    Salva il modello addestrato utilizzando pickle.

    Args:
        df (DataFrame): Il DataFrame contenente i dati di addestramento.

    Returns:
        None
    """
    # Creazione delle features e target
    X = df.drop(['num_persone', 'territorio', 'tipo_bevanda', 'tipo_dato'], axis=1)
    Y = df['num_persone']

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
    with open('prediction/' + nome_modello, "wb") as file:
        pickle.dump(model_pipeline, file)

# Chiamata alla funzione per salvare i modelli
save_model_with_pickle(df_uiu, 'modello_regIta.pkl')
save_model_with_pickle(df_uiu_zone, 'modello_zonIta.pkl')

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

# Esempio di caricamento dei modelli
loaded_model_reg = load_model_with_pickle('modello_regIta.pkl')
loaded_model_zon = load_model_with_pickle('modello_zonIta.pkl')

df_regioni_pred = df_uiu.copy()
df_zone_pred = df_uiu_zone.copy()

Y_pred_reg = loaded_model_reg.predict(df_regioni_pred.drop(['num_persone', 'territorio', 'tipo_bevanda', 'tipo_dato'], axis=1))
Y_pred_zon = loaded_model_zon.predict(df_zone_pred.drop(['num_persone', 'territorio', 'tipo_bevanda', 'tipo_dato'], axis=1))

df_regioni_pred['num_persone_pred'] = Y_pred_reg
df_zone_pred['num_persone_pred'] = Y_pred_zon

# Formatta i valori nella colonna 'num_persone_pred' senza notazione scientifica
df_regioni_pred['num_persone_pred'] = df_regioni_pred['num_persone_pred'].apply(lambda x: '{:.0f}'.format(x))
df_zone_pred['num_persone_pred'] = df_zone_pred['num_persone_pred'].apply(lambda x: '{:.0f}'.format(x))

# Convertire la colonna 'num_persone_pred' in numerico
df_regioni_pred['num_persone_pred'] = pd.to_numeric(df_regioni_pred['num_persone_pred'], errors='coerce')
df_zone_pred['num_persone_pred'] = pd.to_numeric(df_zone_pred['num_persone_pred'], errors='coerce')

# Filtrare il DataFrame per escludere i record con num_persone_pred < 0
df_regioni_filtered = df_regioni_pred[df_regioni_pred['num_persone_pred'] >= 0]
df_zone_filtered = df_zone_pred[df_zone_pred['num_persone_pred'] >= 0]

df_regioni_filtered.loc[:, 'accuratezza_percentuale'] = (df_regioni_filtered['num_persone_pred']  * 100) / df_regioni_filtered['num_persone']
df_zone_filtered.loc[:, 'accuratezza_percentuale'] = (df_zone_filtered['num_persone_pred']  * 100) / df_zone_filtered['num_persone']

# Ora possiamo eliminare le colonne originali dei territori
df_regioni_filtered = df_regioni_filtered.drop(columns=unique_regioni)
df_zone_filtered = df_zone_filtered.drop(columns=unique_zone)

print(df_regioni_filtered.to_string())

print(df_zone_filtered.to_string())

# Genera una lista di anni da 2023 a 2027
anni_da_aggiungere = list(range(2023, 2028))

df_test_reg = df_uiu.copy()

print(df_test_reg)

# Itera attraverso i nuovi anni e aggiungi una riga per ciascun anno per ogni territorio
for anno in anni_da_aggiungere:
    # Per ogni anno, duplica il DataFrame originale, imposta l'anno appropriato e concatena
    df_anno = df_test_reg.copy()
    df_anno['anno'] = anno
    df_test_reg = pd.concat([df_test_reg, df_anno], ignore_index=True)

# df_espanso_reg_1 = df_test_reg[['anno', 'tipo_dato', 'territorio', 'territorio_cat', 'tipo_bevanda_cat', 'Piemonte',
#        'Emilia-Romagna', 'Puglia', 'Calabria', 'Sicilia', 'Veneto', 'Lazio',
#        'Liguria', 'Molise', 'Campania', 'Basilicata', 'Friuli-Venezia Giulia',
#        'Provincia Autonoma Bolzano / Bozen', 'Sardegna', 'Abruzzo', 'Toscana',
#        'Umbria', 'Trentino Alto Adige / Südtirol',
#        'Valle d\'Aosta / Vallée d\'Aoste', 'Lombardia', 'Marche',
#        'Provincia Autonoma Trento']]

df_espanso_reg = df_test_reg[['anno','territorio', 'tipo_dato', 'territorio_cat', 'tipo_bevanda_cat', 'Piemonte',
       'Emilia-Romagna', 'Liguria', 'Campania', 'Friuli-Venezia Giulia',
       'Provincia Autonoma Bolzano / Bozen', 'Abruzzo', 'Toscana',
       'Trentino Alto Adige / Südtirol', 'Valle d\'Aosta / Vallée d\'Aoste',
       'Lombardia', 'Marche', 'Provincia Autonoma Trento']]

df_espanso_reg.sort_values('anno', ascending=True)

df_espanso_reg = df_espanso_reg[df_espanso_reg['anno'] > 2022]

print(df_espanso_reg.columns)
print(df_espanso_reg)

regioni_prediction = loaded_model_reg.predict(df_espanso_reg.drop(['territorio', 'tipo_dato'], axis=1))

df_espanso_reg.loc[:, 'num_persone'] = regioni_prediction

df_espanso_reg['num_persone'] = pd.to_numeric(df_espanso_reg['num_persone'], errors='coerce')

# Filtra in base a 'num_persone_pred'
df_espanso_reg_filtered = df_espanso_reg[df_espanso_reg['num_persone'] >= 0]

df_espanso_reg_filtered['num_persone'] = df_espanso_reg_filtered['num_persone'].apply(lambda x: '{:.0f}'.format(x))

# Ora possiamo eliminare le colonne originali dei territori
regioni_filtered = df_espanso_reg_filtered.drop(columns=unique_regioni)

print(regioni_filtered.head(100).to_string())

# Genera una lista di anni da 2023 a 2027
anni_da_aggiungere_zon = list(range(2023, 2028))

df_test_zon = df_uiu_zone.copy()

# print(df_test_zon)

# Itera attraverso i nuovi anni e aggiungi una riga per ciascun anno per ogni territorio
for anno in anni_da_aggiungere_zon:
    # Per ogni anno, duplica il DataFrame originale, imposta l'anno appropriato e concatena
    df_anno = df_test_zon.copy()
    df_anno['anno'] = anno
    df_test_zon = pd.concat([df_test_zon, df_anno], ignore_index=True)

df_test_zon = df_test_zon[df_test_zon['anno'] > 2022]

# df_espanso_zon_1 = df_test_zon[['anno', 'tipo_dato', 'territorio', 'territorio_cat', 'tipo_bevanda_cat', 'Centro', 'Nord-ovest',
#        'Nord', 'Isole', 'Mezzogiorno', 'Sud']]

df_espanso_zon = df_test_zon[['anno','territorio', 'tipo_dato', 'territorio_cat', 'tipo_bevanda_cat', 'Puglia', 'Calabria',
       'Sicilia', 'Veneto', 'Lazio', 'Molise', 'Basilicata', 'Sardegna',
       'Umbria']]

df_espanso_zon.sort_values('anno', ascending=True)

print(df_espanso_zon.columns)
print(df_espanso_zon)

zone_prediction = loaded_model_zon.predict(df_espanso_zon.drop(['territorio', 'tipo_dato'], axis=1))

df_espanso_zon.loc[:, 'num_persone'] = zone_prediction

df_espanso_zon['num_persone'] = df_espanso_zon['num_persone'].astype(int)

# Filtra in base a 'num_persone_pred'
df_espanso_zon_filtered = df_espanso_zon[df_espanso_zon['num_persone'] >= 0]

df_espanso_zon_filtered['num_persone'] = df_espanso_zon_filtered['num_persone'].apply(lambda x: '{:.0f}'.format(x))

# Ora possiamo eliminare le colonne originali dei territori
zone_filtered = df_espanso_zon_filtered.drop(columns=unique_zone)

# print(zone_filtered)

df_territorioIta_pred = pd.concat([regioni_filtered, zone_filtered], ignore_index=True)

# Cancella i duplicati
df_territorioIta_predizione = df_territorioIta_pred.drop_duplicates()

df_territorioIta_pred = pd.concat([df_territori_s.drop('tipo_bevanda', axis=1), df_territorioIta_predizione], ignore_index=True)

# Esegue il rest dell'indice
df_territorioIta_pred.reset_index(drop=True, inplace=True)

df_territorioIta_pred['num_persone'] = df_territorioIta_pred['num_persone'].astype(int)

print(df_territorioIta_pred)
