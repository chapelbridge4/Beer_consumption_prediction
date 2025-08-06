import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sql_server import connessione_postgres_p
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

# Query per selezionare tutti i dati interessati relativi alle vendite dalla tabella 'dati_predizione'

# Query realizzata per stagione
query_dati_vendite_1 = """
select distinct 
    cod_naz,
    country, 
    anno_oper, 
    sum(litri_tot) as litri_tot,
   EXTRACT(quarter FROM data_oper) AS stagione
from 
    dati_predizione
where
    tipo_causale = 'Vendita' and anno_oper >= '2010' 
group by 
	anno_oper,  
	EXTRACT(quarter FROM data_oper),
	 cod_naz, 
	 country
"""

# Query realizzata per semestre, ed è quella che verrà utilizzata
query_dati_vendite = """
select distinct 
    cod_naz,
    country, 
    anno_oper, 
    sum(litri_tot) AS litri_tot,
    CASE WHEN EXTRACT(MONTH FROM data_oper) < 7 then  '01'
    else '02' end  AS semestre
from 
    dati_predizione
where
    tipo_causale = 'Vendita' and anno_oper >= '2010' 
group by 
	anno_oper,  
	CASE 
        WHEN EXTRACT(MONTH FROM data_oper) < 7 THEN '01'
        ELSE '02'
    END, 
    cod_naz, 
    country
"""

# Percorso da inserire se si runna il codice da qui
filepath_0 = '../postgres.ini'

# Percorso da inserire se si runna il codice dal main
filepath = 'postgres.ini'

try:
    # Creazione del motore per la connessione al database PostgreSQL
    engine_pos = create_engine(connessione_postgres_p.create_connection_string(filepath))

    # Utilizza un contesto with per garantire la chiusura della connessione
    with engine_pos.connect() as conn_pos:
        if conn_pos is None:
            print("Connessione a PostgreSQL non stabilita.")
        else:
            # Esegui la query per ottenere i dati delle vendite dall'azienda dal database
            df_azienda_vendite = pd.read_sql(query_dati_vendite, conn_pos)

except Exception as e:
    # Gestione delle eccezioni nel caso si verifichi un errore durante la connessione o l'esecuzione della query
    print(f"Si è verificato un errore: {e}")

else:
    # Stampare le colonne solo se il blocco try è stato eseguito con successo
    print(df_azienda_vendite.columns)
    # print(df_azienda_vendite)

# Creiamo un nuovo DataFrame df_vendite_selezionato
df_vendite_selezionato = pd.DataFrame(df_azienda_vendite, columns=['cod_naz','country', 'anno_oper', 'semestre' ,'litri_tot'])

# Controllo dei valori nulli per df_vendite_selezionato
rows_with_nan = df_vendite_selezionato[df_vendite_selezionato.isnull().any(axis=1)]
print('righe con valori nan: ', rows_with_nan)

# Controllo valori unici nella colonna country
print(rows_with_nan['country'].unique())

# Creazione di due array di country con distribuzioni simili di litri_tot
country_mln = ['ITALY', 'SWEDEN', 'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND', 'AUSTRALIA', 'UNITED STATES OF AMERICA']
country_mgl = ['CANADA', 'SWITZERLAND', 'DENMARK', 'SPAIN', 'FRANCE', 'JAPAN', 'SINGAPORE', 'BELGIUM', 'ESTONIA','NETHERLANDS (KINGDOM OF THE)', 'NEW ZEALAND']

# Seleziona solo le righe del DataFrame originale che corrispondono ai paesi con il consumo sopra il milione
df_vendite_ml = df_vendite_selezionato[df_vendite_selezionato['country'].isin(country_mln)]
# Raggruppa per 'anno_oper', 'semestre', 'cod_naz', e 'country' e calcola la somma dei litri totali per ogni gruppo
df_vendite_mln = df_vendite_ml.groupby(['anno_oper', 'semestre', 'cod_naz', 'country'])['litri_tot'].apply(lambda x : x.sum()).reset_index()
print(df_vendite_mln.columns)
print('DF con litri sopra il milione: ', df_vendite_mln)
df_vendite_mln.hist(figsize=(20, 15))
# Visualizza gli istogrammi delle colonne del DataFrame
plt.show()

# Seleziona solo le righe del DataFrame originale che corrispondono ai paesi con il consumo sotto il milione
df_vendite_mg = df_vendite_selezionato[df_vendite_selezionato['country'].isin(country_mgl)]
# Raggruppa per 'anno_oper', 'semestre', 'cod_naz', e 'country' e calcola la somma dei litri totali per ogni gruppo
df_vendite_mgl = df_vendite_mg.groupby(['anno_oper', 'semestre', 'cod_naz', 'country'])['litri_tot'].apply(lambda x : x.sum()).reset_index()
print(df_vendite_mgl.columns)
print('DF con litri sopra il milione: ', df_vendite_mgl)
df_vendite_mgl.hist(figsize=(20, 15))
# Visualizza gli istogrammi delle colonne del DataFrame
plt.show()

# Estrarre la colonna 'litri_tot' dal DataFrame df_vendite_mln
df_vend_mln = df_vendite_mln["litri_tot"]
print(df_vend_mln)

# Forma della distribuzione vendite
sns.displot(df_vend_mln)
plt.title('Distribuzione vendite', fontsize=16)  # Aggiungo il titolo
plt.show()
print(df_vend_mln.describe())
data_ven_mln = df_vend_mln  # Accedi alla colonna specifica
# Calcola i limiti superiore e inferiore dello scarto interquartile per i dati della colonna 'litri_tot'
up_limit, low_limit = funzioni.interquartile_range_limits(data_ven_mln)
print(up_limit, low_limit)
# Calcola i limiti dello scarto interquartile per i dati di quella colonna
print(f"Creazione del grafico per la colonna di df_vendite: {'litri_tot'}")
# Crea un grafico a dispersione per i dati di quella colonna, con i limiti dello scarto interquartile
funzioni.plot_generic_distribution(data_ven_mln, up_limit, low_limit)
# Sostituisci tutti i valori nell'array tmp_ven che sono superiori al limite superiore o inferiori al limite inferiore con il valore del limite corrispondente
tmp_ven = data_ven_mln.to_numpy()
# Conta quanti valori superano il limite superiore (outliers)
tmp_try = tmp_ven[tmp_ven > up_limit]
tmpContato = tmp_try.shape[0]
print('Valori che superano up_limit: ', tmpContato)
# Sostituisci gli outliers con i limiti superiore e inferiore
tmp_ven[tmp_ven > up_limit]  = up_limit
tmp_ven[tmp_ven < low_limit] = low_limit
# Visualizza l'istogramma dei dati dopo la rimozione degli outliers
sns.displot(tmp_ven)
plt.show()

# Seleziona solo le colonne numeriche dal DataFrame delle vendite
df_numerico_vendite_mln = df_vendite_mln.select_dtypes(include='number')
print(df_numerico_vendite_mln)

# Matrice di correlazione
corr_matrix_vendite = df_numerico_vendite_mln.corr()

# Aggiungi un titolo per la visualizzazione di litri_tot
print("Correlazione delle colonne rispetto a 'litri_tot' - Vendite di mln")
# plot vendite
plt.figure(figsize=(16, 16))
sns.heatmap(corr_matrix_vendite, annot=True)
plt.title('Matrice di Correlazione - Vendite', fontsize=16)  # Aggiungo il titolo
plt.show()
# Calcola la correlazione delle colonne rispetto alla colonna 'litri_tot' e ordina in modo decrescente
corr_vendite = corr_matrix_vendite['litri_tot'].sort_values(ascending=False)
print("Correlazione delle colonne rispetto a 'litri_tot' - Vendite")
# Visualizza la serie di correlazioni
print(corr_vendite)
df_vendite = df_numerico_vendite_mln.copy()  # Copia il DataFrame per non modificare quello originale
print(df_vendite.columns)

# Verifica le colonne disponibili in df_vendite_mln
print('Colonne in df_vendite_mln:', df_vendite_mln.columns)

# Estrarre la colonna 'litri_tot' dal DataFrame df_vendite_selezionato
df_vend_mgl = df_vendite_mgl["litri_tot"]
print(df_vend_mgl)

#Forma della distribuzione vendite
sns.displot(df_vend_mgl)
plt.title('Distribuzione vendite', fontsize=16)  # Aggiungo il titolo
plt.show()
print(df_vend_mgl.describe())
data_ven_mgl = df_vend_mgl  # Accedi alla colonna specifica
# Calcola i limiti superiore e inferiore dello scarto interquartile per i dati della colonna 'litri_tot'
up_limit, low_limit = funzioni.interquartile_range_limits(data_ven_mgl)
print(up_limit, low_limit)
# Calcola i limiti dello scarto interquartile per i dati di quella colonna
print(f"Creazione del grafico per la colonna di df_vendite: {'litri_tot'}")
# Crea un grafico a dispersione per i dati di quella colonna, con i limiti dello scarto interquartile
funzioni.plot_generic_distribution(data_ven_mgl, up_limit, low_limit)
# Sostituisci tutti i valori nell'array tmp_ven che sono superiori al limite superiore o inferiori al limite inferiore con il valore del limite corrispondente
tmp_ven_mgl = data_ven_mgl.to_numpy()
# Conta quanti valori superano il limite superiore (outliers)
tmp_try = tmp_ven_mgl[tmp_ven_mgl > up_limit]
tmpContato = tmp_try.shape[0]
print('Valori che superano up_limit: ', tmpContato)
# Sostituisci gli outliers con i limiti superiore e inferiore
tmp_ven_mgl[tmp_ven_mgl > up_limit]  = up_limit
tmp_ven_mgl[tmp_ven_mgl < low_limit] = low_limit
# Visualizza l'istogramma dei dati dopo la rimozione degli outliers
sns.displot(tmp_ven_mgl)
plt.show()

# Seleziona solo le colonne numeriche dal DataFrame delle vendite
df_numerico_vendite_mgl = df_vendite_mgl.select_dtypes(include='number')
print(df_numerico_vendite_mgl)

# Matrice di correlazione
corr_matrix_vendite_mgl = df_numerico_vendite_mgl.corr()

# Aggiungi un titolo per la visualizzazione di litri_tot
print("Correlazione delle colonne rispetto a 'litri_tot' - Vendite")
# plot vendite
plt.figure(figsize=(16, 16))
sns.heatmap(corr_matrix_vendite_mgl, annot=True)
plt.title('Matrice di Correlazione - Vendite', fontsize=16)  # Aggiungo il titolo
plt.show()
# Calcola la correlazione delle colonne rispetto alla colonna 'litri_tot' e ordina in modo decrescente
corr_vendite_mgl = corr_matrix_vendite_mgl['litri_tot'].sort_values(ascending=False)
print("Correlazione delle colonne rispetto a 'litri_tot' - Vendite")
# Visualizza la serie di correlazioni
print(corr_vendite_mgl)
df_vendite_mig = df_numerico_vendite_mgl.copy()  # Copia il DataFrame per non modificare quello originale
print(df_vendite_mig.columns)

# Verifica le colonne disponibili in df_vendite_mln
print('Colonne in df_vendite_mln:', df_vendite_mgl.columns)

# Raggruppa i dati delle vendite per paese e anno di operazione, sommando le colonne per litri_tot
df_vendite_grouped = df_vendite_mln.groupby(['cod_naz', 'country', 'anno_oper', 'semestre'])[['litri_tot']].apply(lambda x : x.astype(int).sum()).reset_index()
df_vendite_grouped_mgl = df_vendite_mgl.groupby(['cod_naz', 'country', 'anno_oper', 'semestre'])[['litri_tot']].apply(lambda x : x.astype(int).sum()).reset_index()

print("Vendite mln raggruppate per cod_naz e anno:")
print(df_vendite_grouped)
print("Vendite mgl raggruppate per cod_naz e anno:")
print(df_vendite_grouped_mgl)

# Ottieni i nomi unici dei paesi presenti nel DataFrame df_country
unique_countries = df_vendite_grouped['country'].unique()
unique_countries_mgl = df_vendite_grouped_mgl['country'].unique()

# Codifica i valori usando OneHotEncoder
hot_encoder = OneHotEncoder(sparse_output=False)
hot_encoded = hot_encoder.fit_transform(df_vendite_grouped['country'].values.reshape(-1, 1))
hot_encoded_mgl = hot_encoder.fit_transform(df_vendite_grouped_mgl['country'].values.reshape(-1, 1))

# Creazione del nuovo DataFrame con i nomi dei paesi come nomi delle colonne
country_df_encoded = pd.DataFrame(hot_encoded, columns=unique_countries)
country_df_encoded_mgl = pd.DataFrame(hot_encoded_mgl, columns=unique_countries_mgl)

# Aggiungi il nuovo DataFrame al DataFrame originale
country_df_encoded = pd.concat([df_vendite_grouped.reset_index(drop=True), country_df_encoded], axis=1)
country_df_encoded_mgl = pd.concat([df_vendite_grouped_mgl.reset_index(drop=True), country_df_encoded_mgl], axis=1)

print('CATEGORIZZATO VENDITE mln: ', country_df_encoded)
print('CATEGORIZZATO VENDITE mgl: ', country_df_encoded_mgl)

# Controllo dei valori nulli per mln
rows_with_nan_2 = country_df_encoded[country_df_encoded.isnull().any(axis=1)]
print('righe con valori nan per mln: ', rows_with_nan_2)

# Controllo dei valori nulli per mgl
rows_with_nan = country_df_encoded_mgl[country_df_encoded_mgl.isnull().any(axis=1)]
print('righe con valori nan per mgl: ', rows_with_nan)

# Seleziona le colonne numeriche dal DataFrame country_df_encoded
df_uiu = country_df_encoded.sort_values('anno_oper', ascending=True)

print(df_uiu)

# Seleziona le features X rimuovendo la colonna 'litri_tot'
X = df_uiu.drop(['litri_tot', 'country'], axis=1)
print(X.columns)
# Seleziona il target Y dalla colonna 'litri_tot'
Y = df_uiu['litri_tot']

# test 1
funzioni.pipeline_validation(make_pipeline(StandardScaler(), LinearRegression()), X, Y)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), LinearRegression()), X, Y)

# test 2
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X, Y)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X, Y)

# Valida un modello di regressione lineare polinomiale con cross-validation k-fold
funzioni.pipeline_poly_validation(LinearRegression(), X, Y, 4)

# Valida un modello di regressione Ridge polinomiale con cross-validation k-fold
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 4), Ridge(alpha=10, fit_intercept=False)), X, Y)
# Plotta la curva di apprendimento per il modello di regressione Ridge polinomiale
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 4), Ridge(alpha=10, fit_intercept=False)), X, Y)

# Utilizza la funzione pipeline_poly_validation per validare un modello di regressione Ridge polinomiale
funzioni.pipeline_poly_validation(Ridge(), X, Y, 5)

# Testa un modello di regressione Ridge
funzioni.test_pipeline_ridge(X, Y, 3, 10)

# Ottimizza i parametri del modello di regressione Ridge
poly_degree, ridge_alpha, ridge_intercept = funzioni.ottimization(X, Y)

# Testa il modello utilizzando i dati di input X e il target Y
test_model(X, Y)

# Seleziona le colonne numeriche dal DataFrame country_df_encoded_mgl
df_uiu_mgl = country_df_encoded_mgl.sort_values('anno_oper', ascending=True)

print('Inizio predizione per mgl', df_uiu_mgl)

# Seleziona le features X_mgl rimuovendo la colonna 'litri_tot'
X_mgl = df_uiu_mgl.drop(['litri_tot', 'country'], axis=1)
print(X_mgl.columns)
# Seleziona il target Y_mgl dalla colonna 'litri_tot'
Y_mgl = df_uiu_mgl['litri_tot']

# test 1
funzioni.pipeline_validation(make_pipeline(StandardScaler(), LinearRegression()), X_mgl, Y_mgl)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), LinearRegression()), X_mgl, Y_mgl)

# test 2
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X_mgl, Y_mgl)
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()), X_mgl, Y_mgl)

# Valida un modello di regressione lineare polinomiale con cross-validation k-fold
funzioni.pipeline_poly_validation(LinearRegression(), X_mgl, Y_mgl, 4)

# Valida un modello di regressione Ridge polinomiale con cross-validation k-fold
funzioni.pipeline_validation(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 2), Ridge(alpha=10, fit_intercept=True)), X_mgl, Y_mgl)
# Plotta la curva di apprendimento per il modello di regressione Ridge polinomiale
funzioni.plot_learning_curve(make_pipeline(StandardScaler(), PolynomialFeatures(degree = 2), Ridge(alpha=10, fit_intercept=True)), X_mgl, Y_mgl)

# Utilizza la funzione pipeline_poly_validation per validare un modello di regressione Ridge polinomiale
funzioni.pipeline_poly_validation(Ridge(), X_mgl, Y_mgl, 5)

# Testa un modello di regressione Ridge
funzioni.test_pipeline_ridge(X_mgl, Y_mgl, 3, 10)

# Ottimizza i parametri del modello di regressione Ridge
poly_degree_mgl, ridge_alpha_mgl, ridge_intercept_mgl = funzioni.ottimization(X_mgl, Y_mgl)

# Testa il modello utilizzando i dati di input X_mgl e il target Y_mgl
test_model(X_mgl, Y_mgl)

print('Fine predizione per mgl')

df_vendite_s = pd.concat([df_uiu, df_uiu_mgl], ignore_index= True)

colonne_da_rimuovere = np.concatenate((country_mgl, country_mln))
df_vendite_s = df_vendite_s.drop(columns=colonne_da_rimuovere)

# Check if 'anno_oper' column exists in the DataFrame
if 'anno_oper' in df_vendite_s.columns:
    # Cast 'anno_oper' to integer type
    df_vendite_s['anno_oper'] = df_vendite_s['anno_oper'].astype(int)
    # Filter rows where 'anno_oper' is less than or equal to 2020
    df_vendite_s = df_vendite_s[df_vendite_s['anno_oper'] <= 2019]
else:
    print("Column 'anno_oper' not found in DataFrame.")

def save_model_with_pickle(df, nome_file):
    """
    Salva il modello addestrato utilizzando pickle.

    Args:
        df (DataFrame): Il DataFrame contenente i dati di addestramento.

    Returns:
        None
    """
    # Creazione delle features e target
    X = df.drop(["litri_tot", 'country'], axis=1)
    Y = df["litri_tot"]

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
    with open("prediction/" + nome_file, "wb") as file:
        pickle.dump(model_pipeline, file)


# Chiamata alla funzione per salvare il modello, i modelli sono stati salvati quindi va bene lasciare così
# save_model_with_pickle(df_uiu, 'modello_vendite_mln.pkl')
# save_model_with_pickle(df_uiu_mgl, 'modello_vendite_mgl.pkl')

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

# Fase di testing per controllare se entrambi i modelli predicono in maniera abbastanza accurata basandosi su dati storici

print('Testing modello mln')
# Esempio di caricamento del modello
loaded_model_mln = load_model_with_pickle('modello_vendite_mln.pkl')
# Fai previsioni utilizzando il modello caricato
Y_pred = loaded_model_mln.predict(df_uiu.drop(['litri_tot', 'country'], axis=1))
print(Y_pred)
# Aggiungi la colonna 'litri_tot_pred' contenente i valori predetti al DataFrame df_uiu
df_uiu['litri_pred'] = Y_pred
# Formatta i valori nella colonna 'litri_tot_pred' senza notazione scientifica
df_uiu['litri_pred'] = df_uiu['litri_pred'].apply(lambda x: '{:.0f}'.format(x))
# Convertire la colonna 'litri_tot_pred' in numerico
df_uiu['litri_pred'] = pd.to_numeric(df_uiu['litri_pred'], errors='coerce')
# Filtrare il DataFrame per escludere i record con litri_tot_pred < 0
df_uiu_filtered = df_uiu[df_uiu['litri_pred'] >= 0]
df_uiu_filtered.loc[:, 'accuratezza_percentuale'] = (df_uiu_filtered['litri_pred']  * 100) / df_uiu_filtered['litri_tot']
df_mln_filtered = df_uiu_filtered.drop(columns=country_mln)
print('Output predizione per mln',df_mln_filtered.to_string())

print('Testing modello mgl')
# Esempio di caricamento del modello
loaded_model_mgl = load_model_with_pickle('modello_vendite_mgl.pkl')
# Fai previsioni utilizzando il modello caricato
Y_pred_mgl = loaded_model_mgl.predict(df_uiu_mgl.drop(['litri_tot', 'country'], axis=1))
# print(Y_pred_)
# Aggiungi la colonna 'litri_tot_pred' contenente i valori predetti al DataFrame df_uiu_mgl
df_uiu_mgl['litri_pred'] = Y_pred_mgl
# Formatta i valori nella colonna 'litri_tot_pred' senza notazione scientifica
df_uiu_mgl['litri_pred'] = df_uiu_mgl['litri_pred'].apply(lambda x: '{:.0f}'.format(x))
# Convertire la colonna 'litri_tot_pred' in numerico
df_uiu_mgl['litri_pred'] = pd.to_numeric(df_uiu_mgl['litri_pred'], errors='coerce')
# Filtrare il DataFrame per escludere i record con litri_tot_pred < 0
df_uiu_mgl_filtered = df_uiu_mgl[df_uiu_mgl['litri_pred'] >= 0]
df_uiu_mgl_filtered.loc[:, 'accuratezza_percentuale'] = (df_uiu_mgl_filtered['litri_pred']  * 100) / df_uiu_mgl_filtered['litri_tot']
df_mgl_filtered = df_uiu_mgl_filtered.drop(columns=country_mgl)
print('Output predizione per mgl',df_mgl_filtered.to_string())


# Genera una lista di anni da 2020 a 2025
anni_da_aggiungere = list(range(2021, 2028))
semestri_da_aggiungere = [str(i).zfill(2) for i in range(1, 3)]

# Crea una copia del DataFrame originale per aggiungere le righe espandendo gli anni
df_espanso_mln = df_uiu.copy()
df_espanso_mgl = df_uiu_mgl.copy()

# Itera attraverso i nuovi anni e aggiungi una riga per ciascun anno per ogni nazione
for anno in anni_da_aggiungere:
    # Per ogni anno, duplica il DataFrame originale, imposta l'anno appropriato e concatena
    df_anno = df_uiu.copy()
    df_anno['anno_oper'] = anno

    # Aggiungi i mesi al DataFrame per ogni anno
    for semestre in semestri_da_aggiungere:
        # Duplica il DataFrame dell'anno corrente
        df_semestre = df_anno.copy()

        # Imposta il semestre corrente nel DataFrame duplicato
        df_semestre.loc[:, 'semestre'] = semestre

        # Concatena il DataFrame del semestre al DataFrame espanso
        df_espanso_mln = pd.concat([df_espanso_mln, df_semestre], ignore_index=True)

# Rimozione dei duplicati dopo l'aggiunta dei semestri
df_espanso_mln = df_espanso_mln.drop_duplicates()

# Visualizza il DataFrame per verificare se i duplicati sono stati rimossi correttamente
print(df_espanso_mln)

df_espanso_mln['anno_oper'] = df_espanso_mln['anno_oper'].astype(int)
df_espanso_mln = df_espanso_mln.drop_duplicates()
df_espanso_mln = df_espanso_mln[df_espanso_mln['anno_oper'] >= 2020]

# Mantieni solo le colonne utilizzate durante l'addestramento del modello
df_espanso_mln = df_espanso_mln[['cod_naz', 'country','anno_oper', 'semestre', 'AUSTRALIA',
       'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND', 'ITALY',
       'SWEDEN', 'UNITED STATES OF AMERICA']]

# Ordina il DataFrame per 'anno_oper'
df_espanso_mln = df_espanso_mln.sort_values(by=['anno_oper'])

df_test_mln = df_espanso_mln.copy()

# Visualizza il DataFrame df_test_mln dopo le modifiche
print(df_test_mln.columns)

predizioni_mln = loaded_model_mln.predict(df_test_mln.drop('country', axis=1))

# Aggiungiamo predizioni come colonna al DataFrame df_test_mln
df_test_mln['litri_tot'] = predizioni_mln
df_test_mln['litri_tot'] = df_test_mln['litri_tot'].apply(lambda x: '{:.0f}'.format(x))

# Droppio le colonne create con il metodo di one hot-encoder
df_test_mln = df_test_mln.drop(columns=country_mln)

# Convertire la colonna 'litri_tot' in numerico
df_test_mln['litri_tot'] = pd.to_numeric(df_test_mln['litri_tot'], errors='coerce')

# Controllo i duplicati
df_deduplicato_mln = df_test_mln.drop_duplicates(subset=['cod_naz', 'semestre', 'anno_oper'])

# Filtrare il DataFrame per escludere i record con litri_tot_pred < 0
df_vendite_pred_mln = df_deduplicato_mln[df_deduplicato_mln['litri_tot'] >= 0]

# Dovrebbe essere pronto per essere caricato sul db se va bene una accuratezza del 80%
print('PREDIZIONI PER MLN, lo score da training ecc è 0.8120901305943985', df_vendite_pred_mln.columns)
print(df_vendite_pred_mln.sort_values(['country', 'semestre', 'anno_oper']).to_string())

# Ripeto gli stessi passaggi per mgl
# Itera attraverso i nuovi anni e aggiungi una riga per ciascun anno per ogni nazione
for anno in anni_da_aggiungere:
    # Per ogni anno, duplica il DataFrame originale, imposta l'anno appropriato e concatena
    df_anno = df_uiu_mgl.copy()
    df_anno['anno_oper'] = anno

    # Aggiungi i mesi al DataFrame per ogni anno
    for semestre in semestri_da_aggiungere:
        # Duplica il DataFrame dell'anno corrente
        df_semestre = df_anno.copy()

        # Imposta il mese corrente nel DataFrame duplicato
        df_semestre.loc[:, 'semestre'] = semestre

        # Concatena il DataFrame del mese al DataFrame espanso
        df_espanso_mgl = pd.concat([df_espanso_mgl, df_semestre], ignore_index=True)

# Rimozione dei duplicati dopo l'aggiunta dei mesi
df_espanso_mgl = df_espanso_mgl.drop_duplicates()

# Visualizza il DataFrame per verificare se i duplicati sono stati rimossi correttamente
print(df_espanso_mgl)

df_espanso_mgl['anno_oper'] = df_espanso_mgl['anno_oper'].astype(int)
df_espanso_mgl = df_espanso_mgl.drop_duplicates()
df_espanso_mgl = df_espanso_mgl[df_espanso_mgl['anno_oper'] >= 2020]

# Mantieni solo le colonne utilizzate durante l'addestramento del modello
df_espanso_mgl = df_espanso_mgl[['cod_naz','country', 'anno_oper', 'semestre', 'BELGIUM', 'CANADA', 'SWITZERLAND',
       'DENMARK', 'ESTONIA', 'SPAIN', 'FRANCE', 'JAPAN',
       'NETHERLANDS (KINGDOM OF THE)', 'NEW ZEALAND', 'SINGAPORE']]

# Ordina il DataFrame per 'anno_oper'
df_espanso_mgl = df_espanso_mgl.sort_values(by=['anno_oper'])

df_test_mgl = df_espanso_mgl.copy()

# Visualizza il DataFrame df_test dopo le modifiche
print(df_test_mgl.columns)

predizioni_mgl = loaded_model_mgl.predict(df_test_mgl.drop('country', axis=1))

# Aggiungiamo predizioni come colonna al DataFrame df_test_mgl
df_test_mgl['litri_tot'] = predizioni_mgl

df_test_mgl['litri_tot'] = df_test_mgl['litri_tot'].apply(lambda x: '{:.0f}'.format(x))

df_test_mgl = df_test_mgl.drop(columns=country_mgl)

# Convertire la colonna 'litri_tot' in numerico
df_test_mgl['litri_tot'] = pd.to_numeric(df_test_mgl['litri_tot'], errors='coerce')

df_deduplicato_mgl = df_test_mgl.drop_duplicates(subset=['cod_naz', 'semestre', 'anno_oper'])

# Filtrare il DataFrame per escludere i record con litri_tot_pred < 0
df_vendite_pred_mgl = df_deduplicato_mgl[df_deduplicato_mgl['litri_tot'] >= 0]

# Dovrebbe essere pronto per essere caricato sul db se va bene una accuratezza del 80%
print('PREDIZIONI PER MGL, lo score da training è 0.7584335440734412',df_vendite_pred_mgl.columns)
print(df_vendite_pred_mgl.sort_values(['country', 'semestre', 'anno_oper']).to_string())

# Ora che ci sono le predizioni per entrambi i df fino al 2027 li concateno per poi caricarli sul db
df_vendite_predizione = pd.concat([df_vendite_pred_mln, df_vendite_pred_mgl], ignore_index= True)

df_vendite_pred = pd.concat([df_vendite_s, df_vendite_predizione], ignore_index=True)

print('Dataframe concatenato per mln e mlg', df_vendite_pred)