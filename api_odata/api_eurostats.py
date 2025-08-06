import requests

# API una volta funzionanti ma la struttura del json viene estratta in modo scomodo

# URL del JSON
json_url = 'https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/DS-056120?format=JSON&lang=en&freq=A&decl=600&decl=2027&decl=001&decl=003&decl=004&decl=005&decl=006&decl=007&decl=008&decl=009&decl=010&decl=011&decl=017&decl=018&decl=024&decl=028&decl=030&decl=032&decl=038&decl=046&decl=053&decl=054&decl=055&decl=060&decl=061&decl=063&decl=064&decl=066&decl=068&decl=091&decl=092&decl=093&decl=096&decl=097&decl=098&prccode=11051000&prccode=11051010&indicators=IMPVAL&indicators=QNTUNIT&indicators=OWNPRODVAL&indicators=EXPQNT&indicators=PRODVAL&indicators=EXPVAL&indicators=PRODQNT&indicators=OWNPRODQNT&indicators=IMPQNT&time=2012&time=2013&time=2014&time=2015&time=2016&time=2017&time=2018&time=2019&time=2020&time=2021&time=2022&time=2023&time=2024'

# Percorso del file JSON da salvare
json_file_path_0 = "../file_csv/dati_ue.json"
json_file_path = "file_csv/dati_ue.json"

# Effettua la richiesta al server per ottenere il JSON
response = requests.get(json_url)

# Verifica se la richiesta è andata a buon fine
if response.status_code == 200:
    # Salva il contenuto JSON direttamente nel file
    with open(json_file_path, "w") as json_file:
        json_file.write(response.text)
else:
    print(f"La richiesta del file JSON ha Codice di stato: {response.status_code}")
