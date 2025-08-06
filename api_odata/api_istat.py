import requests
from urllib.parse import urljoin

# API che funzionano se ISTAT non cambiasse link ogni 2 giorni

url = "https://www.istat.it/it/archivio/244222"
response = requests.get(url)

if response.status_code == 200:
    # Trova il link al file
    file_link = "/it/files//2020/06/tavole_consumo_di_alcol_2019.xls"

    # Costruisci l'URL completo del file
    file_url = urljoin(url, file_link)

    # Scarica il file
    file_content = requests.get(file_url).content

    # Salva il file
    with open("../tavole_consumo_di_alcol_2019.xls", "wb") as file:
        file.write(file_content)

    print("Download completato.")
else:
    print(f"Errore nella richiesta della pagina. Codice di stato: {response.status_code}")
