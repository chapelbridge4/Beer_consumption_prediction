from sql_server.sql import truncate_all_tables, extract, load_odata, load_nazioni, load_predizioni
from Postgres.db_postgres import create_tables, crea_viste

def main():

    print('Hello, there!')

    # Funzione che elimina dati se presenti nelle tabelle
    # truncate_all_tables()

    # Funzione dal modulo di postgres/db_postgres.py per creare le tabelle
    # create_tables()

    # load_nazioni()

    # Funzione del modulo sql_server/sql.py per estrarre e caricare i dati sul db di postgresql
    # extract()

    # Funzione del modulo api_odata/db_odata per estrarre e caricare i dati sul db di postgresql
    # load_odata()

    # crea viste per valori mancanti
    # crea_viste()

    # load_predizioni()

# Esegui il main solo se questo script è eseguito direttamente (non importato come modulo)

if __name__ == "__main__":
    main()


    # num persone in territori regioni non carica bene