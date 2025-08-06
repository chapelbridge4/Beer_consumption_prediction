import pandas as pd

# In questo codice creo due array codici e descrizioni che poi andranno ad essere caricati in una tabella sul db che funga da tramite tra più tabelle del db

codici = ['AE', 'AL', 'AR', 'AT', 'AU', 'BE', 'BG', 'BH', 'BM', 'BRA', 'CA', 'CH', 'CL', 'CN', 'CR', 'CV', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES',
          'FI', 'FR', 'GB', 'HK', 'HR', 'HU', 'IE', 'IL', 'IS', 'IT', 'JP', 'KH', 'KR', 'LT', 'LU', 'LV', 'MV','MX', 'MY', 'NL', 'NO', 'NZ', 'PA', 'PH',
          'PL', 'PT', 'PY', 'RO', 'RU', 'SC', 'SE', 'SG', 'SM', 'SX', 'TH', 'TT', 'TW', 'UA', 'US', 'UY', 'ZA', 'MA', 'ME', 'MK', 'RS', 'SK', 'SL']

descrizioni = ['UNITED ARAB EMIRATES', 'ALBANIA', 'ARGENTINA', 'AUSTRIA', 'AUSTRALIA', 'BELGIUM', 'BULGARIA', 'BAHRAIN', 'BERMUDA', 'BRAZIL', 'CANADA', 'SWITZERLAND', 'CHILE', 'CHINA',  'COSTA RICA',
                'CAPE VERDE', 'CZECHIA', 'GERMANY', 'DENMARK', 'ESTONIA',  'GREECE', 'SPAIN', 'FINLAND', 'FRANCE', 'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND', 'HONG KONG', 'CROATIA',
                'HUNGARY', 'IRELAND', 'ISRAEL', 'ICELAND', 'ITALY', 'JAPAN', 'CAMBODIA', 'REPUBLIC OF KOREA', 'LITHUANIA', 'LUXEMBOURG', 'LATVIA', 'MALDIVES', 'MEXICO', 'MALAYSIA', 'NETHERLANDS (KINGDOM OF THE)',
                'NORWAY', 'NEW ZEALAND', 'PANAMA', 'PHILIPPINES', 'POLAND', 'PORTUGAL', 'PARAGUAY', 'ROMANIA', 'RUSSIAN FEDERATION', 'SEYCHELLES', 'SWEDEN', 'SINGAPORE', 'REPUBLIC OF SAN MARINO', 'SINT MAARTEN DUTCH PART',
               'THAILANDIA', 'TRINIDAD AND TOBAGO', 'TAIWAN', 'UKRAINE', 'UNITED STATES OF AMERICA', 'URUGUAY', 'SOUTH AFRICA', 'MALTA', 'MONTENEGRO', 'NORTH MACEDONIA', 'SERBIA', 'SLOVAKIA', 'SLOVENIA']

longitudini = [
    53.8478,  # UNITED ARAB EMIRATES
    20.1683,  # ALBANIA
    -63.6167,  # ARGENTINA
    14.5501,  # AUSTRIA
    133.7751,  # AUSTRALIA
    4.4699,  # BELGIUM
    23.4857,  # BULGARIA
    50.586,  # BAHRAIN
    -64.7546,  # BERMUDA
    -51.9253,  # BRAZIL
    -106.3468,  # CANADA
    8.2275,  # SWITZERLAND
    -71.5375,  # CHILE
    104.1954,  # CHINA
    -84.0907,  # COSTA RICA
    -23.0418,  # CAPE VERDE
    15.4749,  # CZECHIA
    10.4515,  # GERMANY
    9.5018,  # DENMARK
    25.0136,  # ESTONIA
    21.8243,  # GREECE
    -3.7492,  # SPAIN
    2.2137,  # FINLAND
    2.2137,  # FRANCE
    -3.436,  # UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND
    114.1095,  # HONG KONG
    15.2000,  # CROATIA
    19.5033,  # HUNGARY
    -7.6921,  # IRELAND
    34.8516,  # ISRAEL
    -18.5795,  # ICELAND
    12.5674,  # ITALY
    138.2529,  # JAPAN
    104.9903,  # CAMBODIA
    127.7669,  # REPUBLIC OF KOREA
    23.8813,  # LITHUANIA
    6.1296,  # LUXEMBOURG
    24.6032,  # LATVIA
    73.2207,  # MALDIVES
    -102.5528,  # MEXICO
    101.9758,  # MALAYSIA
    4.4699,  # NETHERLANDS (KINGDOM OF THE)
    8.4689,  # NORWAY
    174.8859,  # NEW ZEALAND
    -80.7821,  # PANAMA
    121.774,  # PHILIPPINES
    19.1451,  # POLAND
    -8.2245,  # PORTUGAL
    -58.4438,  # PARAGUAY
    24.9668,  # ROMANIA
    105.3188,  # RUSSIAN FEDERATION
    55.4915,  # SEYCHELLES
    18.6435,  # SWEDEN
    103.8198,  # SINGAPORE
    12.4578,  # REPUBLIC OF SAN MARINO
    -63.0578,  # SINT MAARTEN DUTCH PART
    100.9925,  # THAILAND
    -61.2225,  # TRINIDAD AND TOBAGO
    121.5654,  # TAIWAN
    31.1656,  # UKRAINE
    -95.7129,  # UNITED STATES OF AMERICA
    -55.7658,  # URUGUAY
    24.9916,  # SOUTH AFRICA
    14.3754,  # MALTA
    19.3744,  # MONTENEGRO
    21.7453,  # NORTH MACEDONIA
    21.0059,  # SERBIA
    19.699,  # SLOVAKIA
    14.9955  # SLOVENIA
]

latitudini = [
    23.4241,  # UNITED ARAB EMIRATES
    41.1533,  # ALBANIA
    -38.4161,  # ARGENTINA
    47.5162,  # AUSTRIA
    -25.2744,  # AUSTRALIA
    50.5039,  # BELGIUM
    42.7339,  # BULGARIA
    26.0667,  # BAHRAIN
    32.3078,  # BERMUDA
    -14.235,  # BRAZIL
    56.1304,  # CANADA
    46.8182,  # SWITZERLAND
    -35.6751,  # CHILE
    35.8617,  # CHINA
    9.7489,  # COSTA RICA
    16.5388,  # CAPE VERDE
    49.8175,  # CZECHIA
    51.1657,  # GERMANY
    56.2639,  # DENMARK
    58.5953,  # ESTONIA
    39.0742,  # GREECE
    40.4637,  # SPAIN
    61.9241,  # FINLAND
    46.6034,  # FRANCE
    55.3781,  # UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND
    22.3193,  # HONG KONG
    45.1,  # CROATIA
    47.1625,  # HUNGARY
    53.4129,  # IRELAND
    31.0461,  # ISRAEL
    64.9631,  # ICELAND
    41.8719,  # ITALY
    36.2048,  # JAPAN
    12.5657,  # CAMBODIA
    35.9078,  # REPUBLIC OF KOREA
    55.1694,  # LITHUANIA
    49.8153,  # LUXEMBOURG
    56.8796,  # LATVIA
    3.2028,  # MALDIVES
    23.6345,  # MEXICO
     4.2105,  # MALAYSIA
    52.1326,  # NETHERLANDS (KINGDOM OF THE)
    60.472,  # NORWAY
    -40.9006,  # NEW ZEALAND
    8.538,  # PANAMA
    12.8797,  # PHILIPPINES
    51.9194,  # POLAND
    39.3999,  # PORTUGAL
    -23.4425,  # PARAGUAY
    45.9432,  # ROMANIA
    61.524,  # RUSSIAN FEDERATION
    -4.6796,  # SEYCHELLES
    60.1282,  # SWEDEN
    1.3521,  # SINGAPORE
    43.9424,  # REPUBLIC OF SAN MARINO
    18.0425,  # SINT MAARTEN DUTCH PART
    15.8700,  # THAILAND
    10.6918,  # TRINIDAD AND TOBAGO
    23.6978,  # TAIWAN
    48.3794,  # UKRAINE
    37.0902,  # UNITED STATES OF AMERICA
    -32.5228,  # URUGUAY
    -30.5595,  # SOUTH AFRICA
    35.9375,  # MALTA
    42.7087,  # MONTENEGRO
    41.6086,  # NORTH MACEDONIA
    44.0165,  # SERBIA
    48.669,  # SLOVAKIA
    46.1512  # SLOVENIA
]


# # Stampa le lunghezze degli array
# print("Lunghezza di codici:", len(codici))
# print("Lunghezza di descrizioni:", len(descrizioni))
#
# # Trova gli elementi presenti in codici ma non in descrizioni
# codici_set = set(codici)
# descrizioni_set = set(descrizioni)
#
# codici_mancanti = codici_set - descrizioni_set
# print("Codici mancanti:", codici_mancanti)

#
# # Continua con l'elaborazione o la stampa degli array
#
# if len(record_to_insert_naz['codice']) != len(record_to_insert_naz['descrizione']):
#     raise ValueError("Le liste 'codice' e 'descrizione' devono avere la stessa lunghezza.")

df_nazioni = pd.DataFrame({'codice': codici, 'descrizione': descrizioni, 'longitudine': longitudini, 'latitudine': latitudini})

print(df_nazioni)