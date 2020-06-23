import click
import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

@click.command()
@click.option('--train', prompt='Directorio de dataset train: ')
@click.option('--forecast', prompt='Directorio de dataset forecast: ')
@click.option('--out', prompt='Ruta para guardar el escalador: ')

def crear_escalador(train, forecast, out):
    headers = ['fecha', 'id', 'temp']
    dtypes = {'fecha': 'str', 'id': 'float', 'temp': 'float'}
    parse_dates = ['fecha']

    lista_archivos = glob.glob(train + '\ch*.csv') # son todos, pero es necesario retirar el de validacion
    lista_archivos += glob.glob(forecast + '\ch*.csv')
    print("Archivos a procesar:")
    print(lista_archivos)

    df = pd.DataFrame()
    for f in lista_archivos:
        tmp = pd.read_csv(f, decimal=".", sep=",", header=None, names=headers, dtype=dtypes, parse_dates=parse_dates)
        print("Filas de {}: {}".format(f, tmp.shape[0]))
        df = pd.concat([df, tmp])
        print("Filas de df: {}".format(df.shape[0]))

    print(df.info())

	# Informacion de datos NaN
    null_columns=df.columns[df.isnull().any()]
    print("Datos NaN en el dataset: {}".format(df[df.isnull().any(axis=1)][null_columns]))

	# Crear el escalador
    mmscaler = MinMaxScaler()
    mmscaler.fit(np.array(df['temp'].values).reshape(-1,1))
    print("Datos del escalador")
    print("min_: {}, scale_: {}, data_min_: {}, data_max_: {}, data_range_: {}".format(mmscaler.min_,
                                                                                   mmscaler.scale_,
                                                                                   mmscaler.data_min_,
                                                                                   mmscaler.data_max_,
                                                                                   mmscaler.data_range_))
    pickle.dump(mmscaler, open(out, 'wb'))

if __name__ == "__main__":
    crear_escalador()
