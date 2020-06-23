import click
import glob
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

# PASADO:      TempP0, TempP1, ..., TempP70, TempP71            (nx168)
# REAL:        TempF71, TempP72,..., TempP95                    (nx24)


def crear_dataset(df_pasado):
    matriz_pasado = crear_pasado(df_pasado)
    matriz_real = crear_real(df_pasado)
    # concatenar matrices
    return np.concatenate((matriz_pasado, matriz_real), axis=1)


def crear_pasado(df_pasado):
    matriz_pasado = np.empty((len(df_pasado) - 168 - 24, 168))  # matriz vacia de len(df) x 168
    for i in range(0, len(df_pasado) - 168 - 24):
        for j in range(0, 168):
            matriz_pasado[i, j] = df_pasado.iloc[i + j, 2]  # coger columna 2 que es temperatura
    return matriz_pasado

def crear_real(df_pasado):
    matriz_real = np.empty((len(df_pasado) - 168 - 24, 24))  # matriz vacia de len(df) x 24
    for i in range(168, len(df_pasado) - 24):
        for j in range(0, 24):
            matriz_real[i - 168, j] = df_pasado.iloc[i + j, 2]  # coger columna 2 que es temperatura
    return matriz_real


@click.command()
@click.option('--train', prompt='Directorio de dataset train: ', default='.\\train')
@click.option('--mmscaler', prompt='Ruta del escalador de temperaturas: ', default='.\\mmscaler.pickle')
@click.option('--out', prompt='Ruta para guardar el dataset: ', default='.\\dataset_generalista.csv')

def main(train, mmscaler, out):
    lista_train = glob.glob(train + '\\ch*.csv')
    print("lista_train: {}".format(lista_train))
    headers = ['fecha', 'id', 'temp']
    dtypes = {'fecha': 'str', 'id': 'float', 'temp': 'float'}
    parse_dates = ['fecha']

    train = np.empty((0, 192))
    for f in lista_train:
        df_train = pd.DataFrame()  # Vaciar dataframe
        df_train = pd.read_csv(f, decimal=".", sep=",", header=None, names=headers, dtype=dtypes, parse_dates=parse_dates)
        datos_estacion = crear_dataset(df_train)
        # adjuntar matrices
        print("Estacion: {}, tamaño dataset antes: {}".format(
            f, train.shape))
        train = np.append(train, datos_estacion, axis=0)
        print("Estacion: {}, tamaño dataset despues: {}".format(f, train.shape))

    # Normalizar temperatura
    temp_scaler = pickle.load(open(mmscaler, 'rb'))
    print(temp_scaler)
    train = temp_scaler.transform(train)

    # Guardar el dataset
    pd.DataFrame(train).to_csv(out, header=None, index=None)


if __name__ == "__main__":
    main()
