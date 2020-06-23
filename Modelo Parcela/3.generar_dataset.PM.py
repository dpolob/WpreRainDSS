import click
import glob
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

# PASADO:      TempP0, TempP1, ..., TempP70, TempP71            (nx168)
# FUTURO:      TempF71, TempF72,..., TempF95                    (nx24)
# ALTITUD:     Altitud_Estacion - Altitud_Estacion_Prediccion   (nx1)
# MES:         Onehot encoding mes del año                      (nx12)
# REAL:        TempP71, TempP72,..., TempP95                    (nx24)


def crear_dataset(df_pasado, df_forecast, altitud_estacion, altitud_forecast):
    matriz_pasado = crear_pasado(df_pasado)
    assert len(df_pasado) == len(
        df_forecast), 'La longitud de los dataset de train y forecast no coinciden{} - {}'.format(len(df_pasado), len(df_forecast))
    matriz_futuro = crear_futuro(df_forecast)
    matriz_altitud = np.repeat(altitud_estacion - altitud_forecast,
                               len(df_pasado) - 192).reshape(-1, 1)
    lb = LabelBinarizer()
    lb.fit([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    matriz_mes = lb.transform(np.array(df_forecast['fecha'].dt.month.values)[192:]).astype(int)
    matriz_real = crear_real(df_pasado)
    # concatenar matrices
    return np.concatenate((matriz_pasado, matriz_futuro, matriz_altitud, matriz_mes, matriz_real), axis=1)


def crear_pasado(df_pasado):
    matriz_pasado = np.empty((len(df_pasado) - 168 - 24, 168))  # matriz vacia de len(df) x 168
    for i in range(0, len(df_pasado) - 168 - 24):
        for j in range(0, 168):
            matriz_pasado[i, j] = df_pasado.iloc[i + j, 2]  # coger columna 2 que es temperatura
    return matriz_pasado


def crear_futuro(df_forecast):
    matriz_futuro = np.empty((len(df_forecast) - 168 - 24, 24))  # matriz vacia de len(df) x 24
    for i in range(168, len(df_forecast) - 24):
        for j in range(0, 24):
            # coger columna 2 que es temperatura
            matriz_futuro[i - 168, j] = df_forecast.iloc[i + j, 2]
    return matriz_futuro


def crear_real(df_pasado):
    matriz_real = np.empty((len(df_pasado) - 168 - 24, 24))  # matriz vacia de len(df) x 24
    for i in range(168, len(df_pasado) - 24):
        for j in range(0, 24):
            matriz_real[i - 168, j] = df_pasado.iloc[i + j, 2]  # coger columna 2 que es temperatura
    return matriz_real


@click.command()
@click.option('--train', prompt='Directorio de los datos de la estacion: ', default='.\\validation')
@click.option('--forecast', prompt='Directorio de los datos de la estacion de forescast: ', default='.\\forecast')
@click.option('--mmscaler', prompt='Ruta del escalador de temperaturas: ', default='.\\mmscaler.pickle')
@click.option('--altscaler', prompt='Ruta del escalador de alturas: ', default='.\\altscaler.pickle')
@click.option('--out', prompt='Ruta para guardar el dataset: ', default='.\\dataset.csv')

def generar_dataset(train, forecast, mmscaler, altscaler, out):
    lista_train = glob.glob(train + '\\ch*.csv')
    print("lista_train: {}".format(lista_train))
    lista_forecast = glob.glob(forecast + '\\ch*.csv')
    print("lista_forecast: {}".format(lista_forecast))

    headers = ['fecha', 'id', 'temp']
    dtypes = {'fecha': 'str', 'id': 'float', 'temp': 'float'}
    parse_dates = ['fecha']

    df_forecast = pd.DataFrame()
    for f in lista_forecast:
        df_forecast = pd.concat([df_forecast, pd.read_csv(
            f, decimal=".", sep=",", header=None, names=headers, dtype=dtypes, parse_dates=parse_dates)])
        altitud_forecast = int(f.split('A')[1])
        print("Altitud estacion forecast: {}".format(altitud_forecast))

    train = np.empty((0, 229))
    for f in lista_train:
        df_train = pd.DataFrame()  # Vaciar dataframe
        df_train = pd.read_csv(f, decimal=".", sep=",", header=None,
                               names=headers, dtype=dtypes, parse_dates=parse_dates)
        altitud_estacion = int(f.split('A')[1])
        datos_estacion = crear_dataset(df_train, df_forecast, altitud_estacion, altitud_forecast)
        # adjuntar matrices
        print("Estacion: {}, altitud estacion: {}, tamaño dataset antes: {}".format(
            f, altitud_estacion, train.shape))
        train = np.append(train, datos_estacion, axis=0)
        print("Estacion: {}, tamaño dataset despues: {}".format(f, train.shape))

    # Normalizar temperatura
    temp_scaler = pickle.load(open(mmscaler, 'rb'))
    print(temp_scaler)
    train[:, :192] = temp_scaler.transform(train[:, :192])
    train[:, 204:] = temp_scaler.transform(train[:, 204:])

    # Normalizar la altitud
    alt_scaler = pickle.load(open(altscaler, 'rb'))
    train[:, 192] = alt_scaler.transform(train[:, 192].reshape(-1, 1)).flatten()



    # Guardar el dataset
    pd.DataFrame(train).to_csv(out, header=None, index=None)


if __name__ == "__main__":
    generar_dataset()
