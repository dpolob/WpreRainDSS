import pandas as pd
import numpy as np
import click

@click.command()
@click.option('--infile', prompt='Archivo CSV de entrada')
@click.option('--outfile', prompt='Archivo CSV de salida')

def conversion_horaria(infile, outfile):
    headers = ['id', 'fecha', 'temp']
    dtypes = {'id': 'int', 'fecha': 'str', 'temp': 'float'}
    parse_dates = ['fecha']
    df = pd.read_csv(infile, decimal=",", sep=";", header=None, names=headers, dtype=dtypes, parse_dates=parse_dates)
    df.index = df['fecha']
    print("Filas leidas: {}".format(df.shape[0]))
    df = df.resample('H').mean()
    df.to_csv(outfile, header=False)
    print("Filas generadas: {}".format(df.shape[0]))

if __name__ == "__main__":
    conversion_horaria()