import click
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

# aqui solo definimos nuestra arquitectura


class Red_ArquitecturaB(nn.Module):
    def __init__(self, input_size, output_size, p_drop):
        super(Red_ArquitecturaB, self).__init__()

        # In
        self.layer_in = nn.Linear(input_size, 410)
        self.relu_in = nn.ReLU()
        # Hidden 1
        self.layer_hidden1 = nn.Linear(410, 410)
        self.drop = nn.Dropout(p=p_drop)
        self.relu_hidden1 = nn.ReLU()
        # Hidden 2
        self.layer_hidden2 = nn.Linear(410, 200)
        self.drop2 = nn.Dropout(p=p_drop)
        self.relu_hidden2 = nn.ReLU()
        # Out
        self.layer_output = nn.Linear(200, output_size)

    def forward(self, x):
        out = self.layer_in(x)
        out = self.relu_in(out)
        out = self.layer_hidden1(out)
        out = self.drop(out)
        out = self.relu_hidden1(out)
        out = self.layer_hidden2(out)
        out = self.drop2(out)
        out = self.relu_hidden2(out)
        out = self.layer_output(out)
        return out


class TemperaturaDataSet(Dataset):
    def __init__(self, data, dias):
        """
        Dataset a medida para el ejemplo
        In:
            data: numpy

        Out:
            tupla (entradas, salida)
        """
        super().__init__()
        self.datos = data
        self.dias = dias

    def __len__(self):
        return(self.datos.shape[0])

    def __getitem__(self, index):
        return (self.datos[index,(7 - self.dias) * 24:-24], self.datos[index, -24:])


@click.command()
@click.option('--dataset_train', prompt='Ruta del dataset entrenamiento: ', default='./dataset_generalista.csv')
@click.option('--dataset_validation', prompt='Ruta del dataset validacion: ', default='./dataset_generalista_validacion.csv')
@click.option('--mmscaler', prompt='Ruta del escalador: ', default='./mmscaler.pickle')
@click.option('--batch_size', prompt='Batch size', default=100)
@click.option('--num_epoch', prompt='Num epoch', default=50)
@click.option('--p_dropout', prompt='P Dropout', default=0.5)
@click.option('--dias', prompt='Días pasados', default=7)
def main(dataset_train, dataset_validation, mmscaler, batch_size, num_epoch, p_dropout, dias):
    writer = SummaryWriter('selected/FT_B' + '_' + str(batch_size) +
                           '_' + str(num_epoch) + '_' + str(p_dropout) + '_' + str(dias))

    red = Red_ArquitecturaB(input_size=(24 * dias), output_size=24, p_drop=p_dropout)

    funcion_perdida = nn.MSELoss()
    optimizer = torch.optim.Adam(params=red.parameters(), lr=0.001)

    train = np.array(pd.read_csv(dataset_train, decimal=".", sep=",", header=None).values)
    train_ds = TemperaturaDataSet(train, dias=dias)
    train_dataloader = DataLoader(dataset=train_ds, shuffle=True, batch_size=batch_size)

    # entrenamiento de la red
    red.train()
    for epoch in range(num_epoch):
        batch = 0
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            x_train = x_train.type(torch.float)
            y_train = y_train.type(torch.float)

            y_pred_train = red(x_train)  # es lo mismo que red.forward(entrada)

            loss = funcion_perdida(y_pred_train, y_train)
            loss.backward()
            optimizer.step()
            print("Epoch: %2d Batch: %6d Loss: %2.8f ErrorMean: %2.8f" %
                  (epoch, batch, loss.item(), (y_pred_train - y_train).mean()))
            batch = batch + 1

        writer.add_scalar('data/train/loss', loss.item(), epoch)
        writer.add_scalar('data/train/ErrorMean', (y_pred_train - y_train).mean(), epoch)
        for name, param in red.named_parameters():
            writer.add_histogram(name, param.clone().data.numpy(), epoch)

    # evaluacion de la validacion
    test = np.array(pd.read_csv(dataset_validation, decimal=".", sep=",", header=None).values)
    test_ds = TemperaturaDataSet(test,dias=dias)
    test_dataloader = DataLoader(dataset=test_ds, shuffle=False, batch_size=1)

    temp_scaler = pickle.load(open(mmscaler, 'rb'))
    print(temp_scaler)

    red.eval()

    batch = 0
    for x_test, y_test in test_dataloader:
        x_test = x_test.type(torch.float)
        y_test = y_test.type(torch.float)

        y_pred_test = red(x_test)
        loss = funcion_perdida(y_pred_test, y_test)
        print("Batch: %6d Loss: %2.8f ErrorMean: %2.8f" %
              (batch, loss.item(), (y_pred_test - y_test).mean()))
        writer.add_scalar('data/test/loss', loss.item(), batch)
        writer.add_scalar('data/test/ErrorMean', (y_pred_test - y_test).mean(), batch)
        if batch % 100 == 0:
            fig = plt.figure(figsize=(13, 6))
            plt.plot(temp_scaler.inverse_transform(y_test.data.numpy().reshape(-1, 1)), 'b',
                     temp_scaler.inverse_transform(y_pred_test.data.numpy().reshape(-1, 1)), 'r')
            writer.add_figure('data/test/resultados', fig, batch)

        batch = batch + 1

    writer.close()


if __name__ == "__main__":
    main()
