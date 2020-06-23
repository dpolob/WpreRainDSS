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
@click.option('--dataset_train', prompt='Ruta del dataset entrenamiento: ', default='.\\dataset_generalista.csv')
@click.option('--mmscaler', prompt='Ruta del escalador: ', default='.\\mmscaler.pickle')
@click.option('--outmodel', prompt='Ruta para guardar el modelo', default='.\\B_100_50_05_7.model.pt')
def main(dataset_train, mmscaler, outmodel, batch_size=100, num_epoch=50, p_dropout=0.5, dias=7):

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

    torch.save(red.state_dict(), outmodel)
    print("Modelo guardado con éxito!")


if __name__ == "__main__":
    main()
