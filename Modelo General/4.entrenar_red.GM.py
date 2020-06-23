import numpy as np
import pandas as pd
import click

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter


# crear redes neuronales
class Red_ArquitecturaA(nn.Module):
    def __init__(self, input_size, output_size, p_drop):
        super(Red_ArquitecturaA, self).__init__()

        # In
        self.layer_in = nn.Linear(input_size, 410)
        self.relu_in = nn.ReLU()
        # Hidden 1
        self.layer_hidden1 = nn.Linear(410, 200)
        self.drop = nn.Dropout(p=p_drop)
        self.relu_hidden1 = nn.ReLU()
        # Out
        self.layer_output = nn.Linear(200, output_size)

    def forward(self, x):
        out = self.layer_in(x)
        out = self.relu_in(out)
        out = self.layer_hidden1(out)
        out = self.drop(out)
        out = self.relu_hidden1(out)
        out = self.layer_output(out)
        return out

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

class Red_ArquitecturaC(nn.Module):
    def __init__(self, input_size, output_size, p_drop):
        super(Red_ArquitecturaC, self).__init__()

        # In
        self.layer_in = nn.Linear(input_size, 100)
        self.relu_in = nn.ReLU()
        # autoencoder
        self.layer_hidden1 = nn.Linear(100, 50)
        self.relu_hidden1 = nn.ReLU()
        self.layer_hidden2 = nn.Linear(24, 100)
        self.relu_hidden2 = nn.ReLU()
        self.layer_hidden3 = nn.Linear(100, input_size)
        self.relu_hidden3 = nn.ReLU()
        # Out
        self.layer_output = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.layer_in(x)
        out = self.relu_in(out)
        out = self.layer_hidden1(out)
        out = self.relu_hidden1(out)
        out = self.layer_hidden2(out)
        out = self.relu_hidden2(out)
        out = self.layer_hidden3(out)
        out = self.relu_hidden3(out)
        out = self.layer_output(out)
        return out

class TemperaturaDataSet(Dataset):
    def __init__(self, data):
        """
        Dataset a medida para el ejemplo
        In:
            data: numpy
        Out:
            tupla (entradas, salida)
        """
        super().__init__()
        self.datos = data

    def __len__(self):
        return(self.datos.shape[0])

    def __getitem__(self, index):
        return (self.datos[index, :-24], self.datos[index, -24:])

@click.command()
@click.option('--dataset', prompt='Ruta del dataset: ', default='./dataset.csv')
@click.option('--modelo', prompt='Modelo a usar: ', default='A', type=click.Choice(['A', 'B', 'C']))
@click.option('--batch_size', prompt='batch_size: ', default=1000)
@click.option('--num_epoch', prompt='num_epoch: ', default=50)
@click.option('--p_dropout', prompt='p_dropout: ', default=0.3)

def main(dataset, modelo, batch_size, num_epoch, p_dropout):
    writer = SummaryWriter('runs/' + modelo + '_' + str(batch_size) + '_' + str(num_epoch) + '_' + str(p_dropout))

    if modelo=='A':
        red = Red_ArquitecturaA(input_size=168, output_size=24, p_drop=p_dropout)
    if modelo=='B':
        red = red = Red_ArquitecturaB(input_size=168, output_size=24, p_drop=p_dropout)
    if modelo=='C':
        red = Red_ArquitecturaC(input_size=168, output_size=24, p_drop=p_dropout)

    funcion_perdida = nn.MSELoss()
    optimizer = torch.optim.Adam(params=red.parameters(), lr=0.001)

    train = np.array(pd.read_csv(dataset, decimal=".", sep=",", header=None).values)
    train_ds = TemperaturaDataSet(train)
    train_dataloader = DataLoader(dataset=train_ds, shuffle=True, batch_size=batch_size)

    red.train()
    for epoch in range(num_epoch):
        batch = 0
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            x_train = x_train.type(torch.float)
            y_train = y_train.type(torch.float)

            y_pred_train = red(x_train) # es lo mismo que red.forward(entrada)

            loss = funcion_perdida(y_pred_train, y_train)
            loss.backward()
            optimizer.step()
            print("Epoch: %2d Batch: %6d Loss: %2.8f ErrorMean: %2.8f" %(epoch, batch, loss.item(), (y_pred_train - y_train).mean()))
            batch = batch + 1

        writer.add_scalar('data/train/loss', loss.item(), epoch)
        writer.add_scalar('data/train/ErrorMean', (y_pred_train - y_train).mean(), epoch)
        #for name, param in red.named_parameters():
        #        writer.add_histogram(name, param.clone().data.numpy(), epoch)
    writer.close()

if __name__ == "__main__":
    main()
