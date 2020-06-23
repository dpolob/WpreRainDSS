import numpy as np
import pandas as pd
import click
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter


def SME(real, pred):
    return np.sum((real - pred)**2)/real.shape[0]

# crear redes neuronales

class Red_Parcela(nn.Module):
    def __init__(self, input_size, output_size):
        super(Red_Parcela, self).__init__()

        # In
        self.layer_in = nn.Linear(input_size, 150)
        self.relu_in = nn.ReLU()
        # Hidden 1
        self.layer_hidden1 = nn.Linear(150, 75)
        self.relu_hidden1 = nn.ReLU()
        # Out
        self.layer_output = nn.Linear(75, output_size)

    def forward(self, x):
        out = self.layer_in(x)
        out = self.relu_in(out)
        out = self.layer_hidden1(out)
        out = self.relu_hidden1(out)
        out = self.layer_output(out)
        return out

class Red_General(nn.Module):
    def __init__(self, input_size, output_size, p_drop):
        super(Red_General, self).__init__()
        self.layer_in = nn.Linear(input_size, 410)
        self.relu_in = nn.ReLU()
        self.layer_hidden1 = nn.Linear(410, 410)
        self.drop = nn.Dropout(p=p_drop)
        self.relu_hidden1 = nn.ReLU()
        self.layer_hidden2 = nn.Linear(410, 200)
        self.drop2 = nn.Dropout(p=p_drop)
        self.relu_hidden2 = nn.ReLU()
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
    def __init__(self, data, red):
        """
        Dataset a medida para el ejemplo
        In:
            data: numpy
        Out:
            tupla (entradas_parcela, salida)
        """
        super().__init__()
        self.datos = data
        self.red = red

    def __len__(self):
        return(self.datos.shape[0])

    def __getitem__(self, index):
        general = self.red(torch.tensor(self.datos[index, 0:168], dtype=torch.float))
        return (np.concatenate((general.detach().numpy(), self.datos[index, 168:-24]), 0), self.datos[index, -24:])



@click.command()
@click.option('--dataset', prompt='Ruta del dataset a validar entrenar: ', default='.\\dataset.csv')
@click.option('--horastrain', prompt='Horas a preentrenar: ', default=2880)
@click.option('--generalmodel', prompt='Ruta del modelo general', default='.\\B_100_50_05_7.model.pt')
@click.option('--mmscaler', prompt='Ruta del escalador de temperaturas', default='.\\mmscaler.pickle')
@click.option('--outmodel', prompt='Ruta para guardar el modelo de parcela ', default='.\\parcela101_model.pt')
def main(dataset, horastrain, generalmodel, mmscaler, outmodel):

    # cargar modelo de la red general
    red_general = Red_General(input_size=168, output_size=24, p_drop=0.5)
    red_general.load_state_dict(torch.load(generalmodel))
    red_general.eval()

    # crear modelo de la red de parcela
    red_parcela = Red_Parcela((24 + 24 + 12 + 1), 24)
    funcion_perdida = nn.MSELoss()
    optimizer = torch.optim.Adam(params=red_parcela.parameters(), lr=0.001)

    train = np.array(pd.read_csv(dataset, decimal=".", sep=",", header=None).values)
    train_ds = TemperaturaDataSet(train[0:horastrain, :], red_general) # solo las de preentrenamiento
    train_dataloader = DataLoader(dataset=train_ds, shuffle=True, batch_size=50)

    # preentramiento
    writer = SummaryWriter('runs/pre_training/MP2')

    for epoch in range(25):
        batch = 0
        for x_train_parcela, y_train_parcela in train_dataloader:
            x_train_parcela = x_train_parcela.type(torch.float)
            y_train_parcela = y_train_parcela.type(torch.float)

            optimizer.zero_grad()
            y_pred_parcela = red_parcela(x_train_parcela)
            loss = funcion_perdida(y_pred_parcela, y_train_parcela)
            loss.backward()
            optimizer.step()
            print("Epoch: %2d Batch: %6d Loss: %2.8f ErrorMean: %2.8f" %(epoch, batch, loss.item(), (y_pred_parcela - y_train_parcela).mean()))
            batch = batch + 1

        writer.add_scalar('data/train/loss', loss.item(), epoch)
        writer.add_scalar('data/train/ErrorMean', (y_pred_parcela - y_train_parcela).mean(), epoch)
        #for name, param in red.named_parameters():
        #        writer.add_histogram(name, param.clone().data.numpy(), epoch)
    writer.close()

    # cambiamos el dataset
    train = np.array(pd.read_csv(dataset, decimal=".", sep=",", header=None).values)
    train_ds = TemperaturaDataSet(train[horastrain:, :], red_general) # el resto
    train_dataloader = DataLoader(dataset=train_ds, shuffle=False, batch_size=1)

    # funcion_RMS = nn.MSELoss()
    temp_scaler = pickle.load(open(mmscaler, 'rb'))

    writer = SummaryWriter('runs/deployed/MP_24')
    batch = 0
    for x_train_parcela, y_train_parcela in train_dataloader:
        x_train_parcela = x_train_parcela.type(torch.float)
        y_train_parcela = y_train_parcela.type(torch.float)

        # Evaluacion
        red_parcela.eval()
        y_pred_parcela = red_parcela(x_train_parcela)
        RMS = SME(y_pred_parcela.detach().numpy(), y_train_parcela.detach().numpy())
        print("Evaluacion    - Batch: %6d RMS: %2.8f ErrorMean: %2.8f" %(batch, RMS.item(), (y_pred_parcela - y_train_parcela).mean()))
        writer.add_scalar('data/evaluation/RMS', RMS.item(), batch)
        writer.add_scalar('data/evaluation/ErrorMean', (y_pred_parcela - y_train_parcela).mean(), batch)
        if batch % 24 == 0:
            fig =plt.figure(figsize=(13,6))
            plt.plot(temp_scaler.inverse_transform(y_train_parcela.data.numpy().reshape(-1,1)), 'b',
            temp_scaler.inverse_transform(y_pred_parcela.data.numpy().reshape(-1,1)), 'r')
            # plt.figtext(0, 0, "Batch: " + str(batch))
            # plt.figtext(0.3, 0, "EMean: " + str((y_pred_parcela - y_train_parcela).data.numpy().mean()))
            # plt.figtext(0.6, 0, "MSE: " + str( RMS.item()))
            # plt.savefig("img_" + str(batch) + ".jpg")
            writer.add_figure('data/evaluation/resultados', fig, batch)

            red_parcela.train()
            train_ds_recursivo = TemperaturaDataSet(train[0:(batch + horastrain), :], red_general) # el resto
            train_dataloader_recursivo = DataLoader(dataset=train_ds_recursivo, shuffle=True, batch_size=50)

            for x_train_parcela, y_train_parcela in train_dataloader_recursivo:
                x_train_parcela = x_train_parcela.type(torch.float)
                y_train_parcela = y_train_parcela.type(torch.float)
                # Entrenamiento
                optimizer.zero_grad()
                y_pred_parcela = red_parcela(x_train_parcela)
                loss = funcion_perdida(y_pred_parcela, y_train_parcela)
                loss.backward()
                optimizer.step()

            print("Entrenamiento - Batch: %6d Loss: %2.8f" %(batch, loss.item()))
            writer.add_scalar('data/train/loss', loss.item(), batch)

        batch = batch + 1
    writer.close()

    # Guardar para simulaciones
    torch.save(red_parcela.state_dict(), outmodel)
    print("Modelo guardado con éxito!")

if __name__ == "__main__":
    main()
