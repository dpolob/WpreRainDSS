import torch
import torch.nn as nn

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
