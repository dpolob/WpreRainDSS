import torch.nn as nn


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
