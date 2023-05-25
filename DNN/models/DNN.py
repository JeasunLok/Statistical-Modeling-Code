import torch
import torch.nn as nn

#build the model
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.sequential = nn.Sequential(
        nn.Linear(input_size,hidden_size1),
        nn.ReLU(),
        nn.Linear(hidden_size1,hidden_size2),
        nn.ReLU(),
        nn.Linear(hidden_size2, output_size),
        )
    def forward(self, input):
        theta = self.sequential(input)
        return theta 
    