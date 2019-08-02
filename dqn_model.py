import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQNetwork(nn.Module):
    """ Actor (Policy) Model. """

    def __init__(self, state_size, action_size, seed, layers=[64, 64]):
        """ Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            layers [(int)]: number of nodes for each hidden layer
        """
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_layer = nn.Linear(state_size, layers[0])
        
        self.hidden = nn.ModuleList()
        for i in range((len(layers)-1)):
            self.hidden.append(nn.Linear(layers[i], layers[i+1]))

        self.out_layer = nn.Linear(layers[-1], action_size)

    def forward(self, state):
        """ Forward pass to map state -> action values."""
        # state input to network and ReLU activation
        x = F.relu(self.input_layer(state))
        
        # pass through hidden layers with ReLU activations
        for layer in self.hidden:
            x = F.relu(layer(x))

        # return action vector without activation    
        return self.out_layer(x)