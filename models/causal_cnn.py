from .causal_wrapper import S2TMB
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalPruningLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CausalPruningLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        def causal_prune(features, output_dim):
            data = features.detach().cpu().numpy()
            print("Causal Prune Input Features Shape:", data.shape)
            _, mb = S2TMB(data, 128)
            print("MB Length:", len(mb))
            print("MB:", mb)
            if len(mb) < output_dim:
                selected_features = features[:, mb]
            else:
                selected_features = features[:, mb[:output_dim]]
            return selected_features

        pruned_features = causal_prune(x, self.output_dim)
        print("Pruned Features Shape:", pruned_features.shape)
        return pruned_features





class Net(nn.Module):
    def __init__(self, num_features=24, num_classes_fer=7, num_classes_au=12):
        super(Net, self).__init__()
        self.num_features = num_features
        self.layer1 = self.one_block(1, 3)
        self.layer2 = self.one_block(2, self.num_features)
        self.layer3 = self.one_block(4, 2*self.num_features)
        self.layer4 = self.one_block(8, 4*self.num_features)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(72*2*2*self.num_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.causal_pruning = CausalPruningLayer(256, 128)
        self.last = nn.Linear(128, self.num_features)

        

    def forward(self, x):
        out = {}

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flat(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.causal_pruning(x)

        try:
            out['FER'] = self.last(x)
        except:
            pass

        try:
            out['AV'] = self.last(x)
        except:
            pass

        try:
            out['AU'] = self.last(x)
        except:
            pass

        return out

    def one_block(self, a, inp):
        block = nn.Sequential(
            nn.Conv2d(inp, a * self.num_features, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(a * self.num_features),
            nn.Conv2d(a * self.num_features, a * self.num_features, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(a * self.num_features),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(p=0.4)
        )
        return block

def simple_cnn():
    return Net()
