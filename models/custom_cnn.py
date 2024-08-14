

import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, num_features=24, num_classes_fer=7, num_classes_au=12):
        super(Net, self).__init__()
        self.num_features = num_features

        self.layer1 = self.one_block(1, 3)
        self.layer2 = self.one_block(2, self.num_features)
        self.layer3 = self.one_block(4, 2*self.num_features)
        self.layer4 = self.one_block(8, 4*self.num_features)
        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(72*2*2*self.num_features, 192)
        self.fc2 = nn.Linear(192, 96)
        self.last2 = nn.Linear(96, 48)

        self.last = nn.Linear(48, self.num_features)

    def forward(self, x):
        
        out = {}
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.flat(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.last2(x)

        try:
            out['All'] = self.last['All'](x)
        except:
            pass 
        
        try:
            out['FER'] = self.last['FER'](x)
        except:
            pass 

        try:
            out['AV'] = self.last['AV'](x)
        except:
            pass 

        try:
            out['AU'] = self.last['AU'](x)
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