import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.block2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1, groups=32),
                                   nn.Conv2d(32, 64, 1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.block3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, groups=64),
                                   nn.Conv2d(64, 128, 1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.block4 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 0, groups=128),
                                   nn.Conv2d(128, 256, 1, 1))
        self.bn4 = nn.BatchNorm2d(256)
        self.block5 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, groups=256),
                                   nn.Conv2d(256, 256, 1, 1))
        self.bn5 = nn.BatchNorm2d(256)
        self.block6 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 0, groups=256),
                                   nn.Conv2d(256, 512, 1, 1))
        self.bn6 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512*2*2, 512)
        self.fc2 = nn.Linear(512, 136)
        self.drop1 = nn.Dropout(p=0.4)
        self.drop2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2, stride=2) #112
        x = F.max_pool2d(F.relu(self.bn2(self.block2(x))), 2, stride=2) #56
        x = F.max_pool2d(F.relu(self.bn3(self.block3(x))), 2, stride=2) #28
        x = F.max_pool2d(F.relu(self.bn4(self.block4(x))), 2, stride=2) #12
        x = F.max_pool2d(F.relu(self.bn5(self.block5(x))), 2, stride=2) #6
        x = F.max_pool2d(F.relu(self.bn6(self.block6(x))), 2, stride=2) #2
        x = x.view(-1, 512*2*2)
        x = self.drop2(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x