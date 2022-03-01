import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize
class AlexNet(nn.Module):
    def __init__ (self):
        super().__init__()
        self.resizor = Resize(224)
        self.conv1 = nn.Conv2d(
            in_channels= 1,
            out_channels=96,
            stride= 4,
            kernel_size= 11,
        ) 
        self.pool1 = nn.MaxPool2d(
            kernel_size= 3,
            stride = 2
        )
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=3,
            stride = 2
        )
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            padding=1
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size= 3, 
            stride=2
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(9600, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, 10)
    def forward(self, x):
        #print(x.shape)
        x = self.resizor(x)
        #print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        #print(x.shape)
        output = F.log_softmax(x)
        #print(output.shape)
        return output



    