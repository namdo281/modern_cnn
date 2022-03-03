import torch
from torch import nn
import torch.nn.functional as F
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels= in_channels,
            out_channels= c1,
            kernel_size=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels= c2[0],
            kernel_size=1,            
        )
        self.conv3 = nn.Conv2d(
            in_channels = c2[0],
            out_channels=c2[1],
            kernel_size=3,
            padding=1            
        )
        self.conv4 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c3[0],
            kernel_size=1,
        )
        self.conv5 = nn.Conv2d(
            in_channels=c3[0],
            out_channels=c3[1],
            kernel_size=5,
            padding=2
        )
        self.pool = nn.MaxPool2d(
            padding=1,
            kernel_size=3,
            stride = 1
        )
        self.conv6 = nn.Conv2d(
            in_channels=in_channels,
            kernel_size=1,
            out_channels=c4
        )
        self.out_channels = c1 + c2[1]+ c3[1] +c4


    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = F.relu(x1)

        x2 = self.conv2(x)
        x2 = F.relu(x2)
        x2 = self.conv3(x2)
        x2 = F.relu(x2)

        x3 = self.conv4(x)
        x3 = F.relu(x3)
        x3 = self.conv5(x3)
        x3 = F.relu(x3)
        
        x4 = self.pool(x)
        x4 = self.conv6(x4)
        x4 = F.relu(x4)

        return torch.concat([x1, x2, x3, x4], axis = 1)

class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride = 2,
            padding = 1
        )
        self.conv2 = nn.Conv2d(
            in_channels= self.conv1.out_channels,
            out_channels=192,
            kernel_size=3,
            padding=1

        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=3, 
            stride = 2,
            padding = 1
        )
        self.inception3a = Inception(
            in_channels = self.conv2.out_channels,
            c1 = 64,
            c2 = (96, 128),
            c3 = (16, 32),
            c4 = 32
        )
        self.inception3b = Inception(
            in_channels = self.inception3a.out_channels,
            c1 = 128,
            c2 = (128, 192),
            c3 = (32, 96),
            c4 = 64
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size=3,
            stride = 2,
            padding = 1
        )
        self.inception4a = Inception(
            in_channels= self.inception3b.out_channels,
            c1 = 192,
            c2 = (96, 208),
            c3 = (16, 48),
            c4 = 64
        )
        self.inception4b = Inception(
            in_channels= self.inception4a.out_channels,
            c1 = 160,
            c2 = (112, 224),
            c3 = (24, 64),
            c4 = 64
        )
        self.inception4c = Inception(
            in_channels = self.inception4b.out_channels,
            c1 = 128,
            c2 = (128, 256),
            c3 = (24, 64),
            c4 = 64
        )
        self.inception4d = Inception(
            in_channels = self.inception4c.out_channels,
            c1 = 112,
            c2 = (144, 288),
            c3 = (32, 64),
            c4 = 64
        )
        self.inception4e = Inception(
            in_channels = self.inception4d.out_channels,
            c1 = 256,
            c2 = (160, 320),
            c3 = (32, 128),
            c4 = 128
        )
        self.pool4 = nn.MaxPool2d(
            kernel_size=3,
            stride = 2,
            padding = 1
        )
        self.inception5a = Inception(
            in_channels = self.inception4e.out_channels,
            c1 = 256, 
            c2 = (160, 320),
            c3 = (32, 128),
            c4 = 128
        )
        self.inception5b = Inception(
            in_channels= self.inception5a.out_channels,
            c1 = 384,
            c2 = (192, 384),
            c3 = (48, 128),
            c4 = 128
        )
        self.pool5 = nn.AvgPool2d(
            kernel_size= 7,
            stride = 1
        )
        self.dropout = nn.Dropout2d(0.4)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024, 10)
    
    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        #print(x.shape)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        #print(x.shape)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        #print(x.shape)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.pool5(x)
        #print(x.shape)

        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim = 1)
        return x



