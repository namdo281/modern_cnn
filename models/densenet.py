import torch
from torch import nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding)
    def forward(self, x):
        #print(x.shape)
        x = self.conv(self.relu(self.bn(x)))
        #print(x.shape)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=1, padding = 0)
        self.pool = nn.AvgPool2d(
            kernel_size= 2,
            stride = 2
        )
    def forward(self, x):
        #print(x.shape)
        #print(1)
        x = self.conv(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape) 
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_block):
        super().__init__()
        convs = []
        for i in range(num_block):
            if i == 0:
                conv1 = ConvLayer(in_channels, out_channels, kernel_size=1, padding = 0)
            else: 
                conv1 = ConvLayer(out_channels, out_channels, kernel_size=1, padding = 0)
            
            conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, padding=1)
            convs.append(nn.Sequential(*[conv1, conv2]))
        self.convs = convs
        self.out_channels = in_channels + out_channels*num_block

    def forward(self, x):
        xs = [x]
        for c in self.convs:
            #print(c)
            x = c(x)
            xs.append(x)
        return torch.concat(xs, axis = 1)


class DenseNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.conv1 = ConvLayer(1, k, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)


        self.dense1 = DenseBlock(k, k, 6)
        dense1_oc = self.dense1.out_channels
        self.transition1 = TransitionLayer(dense1_oc, dense1_oc//2)
        
        
        self.dense2 = DenseBlock(dense1_oc // 2, k, 12)
        dense2_oc = self.dense2.out_channels
        self.transition2 = TransitionLayer(dense2_oc, dense2_oc//2)

        self.dense3 = DenseBlock(dense2_oc // 2, k, 12)
        dense3_oc = self.dense3.out_channels
        self.transition3 = TransitionLayer(dense3_oc, dense3_oc//2)

        self.dense4 = DenseBlock(dense3_oc // 2, k, 12)

        self.pool2 = nn.AvgPool2d(kernel_size= 7)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(700, 10)
        self.dropout = nn.Dropout()

    def forward(self, x):
        #print("forward")
        #print(x.shape,1)
        x = self.conv1(x)
        #print(x.shape,2)
        x = self.pool1(x)
        #print(x.shape,3)
        x = self.dense1(x)
        #print(x.shape)
        x = self.transition1(x)
        #print(x.shape)
        x = self.dense2(x)
        #print(x.shape)
        x = self.transition2(x)
        #print(x.shape)
        x = self.dense3(x)
        #print(x.shape)
        x = self.transition3(x)
        #print(x.shape)
        x = self.dense4(x)
        #print(x.shape)
        x = self.pool2(x)
        #print(x.shape)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
