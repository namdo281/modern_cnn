from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize
def vgg_module(conv_arch):
    vm = []
    in_channels = 1
    for (num_conv, num_channels) in conv_arch:
        #in_channels = 1
        print(in_channels)
        for j in range(num_conv):
            vm.append(  
                nn.Conv2d(
                    in_channels= in_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    padding = 1
                ),
                
            )
            vm.append(nn.ReLU())
            in_channels = num_channels
        vm.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*vm)
                 

class VGGNet(nn.Module):
    def __init__(self, conv_arch):
        super().__init__()
        self.vm = vgg_module(conv_arch)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6272, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(256, 10) 
        #print(self.vm)
    def forward(self, x):
        #print(x.shape)
        #print(x.shape)
        x = self.vm(x)
        #print(x.shape)
        #print(x.shape)
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        #print(x.shape)
        output = F.log_softmax(x, dim=-1)
        #print(x)
        return output

