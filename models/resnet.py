from torch import nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv = num_conv
        self.use_id_conv = in_channels != out_channels
        self.bn = nn.BatchNorm2d(out_channels)
        convs = []
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride = 2
        )
        for i in range(num_conv):
            if i == 0:
                conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride = 1,
                    padding = 1
                    )
            else:
                conv = nn.Conv2d(
                    out_channels, 
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            convs.append(conv)
            convs.append(nn.BatchNorm2d(out_channels))
            convs.append(nn.ReLU())
        self.convs = nn.Sequential(*convs)
        if self.use_id_conv:
            self.id_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
    def forward(self, x):
        x = self.pool(x)
        x1 = self.convs(x)
        if self.use_id_conv:
            x2 = self.id_conv(x)    
        else:
            x2 = x
        # print(x1.shape)
        # print(x2.shape)
        return x1+x2



class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, stride = 2 , padding= 3 )
        self.residual1 = ResidualBlock(64, 64, 2)
        self.residual2 = ResidualBlock(64, 128, 2)
        self.residual3 = ResidualBlock(128, 256, 2)
        self.residual4 = ResidualBlock(256, 512, 2)
        self.pool = nn.AvgPool2d(kernel_size= 7)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 10)
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.residual1(x)
        # print(x.shape)
        x = self.residual2(x)
        # print(x.shape)
        x = self.residual3(x)
        # print(x.shape)
        x = self.residual4(x)
        # print(x.shape)
        x = self.pool(x)
        # `print(x.shape)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = F.relu(x)
        return x


