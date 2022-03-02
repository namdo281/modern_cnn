import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from transforms import Rescale
from models import *
from functions import train, test


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_data = datasets.FashionMNIST(
    'data', 
    train = True, 
    download = True, 
    transform= Rescale(224)
    )

test_data = datasets.FashionMNIST(
    'data',
    train = False,
    download= True,
    transform= Rescale(224)   
)
batch_size = 64

train_dataloader = DataLoader(train_data, batch_size= batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
for i, (X, y) in enumerate(train_dataloader):
    print(X.shape)
    print(y.shape)
    #train_dataloader[i][0] = interpolate(X, 224)
    break
# for i, (X, y) in enumerate(test_dataloader):
#     test_dataloader[i][0] = interpolate(X, 224)
# model = AlexNet().to(device)

# conv_arch = ((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))
# model = VGGNet(conv_arch=conv_arch).to(device)

model = GoogLeNet()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)


# x = torch.randn((64, 1, 224, 224))
# model.forward(x)
print(model)
epochs = 10
for t in range(epochs):
    print(f"epoch {t+1}")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)