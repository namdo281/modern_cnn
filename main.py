import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
from models import *
from functions import train, test


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_data = datasets.FashionMNIST(
    'data', 
    train = True, 
    download = True, 
    transform= ToTensor()
    )

test_data = datasets.FashionMNIST(
    'data',
    train = False,
    download= True,
    transform= ToTensor()   
)
batch_size = 64
resizor = Resize(224)

train_dataloader = DataLoader(train_data, batch_size= batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
for X, y in train_dataloader:
    print(X.shape)
    print(y.shape)
    break

model = AlexNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

print(model)
epochs = 10
for t in range(epochs):
    print(f"epoch {t+1}")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)