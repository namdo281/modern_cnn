import torch
def train(data_loader, model, loss_fn, optimizer, device, print_after = 100):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X,y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        #print(pred)
        #print(y)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % print_after == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}[{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")