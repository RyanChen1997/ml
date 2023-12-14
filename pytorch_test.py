import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from load_data import load_data

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential( # 输入(3, 100, 100)
            nn.Conv2d(
                in_channels=3,
                out_channels=30,
                kernel_size=5,
                padding=0,
                stride=1,
            ), # 经过卷积层(30, 96, 96)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 经过池化层(30, 48, 48)
        )
        self.out = nn.Linear(30*48*48, 4)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        logits = self.out(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward() # 这一步会通过方向传播计算出每一层的梯度
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def optimize():
    (x_train, t_train), (x_test, t_test) = load_data(one_hot_label=False, normalize=True)
    x_train_tensor, t_train_tensor, x_test_tensor, t_test_tensor = torch.tensor(x_train), torch.tensor(t_train), torch.tensor(x_test), torch.tensor(t_test)
    train_data = TensorDataset(x_train_tensor, t_train_tensor)
    test_data = TensorDataset(x_test_tensor, t_test_tensor)

    # create dataloader
    batch_size= 5
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # create model
    model = NeuralNetwork().to(device)
    print(model)

    # optmize
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters, lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    # save model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

def predict():
    pass

if __name__ == "__main__":
    optimize()