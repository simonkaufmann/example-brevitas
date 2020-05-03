import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

epochs = 20

device = "cuda"

batch_size = 64

test_batch_size = 1000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Linear(28*28,10)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        return self.layer(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train_epoch(epoch, model, criterion, optimizer, train_loader):
        model.train()
        running_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 50 == 49:
                print('[%d, %5d] loss%.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))

def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True)


    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for i in range(epochs):
        print("\n\n\nEpoch {}\n\n".format(i))
        train_epoch(i, model, criterion, optimizer, train_loader)
        test(model, criterion, test_loader)
        torch.save(model.state_dict(), 'my_mnist.pt')
        print("Model saved")


if __name__ == '__main__':
    main()
