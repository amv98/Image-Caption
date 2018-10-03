import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels= in_channels, kernel_size = 3, out_channels = out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

class cnn(nn.Module):
    def __init__(self, num_classes=10):
        super(cnn, self).__init__()
        self.unit1 = Unit(3, 32)
        self.unit2 = Unit(32, 32)
        self.unit3 = Unit(32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)

        self.unit4 = Unit(32, 64)
        self.unit5 = Unit(64, 64)
        self.unit6 = Unit(64, 64)
        self.unit7 = Unit(64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)

        self.unit8 = Unit(64, 128)
        self.unit9 = Unit(128, 128)
        self.unit10 = Unit(128, 128)
        self.unit11 = Unit(128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size = 2)

        self.unit12 = Unit(128, 128)
        self.unit13 = Unit(128, 128)
        self.unit14 = Unit(128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size = 4)

        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4
                                ,self.unit5, self.unit6, self.unit7, self.pool2, self.unit8
                                ,self.unit9, self.unit10, self.unit11, self.pool3, self.unit12
                                ,self.unit13, self.unit14, self.avgpool)
        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        return output

def save_models(epoch):
    torch.save(model.state_dict(), "cifarmodel_{}".format(epoch))
    print("Saved model")

transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers = 4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

gpu = torch.cuda.is_available()
model = cnn(num_classes = 10)
if gpu:
    model.cuda()



loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if gpu:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == labels.data)
    return test_acc / 10000

def train(num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if gpu:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_acc += torch.sum(prediction == labels.data)

        train_acc = train_acc / 50000
        train_loss = train_loss / 50000
        test_acc = test()
        if epoch % 15 == 0:
            save_models(epoch)
        print("Epoch {}, Train acc: {}, Train loss: {}, test acc: {}".format(epoch, train_acc, train_loss, test_acc))

if __name__ == "__main__":
    train(100)
