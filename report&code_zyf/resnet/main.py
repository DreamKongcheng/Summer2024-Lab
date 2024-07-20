import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

# define hyper parameters
Batch_size = 32
Lr = 0.1
Epoch = 10
# define train set and test set
train_dataset = torchvision.datasets.MNIST(
    root='./MNIST',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_dataset = torchvision.datasets.MNIST(
    root='./MNISt',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
# define train loader
train_loader = Data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=Batch_size
)

test_x = torch.unsqueeze(test_dataset.data, dim=1).type(torch.Tensor)
test_y = test_dataset.targets

# construct network
class Basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block1 = self._make_layer(block, 16, num_block[0], stride=1)
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2)

        self.outlayer = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_block, stride):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        out = self.outlayer(x)
        return out

ResNet18 = ResNet(Basicblock, [1, 1, 1, 1], 10)
# print(ResNet18)

opt = torch.optim.SGD(ResNet18.parameters(), lr=0.01, momentum=0.5, weight_decay=1e-4)
loss_fun = nn.CrossEntropyLoss()
a = []
ac_list = []
for epoch in range(Epoch):
    batch_idx = -1
    for i, (x, y) in enumerate(train_loader):
        batch_idx += 1
        output = ResNet18(x)
        loss = loss_fun(output, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, loss))
            loss = 0

    test_output = torch.max(ResNet18(test_x), dim=1)[1]
    loss = loss_fun(ResNet18(test_x), test_y).item()
    accuracy = torch.sum(torch.eq(test_y, test_output)).item() / test_y.numpy().size
    print('accuracy on test set: %.4f %% ' % accuracy)

