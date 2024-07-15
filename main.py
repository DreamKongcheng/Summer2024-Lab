import torch
import torch.utils
import torch.nn as nn
#import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformer import *

def load_mnist(batch_size):
    #定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    #下载训练数据与测试数据
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #print("Shape of data before: ", data.shape)
        data = data.view(data.shape[0], -1) #把batch*w*h*channel压缩成batch*d
        #print("Shape of data after: ", data.shape)
        optimizer.zero_grad()
        output = model(data, data, None, None)

        #print(output.shape, target.shape)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.shape[0], -1)
            output = model(data, data, None, None)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')
    
def main():
    epochs = 10
    lr = 0.01
    batch_size = 64
    #prepare datasets
    train_loader, test_loader = load_mnist(batch_size)
    
    d_model = 784
    num_heads = 8
    d_ffn = 2048
    drop_prob = 0.1
    num_blocks = 6
    d_out = 10

    model = Transformer(d_model, num_heads, d_ffn, drop_prob, num_blocks, d_out)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() #todo 逆天，开始忘了加上括号

    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, epoch)
        test(model, test_loader, criterion)

if __name__ == '__main__':
    main()