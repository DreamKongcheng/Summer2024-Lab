import torch
import torch.utils
import torch.nn as nn
#import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
#from transformer import *
from vit import VIT
import os

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

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #print("Shape of data before: ", data.shape)  b c h w
        #data = data.view(data.shape[0], -1) 
        #print("Shape of data after: ", data.shape)
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)

        #print(output.shape, target.shape)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data = data.view(data.shape[0], -1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return test_loss
    
def main():
    if torch.cuda.is_available():
        print("gpu")
    else: 
        print("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50
    lr = 0.001
    batch_size = 64
    #prepare datasets
    train_loader, test_loader = load_mnist(batch_size)
    
    num_blocks = 6
    d_model = 64
    d_liner = 128
    num_heads = 8
    drop_prob = 0.2
    img_size = 28
    channels = 1
    patch_size = 7
    num_classes = 10

    model = VIT(num_blocks, d_model, d_liner, num_heads, drop_prob, img_size, channels, patch_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() #todo 逆天，开始忘了加上括号


    best_loss = float('inf')
    best_model = None
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, criterion, epoch)
        loss = test(model, device, test_loader, criterion)
        if loss < best_loss:
            best_loss = loss
            best_model = model
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(best_model.state_dict(), os.path.join(model_dir, "best_model.pth"))

if __name__ == '__main__':
    main()