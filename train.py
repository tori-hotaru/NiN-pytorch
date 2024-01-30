import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model import *

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    net = NiN().to(device)
    # 载入预训练模型（如果存在）
    model_path = 'model_weights.pth'
    try:
        net.load_state_dict(torch.load(model_path))
        print("Loaded saved model")
    except FileNotFoundError:
        print("No saved model found, starting training from scratch")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
                # 每个epoch结束后保存模型
            torch.save(net.state_dict(), model_path)
            print(f"Saved model at epoch {epoch + 1}")
            
        epoch += 1
