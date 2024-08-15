import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 64
learning_rate = 0.001
num_epochs = 10


transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Modify input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Modify ResNet18 to fit MNIST
class ResNet18MNIST(nn.Module):
    def __init__(self):
        super(ResNet18MNIST, self).__init__()
        self.resnet = resnet18(pretrained=False, num_classes=10)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return self.resnet(x)


model = ResNet18MNIST().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train():
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        end_time = time.time()
        epoch_time = end_time - start_time
        print('Epoch[{}/{}], Loss: {:.4f}, Time: {:.2f} seconds'.format(epoch+1, num_epochs, loss.item(), epoch_time))


def evaluate():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0.0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = test_loss / len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        average_loss, correct, len(test_loader.dataset), accuracy))



if __name__ == '__main__':
    train()
    evaluate()