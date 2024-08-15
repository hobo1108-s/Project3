import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleViT(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleViT, self).__init__()
        self.patch_size = 7
        self.embed_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.dropout_rate = 0.1

        # linear embedding layer
        self.patch_embedding = nn.Conv2d(1, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)

        # Classification header
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        # The shape of the input x is (batch_size, 1, 28, 28)
        x = self.patch_embedding(x)  # (batch_size, embed_dim, 4, 4)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, 16, embed_dim)
        x = self.transformer_encoder(x)  # (batch_size, 16, embed_dim)
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        x = self.fc(x)  # (batch_size, num_classes)
        return x


batch_size = 64
learning_rate = 0.001
num_epochs = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = SimpleViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train():
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        end_time = time.time()
        epoch_time = end_time - start_time
        print('Epoch[{}/{}], Loss: {:.4f}, Time: {:.2f} seconds'.format(epoch + 1, num_epochs, loss.item(), epoch_time))


def evaluate():
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
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