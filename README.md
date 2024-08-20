# 基于Pytorch的MNIST 数据集分类

## 目录
- [项目简介](#项目简介)
- [数据集介绍](#数据集介绍)
- [分类方法](#分类方法)
  - [LeNet](#lenet)
  - [VGG8](#vgg8)
  - [ResNet18](#resnet18)
  - [ViT (Vision Transformer)](#vit-vision-transformer)
- [实现过程](#实现过程)
- [完成效果](#完成效果)

## 项目简介
本项目旨在对MNIST手写数字数据集进行分类，在项目中实现并评估了LeNet、VGG8、ResNet18和ViT等模型的性能。

## 数据集介绍
MNIST数据集是一个经典的手写数字识别数据集，包含60,000个训练样本和10,000个测试样本，每个样本是28x28像素的灰度图像，表示0到9的手写数字。

## 分类方法

### LeNet
- **简介**：LeNet是最早的卷积神经网络之一，专为手写数字识别设计。

- **网络结构和实现细节**：LeNet由两个“卷积-池化”结构和两个全连接层串联而成

  ```python
  
  class LeNet(nn.Module):
      def __init__(self):
          super(LeNet, self).__init__()
          # 1 input image channel, 6 output channels, 5x5 square convolution
          # kernel
          self.conv1 = nn.Conv2d(1, 6, 5)
          self.conv2 = nn.Conv2d(6, 16, 5)
          # an affine operation: y = Wx + b
          self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4*4 from image dimension
          self.fc2 = nn.Linear(120, 84)
          self.fc3 = nn.Linear(84, 10)
  
      def forward(self, x):
          # Max pooling over a (2, 2) window
          x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
          # If the size is a square, you can specify with a single number
          x = F.max_pool2d(F.relu(self.conv2(x)), 2)
          # x = F.relu(self.conv2(x))
  
          x = torch.flatten(x, 1)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          x = F.log_softmax(x, dim=1)
          return x

### VGG8
- **简介**：VGG8是VGG网络的一个变种，具有较小的参数量和较深的网络结构，可以节省训练时长。

- **网络结构和实现细节**：VGG8串联四个“卷积-卷积-池化”的结构和三个全连接层；由于torchvision库中没有VGG8模型，网络结构需要手动编写，为了适应MNIST的灰度图输入，第一个卷积层的输入通道改为1；为了优化训练过程，对网络参数进行He初始化。

  ```python
  
  class VGG8(nn.Module):
      def __init__(self, num_classes):
          super(VGG8, self).__init__()
          self.features = nn.Sequential(
              nn.Conv2d(1, 64, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.Conv2d(64, 64, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
  
              nn.Conv2d(64, 128, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
  
              nn.Conv2d(128, 256, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.Conv2d(256, 256, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
  
              nn.Conv2d(256, 256, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.Conv2d(256, 256, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
          )
          self.classifier = nn.Sequential(
              nn.Linear(256 * 7 * 7, 4096),
              nn.ReLU(True),
              nn.Dropout(),
              nn.Linear(4096, 4096),
              nn.ReLU(True),
              nn.Dropout(),
              nn.Linear(4096, num_classes),
          )
          self._initialize_weights()
  
      def forward(self, x):
          x = self.features(x)
          x = x.view(x.size(0), -1)
          x = self.classifier(x)
          return x
  
      def _initialize_weights(self):
          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Initialize with He
                  if m.bias is not None:
                      nn.init.constant_(m.bias, 0)
              elif isinstance(m, nn.Linear):
                  nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Initialize the classifier using He
                  if m.bias is not None:
                      nn.init.constant_(m.bias, 0)

### ResNet18
- **简介**：ResNet18是残差网络的一种，通过引入残差学习来解决深层神经网络训练中的梯度消失和退化问题。

- **网络结构和实现细节**：ResNet18是torchvision中存有的模型，实际编写结构的时候只需要修改第一个卷积层以适应灰度图输入即可。

  ```python
  
  class ResNet18MNIST(nn.Module):
      def __init__(self):
          super(ResNet18MNIST, self).__init__()
          self.resnet = resnet18(pretrained=False, num_classes=10)
          self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  
      def forward(self, x):
          return self.resnet(x)

### ViT (Vision Transformer)
- **简介**：Vision Transformer是一种基于Transformer架构的视觉模型，通过自注意力机制处理图像数据。

- **网络结构和实现细节**：ViT的基本结构为“卷积-Transformer编码-全连接层”

  ```python
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
  ```

  

## 实现过程
- **环境配置**：

  \- GPU: NVIDIA GeForce RTX 3090, 24GB 显存

  \- 操作系统: Ubuntu 20.04 

  \- IDE: PyCharm

  \- Python 版本: 3.11 (Conda)

  \- PyTorch 版本: 1.12.1

- **数据预处理**：在实际的操作过程中要修改transform参数以适应模型的输入

  ```python
  rain_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
  test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

- **模型训练**：batch_size = 64, num_epochs = 10, learning_rate = 0.001

- **模型评估**：在测试集上评估模型性能，用交叉熵损失作为测试损失

## 完成效果
- **性能比较**：

| 模型     | 训练时长  | 在测试集上的准确度 |
| -------- | --------- | ------------------ |
| LeNet    | 140.76 秒 | 99 %               |
| VGG8     | 592.56 秒 | 99.21 %            |
| ResNet18 | 393.14 秒 | 99.42 %            |
| ViT      | 221.13 秒 | 97.87 %            |
