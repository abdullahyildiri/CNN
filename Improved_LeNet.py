import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# Veri seti dönüşümleri (Model 1 ile aynı)
transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST veri seti (Model 1 ile aynı)
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# İyileştirilmiş LeNet-5 modüleri (BatchNorm ve Dropout ile)
class C1_Improved(nn.Module):
    def __init__(self):
        super(C1_Improved, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('bn1', nn.BatchNorm2d(6)),  # Batch Normalization eklendi
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('dropout1', nn.Dropout(0.1))  # Dropout eklendi
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2_Improved(nn.Module):
    def __init__(self):
        super(C2_Improved, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('bn2', nn.BatchNorm2d(16)),  # Batch Normalization eklendi
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('dropout2', nn.Dropout(0.2))  # Dropout eklendi
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3_Improved(nn.Module):
    def __init__(self):
        super(C3_Improved, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('bn3', nn.BatchNorm2d(120)),  # Batch Normalization eklendi
            ('relu3', nn.ReLU()),
            ('dropout3', nn.Dropout(0.3))  # Dropout eklendi
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class F4_Improved(nn.Module):
    def __init__(self):
        super(F4_Improved, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('bn4', nn.BatchNorm1d(84)),  # Batch Normalization eklendi
            ('relu4', nn.ReLU()),
            ('dropout4', nn.Dropout(0.4))  # Dropout eklendi
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5_Improved(nn.Module):
    def __init__(self):
        super(F5_Improved, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class ImprovedLeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    İyileştirilmiş LeNet5 mimarisi: BatchNorm ve Dropout eklenmiş
    """

    def __init__(self):
        super(ImprovedLeNet5, self).__init__()

        self.c1 = C1_Improved()
        self.c2_1 = C2_Improved()
        self.c2_2 = C2_Improved()
        self.c3 = C3_Improved()
        self.f4 = F4_Improved()
        self.f5 = F5_Improved()

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x  # Paralel yapıların toplamı (Model 1 ile aynı mantık)

        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output


# Model oluşturma, kayıp fonksiyonu ve optimize edici tanımlama
model = ImprovedLeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Eğitim ve kayıp değerlerini saklamak için liste
epochs = 5  # Daha uzun eğitim
train_losses = []

# Eğitim
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

# Test
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy for Improved LeNet-5: {accuracy:.2f}%')

# Loss değerlerini gösteren grafik
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('improved_lenet5_loss.png')
plt.show()