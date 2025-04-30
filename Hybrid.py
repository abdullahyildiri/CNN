import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import os

# Fashion MNIST veri seti için dönüşümler
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Fashion MNIST veri setini yükleme
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Veri yükleyiciler
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Fashion MNIST sınıf adları
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Özellik çıkarma için CNN mimarisi
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Özellik çıkarma katmanları
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten işlemi
        return x


# Tam CNN mimarisi (karşılaştırma için)
class FullCNN(nn.Module):
    def __init__(self):
        super(FullCNN, self).__init__()
        # Özellik çıkarma katmanları (aynı)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Sınıflandırıcı katmanları
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten işlemi
        x = self.classifier(x)
        return x


# 1. Özellik çıkarıcıyı eğitme
feature_extractor = FeatureExtractor()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_extractor = feature_extractor.to(device)


# Özellik çıkarma işlemi
def extract_features(model, data_loader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    return np.vstack(features), np.concatenate(labels)


print("Özellik çıkarma işlemi başlıyor...")
X_train, y_train = extract_features(feature_extractor, train_loader)
X_test, y_test = extract_features(feature_extractor, test_loader)
print(f"Eğitim veri boyutu: {X_train.shape}, Test veri boyutu: {X_test.shape}")

# Özellikleri kaydetme
os.makedirs('features', exist_ok=True)
np.save('features/X_train.npy', X_train)
np.save('features/y_train.npy', y_train)
np.save('features/X_test.npy', X_test)
np.save('features/y_test.npy', y_test)
print("Özellikler kaydedildi.")

# 2. SVM ile sınıflandırma
print("SVM eğitimi başlıyor...")
svm_classifier = SVC(kernel='rbf', C=10, gamma='scale')
svm_classifier.fit(X_train, y_train)
print("SVM eğitimi tamamlandı.")

# SVM tahminleri
y_pred_svm = svm_classifier.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm) * 100
print(f"SVM Doğruluk: {accuracy_svm:.2f}%")

# Karışıklık matrisi (SVM için)
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.title('SVM Karışıklık Matrisi')
plt.savefig('svm_confusion_matrix.png')

# 3. Full CNN modelini eğitme (karşılaştırma için)
full_cnn = FullCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(full_cnn.parameters(), lr=0.001)

# Eğitim
epochs = 10
train_losses = []

print("Tam CNN eğitimi başlıyor...")
for epoch in range(epochs):
    full_cnn.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = full_cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

print("Tam CNN eğitimi tamamlandı.")

# Full CNN değerlendirme
full_cnn.eval()
correct = 0
total = 0

y_pred_cnn = []
y_true_cnn = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = full_cnn(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        y_pred_cnn.extend(predicted.cpu().numpy())
        y_true_cnn.extend(labels.cpu().numpy())

accuracy_cnn = 100 * correct / total
print(f"Tam CNN Doğruluk: {accuracy_cnn:.2f}%")

# Karışıklık matrisi (CNN için)
cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.title('Tam CNN Karışıklık Matrisi')
plt.savefig('cnn_confusion_matrix.png')

# Sonuçları karşılaştırma
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(['SVM', 'CNN'], [accuracy_svm, accuracy_cnn])
plt.ylim([0, 100])
plt.title('Model Doğruluk Karşılaştırması')
plt.ylabel('Doğruluk (%)')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title('Tam CNN Eğitim Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('hybrid_vs_cnn_comparison.png')
plt.show()

print(f"Hibrit Model (CNN+SVM) Doğruluk: {accuracy_svm:.2f}%")
print(f"Tam CNN Modeli Doğruluk: {accuracy_cnn:.2f}%")

# Modelleri kaydet
torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')
torch.save(full_cnn.state_dict(), 'full_cnn.pth')
# SVM modelini kaydet
import pickle

with open('svm_classifier.pkl', 'wb') as f:
    pickle.dump(svm_classifier, f)