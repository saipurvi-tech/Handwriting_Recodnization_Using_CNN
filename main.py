import torch 
import os
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 

#datasett and preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda img: torch.transpose(img,1,2)), transforms.Normalize((0.1307,),(0.3081,))])
import os

# Store data outside OneDrive sync to avoid permission locks
data_dir = os.path.expanduser('~/emnist_data')
train_data = datasets.EMNIST(root=data_dir, split='balanced', train=True, download=True, transform=transform)
test_data = datasets.EMNIST(root=data_dir, split='balanced', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

#cnn model architecture 
class CharacterCNN(nn.Module):
    def __init__(self, num_classes = 47):
        super(CharacterCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

#traingn the setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterCNN(num_classes=47).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epochs=5):
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss, correct, total = 0.0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
            
        epoch_acc = 100. * correct / total
        epoch_loss = running_loss / total
        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

#evaluation 
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
    print(f"\nFinal Test Accuracy: {100. * correct / total:.2f}%")

if __name__ == '__main__':
    train(epochs=5)
    evaluate()