import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import zipfile

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
#paths for kaggle
TRAIN_DIR = '/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Training'
TEST_DIR = '/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Testing'

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        #3 convolutions with increasing depth, 3 kernel size, then max pooling, then fully connected layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_cnn():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCNN(num_classes=len(train_dataset.classes)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epoch_losses = [] #to track losses for visualization
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        
    #evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Custom CNN Accuracy on Test Set: {100 * correct / total:.2f}%")
    
    model_path = "/kaggle/working/custom_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to {model_path}")
    graph_path = "/kaggle/working/cnn_loss_graph.png"
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('CNN Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, EPOCHS + 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(graph_path)
    plt.close()
    print(f"--> Saved loss graph to {graph_path}")

if __name__ == "__main__":
        print(f"Found Training Directory: {TRAIN_DIR}")
        print(f"Found Testing Directory: {TEST_DIR}")
        
        train_cnn()
        
        #zipping to download from kaggle
        zip_filename = '/kaggle/working/model_outputs.zip'
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            if os.path.exists('/kaggle/working/custom_cnn.pth'):
                zipf.write('/kaggle/working/custom_cnn.pth', arcname='custom_cnn.pth')
            if os.path.exists('/kaggle/working/cnn_loss_graph.png'):
                zipf.write('/kaggle/working/cnn_loss_graph.png', arcname='cnn_loss_graph.png')
        print(f"📦 All done! Download your zip file from the 'Output' /kaggle/working/ directory: {zip_filename}")
        