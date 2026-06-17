#efficientnet
import os
import time
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

IMG_SIZE = 224 
BATCH_SIZE = 32
EPOCHS = 40 
LR = 5e-05 
DROPOUT_RATE = 0.2 
WEIGHT_DECAY = 1e-4 
EARLY_STOPPING_PATIENCE = 5 
TRAIN_DIR = '/kaggle/input/datasets/emaimkaggle/mri-dataset/Training'
VAL_DIR = '/kaggle/input/datasets/emaimkaggle/mri-dataset/Validation'
TEST_DIR = '/kaggle/input/datasets/emaimkaggle/mri-dataset/Testing'
WORKING_DIR = '/kaggle/working'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), 
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def run_efficientnet_experiment():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #batch size 1 for testing to get accurate latency per single scan
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 
    
    class_names = train_dataset.classes
    print(f"Classes detected: {class_names}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    #loading efficient net
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(num_ftrs, len(class_names))
    )
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    #training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    best_model_path = os.path.join(WORKING_DIR, 'efficientnet_best.pth')
    plot_path = os.path.join(WORKING_DIR, 'efficientnet_history.png')
    
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss, correct_train, total_train = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_acc = correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        
        #setting up schedular
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:02d}/{EPOCHS} [LR: {current_lr:.6f}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}%")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc*100:.2f}%")
        
        epochs_so_far = len(train_losses)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs_so_far+1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs_so_far+1), val_losses, label='Val Loss')
        plt.title('EfficientNet Loss History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs_so_far+1), train_accs, label='Train Acc')
        plt.plot(range(1, epochs_so_far+1), val_accs, label='Val Acc')
        plt.title('EfficientNet Accuracy History')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        #early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Val Loss improved! Saved best model weights.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")


        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
            break

    #evaluation 
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded BEST model weights for final testing.")
        
    model.eval()
    all_preds, all_labels = [], []
    
    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    end_time = time.time()
    total_scans = len(all_labels)
    latency_per_scan = (end_time - start_time) / total_scans
    test_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    class_recalls = recall_score(all_labels, all_preds, average=None)
    
    #evaluation mtrics report
    acc_pass = "PASS" if test_acc > 0.90 else "FAIL"
    print(f"Overall Accuracy:   {test_acc * 100:.2f}%  (Goal: > 90%) {acc_pass}")
    f1_pass = "PASS" if macro_f1 > 0.90 else "FAIL"
    print(f"Macro F1-Score:     {macro_f1:.4f}  (Goal: > 0.90) {f1_pass}")
    lat_pass = "PASS" if latency_per_scan < 2.0 else "FAIL"
    print(f"Inference Latency:  {latency_per_scan:.4f}s (Goal: < 2.0s) {lat_pass}")
    print("\nSensitivity (Recall) Per Class:")
    for i, class_name in enumerate(class_names):
        recall = class_recalls[i]
        c_pass = "PASS" if recall > 0.90 else "FAIL"
        print(f" - {class_name}: {recall * 100:.2f}% (Goal: > 90%) {c_pass}")
        
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    #zipping ffor easy kaggle download
    final_zip = os.path.join(WORKING_DIR, 'efficientnet_FINAL_outputs.zip')
    with zipfile.ZipFile(final_zip, 'w') as zipf:
        if os.path.exists(best_model_path):
            zipf.write(best_model_path, arcname='efficientnet_best.pth')
        if os.path.exists(plot_path):
            zipf.write(plot_path, arcname='efficientnet_history.png')
            
    print(f"All done! Download {final_zip}")

if __name__ == "__main__":
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR) and os.path.exists(TEST_DIR):
        run_efficientnet_experiment()
    else:
        print("directory not found")