import os
import numpy as np
import joblib
import zipfile
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


IMG_SIZE = 128
TRAIN_DIR = '/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Training'
TEST_DIR = '/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Testing'

def train_logistic_regression():
    train_data = datasets.ImageFolder(TRAIN_DIR, transform=transforms.Resize((IMG_SIZE, IMG_SIZE)))
    test_data = datasets.ImageFolder(TEST_DIR, transform=transforms.Resize((IMG_SIZE, IMG_SIZE)))

    #flatten images as sklearn requires 1D
    def extract_features_labels(dataset):
        features, labels = [], []
        for img, label in dataset:
            img_arr = np.array(img).flatten()
            features.append(img_arr)
            labels.append(label)
        return np.array(features), np.array(labels)
    
    X_train, y_train = extract_features_labels(train_data)
    X_test, y_test = extract_features_labels(test_data)
    
    # Training: max_iter=500 gives the solver enough time to find a solution without stopping early
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    
    print("Evaluating")
    preds = model.predict(X_test)
    print("\nLogistic Regression Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=train_data.classes))
    log_model_path = "/kaggle/working/logistic_model.joblib"
    joblib.dump(model, log_model_path)

if __name__ == "__main__":
    if os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR):
        print(f"Found Training Directory: {TRAIN_DIR}")
        print(f"Found Testing Directory: {TEST_DIR}")
        train_logistic_regression()
        
        #zipping for easy download
        print("\n--- Zipping Output Files ---")
        zip_filename = '/kaggle/working/logistic_output.zip'
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            if os.path.exists('/kaggle/working/logistic_model.joblib'):
                zipf.write('/kaggle/working/logistic_model.joblib', arcname='logistic_model.joblib')
                
    else:
        print("Dataset not found")