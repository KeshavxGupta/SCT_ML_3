import os
import glob
import cv2
import numpy as np
from warnings import filterwarnings
from tqdm import tqdm

from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, confusion_matrix as sk_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

filterwarnings('ignore')

BASE_DATASET_PATH = 'dataset' 

TRAIN_DATA_PATH = os.path.join(BASE_DATASET_PATH, 'train')
TEST_DATA_PATH = os.path.join(BASE_DATASET_PATH, 'test')
VALIDATION_DATA_PATH = os.path.join(BASE_DATASET_PATH, 'validation')

IMG_SIZE = 100

TRAIN_LIMIT_PER_CATEGORY = 2000
TEST_LIMIT_PER_CATEGORY = 500

CATEGORIES = ["cats", "dogs"]
LABELS = {"cats": 0, "dogs": 1}

print("Starting Dog/Cat Image Classification Project (Task 03)")

print("\nVerifying Data Directory Structure...")

def verify_and_get_paths(base_path, categories):
    all_paths = []
    for category in categories:
        cat_path = os.path.join(base_path, category)
        if not os.path.exists(cat_path):
            print(f"Error: Directory '{cat_path}' not found. Check your folder structure.")
            exit()
        all_paths.extend(glob.glob(os.path.join(cat_path, '*.jpg')))
    return all_paths

train_image_paths = verify_and_get_paths(TRAIN_DATA_PATH, CATEGORIES)
print(f"Found {len(train_image_paths)} images in training directory.")

test_image_paths = verify_and_get_paths(TEST_DATA_PATH, CATEGORIES)
print(f"Found {len(test_image_paths)} images in test directory.")

validation_image_paths = verify_and_get_paths(VALIDATION_DATA_PATH, CATEGORIES)
print(f"Found {len(validation_image_paths)} images in validation directory.")
train_image_paths.extend(validation_image_paths)
print(f"Combined train and validation sets: {len(train_image_paths)} total images for processing.")

print("\nLoading and Preprocessing Images...")
print(f"Loading up to {TRAIN_LIMIT_PER_CATEGORY} images per category for training.")
print(f"Loading up to {TEST_LIMIT_PER_CATEGORY} images per category for testing.")

def load_and_preprocess_images(image_paths, img_size, labels_dict, limit_per_category=None):
    data = []
    skipped_count = 0
    category_counts = {category: 0 for category in labels_dict.keys()}

    for img_path in tqdm(image_paths):
        try:
            category_name = os.path.basename(os.path.dirname(img_path))
            label = labels_dict[category_name]

            if limit_per_category is not None and category_counts[category_name] >= limit_per_category:
                continue

            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                raise IOError("Image failed to load")

            img_resized = cv2.resize(img_array, (img_size, img_size))
            images_normalized = img_resized / 255.0
            data.append([images_normalized, label])
            category_counts[category_name] += 1
        except Exception as e:
            skipped_count += 1

    return data, skipped_count

train_data, skipped_train = load_and_preprocess_images(train_image_paths, IMG_SIZE, LABELS, limit_per_category=TRAIN_LIMIT_PER_CATEGORY)
test_data, skipped_test = load_and_preprocess_images(test_image_paths, IMG_SIZE, LABELS, limit_per_category=TEST_LIMIT_PER_CATEGORY)

print(f"\nLoaded {len(train_data)} training images (skipped {skipped_train} corrupt).")
print(f"Loaded {len(test_data)} test images (skipped {skipped_test} corrupt).")

X_data_loaded = np.array([item[0] for item in train_data])
y_data_loaded = np.array([item[1] for item in train_data])

X_test_data_loaded = np.array([item[0] for item in test_data])
y_test_data_loaded = np.array([item[1] for item in test_data])

print(f"\nTrain images shape: {X_data_loaded.shape}")
print(f"Test images shape: {X_test_data_loaded.shape}")

print("\nExtracting HOG Features...")
print(f"Processing {len(X_data_loaded)} training images and {len(X_test_data_loaded)} test images for HOG features.")

def extract_hog_features(image_arrays):
    features_list = []
    for img_array in tqdm(image_arrays):
        features = hog(img_array, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), transform_sqrt=True, feature_vector=True)
        features_list.append(features)
    return np.array(features_list)

X_train_hog = extract_hog_features(X_data_loaded)
X_test_hog = extract_hog_features(X_test_data_loaded)

print(f"\nHOG feature extraction complete. Train HOG shape: {X_train_hog.shape}, Test HOG shape: {X_test_hog.shape}")

print("Scaling HOG features using StandardScaler...")
hog_scaler = StandardScaler()
X_train_scaled = hog_scaler.fit_transform(X_train_hog)
X_test_scaled = hog_scaler.transform(X_test_hog)

print(f"HOG features scaled. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

print("\nTraining Support Vector Machine (SVM) with Hyperparameter Tuning...")

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.0001, 0.001, 0.01],
    'kernel': ['rbf']
}

base_svm = SVC(random_state=42)

grid_search = GridSearchCV(estimator=base_svm, param_grid=param_grid,
                           cv=3, verbose=2, n_jobs=-1, scoring='accuracy')

print("Starting GridSearchCV for SVM tuning (This will take significant time)...")
grid_search.fit(X_train_scaled, y_data_loaded)
print("GridSearchCV training complete.")

print("\nBest SVM Parameters Found:")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-validation Accuracy: {grid_search.best_score_:.4f}")

svm_model = grid_search.best_estimator_
print("\nSVM model optimized with best parameters.")

print("\nFinal Model Evaluation...")

y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test_data_loaded, y_pred)
print(f"\nFinal Model Accuracy on Test Set: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_data_loaded, y_pred))

print("\nConfusion Matrix:")
cm = sk_confusion_matrix(y_test_data_loaded, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Best Model')
plt.show()

print("\nProject finished!")