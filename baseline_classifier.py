import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load features and labels
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

# Flatten features
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Scale features
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_val_flat = scaler.transform(X_val_flat)

# Convert one-hot labels to single class labels
if y_train.shape[1] == 2:
    y_train_labels = np.argmax(y_train, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)
else:
    y_train_labels = y_train
    y_val_labels = y_val

# Train logistic regression classifier
clf = LogisticRegression(max_iter=2000, solver='saga', verbose=1)
clf.fit(X_train_flat, y_train_labels)

# Predict and evaluate
val_preds = clf.predict(X_val_flat)
val_acc = accuracy_score(y_val_labels, val_preds)
print(f"Baseline validation accuracy: {val_acc:.4f}")
print("Classification report:")
print(classification_report(y_val_labels, val_preds, target_names=['fake', 'real'])) 