import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping
import shap

np.random.seed(42)
tf.random.set_seed(42)

os.makedirs('output', exist_ok=True)
os.makedirs('output/plots', exist_ok=True)
os.makedirs('output/models', exist_ok=True)

data_file = 'data/cleaned_combined_person.csv'

print("Loading dataset...")
df = pd.read_csv(data_file)
print(f"Loaded {len(df)} person records")

print("Dataset columns:", df.columns.tolist())

print("Processing data...")

df['FATAL'] = (df['INJ_SEV'] == 4).astype(int)

print("Class distribution (Fatal vs Non-Fatal):")
print(df['FATAL'].value_counts())

selected_features = [
    'AGE', 'SEX', 'PER_TYP',  
    'SEAT_POS', 'REST_USE', 'AIR_BAG',  
    'EJECTION', 'DRINKING', 'DRUGS',  
    'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE'
]

existing_features = [f for f in selected_features if f in df.columns]
print(f"Using features: {existing_features}")

df = df.dropna(subset=existing_features + ['FATAL'])

X = df[existing_features]
y = df['FATAL']

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")

categorical_features = ['SEX', 'PER_TYP', 'SEAT_POS', 'REST_USE', 'AIR_BAG', 
                        'EJECTION', 'DRINKING', 'DRUGS', 'MONTH', 'DAY', 'HOUR']
categorical_features = [f for f in categorical_features if f in existing_features]

numeric_features = [f for f in existing_features if f not in categorical_features]

print(f"Categorical features: {categorical_features}")
print(f"Numeric features: {numeric_features}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
    X_test_processed = X_test_processed.toarray()

print(f"Processed feature shape: {X_train_processed.shape}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)

print(f"Resampled training set size: {X_train_res.shape[0]}")

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_res.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_res, y_train_res,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

print("\nEvaluating the model...")
loss, accuracy = model.evaluate(X_test_processed, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_prob = model.predict(X_test_processed)
y_pred = (y_pred_prob > 0.5).astype(int)
y_pred = y_pred.flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Non-Fatal", "Fatal"],
            yticklabels=["Non-Fatal", "Fatal"])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('output/plots/confusion_matrix.png')
plt.close()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('output/plots/training_history.png')
plt.close()

from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('output/plots/roc_curve.png')
plt.close()

plt.figure(figsize=(10, 6))
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 1.0)
plt.title('Classification Metrics')
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
plt.savefig('output/plots/metrics_summary.png')
plt.close()

model.save('output/models/fatality_classification_model.h5')

