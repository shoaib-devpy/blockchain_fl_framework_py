# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def plot_training_history(histories, filename):
    for i, history in enumerate(histories):
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Client {i+1} Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(filename.replace('.png', f'_accuracy.png'))
        plt.close()

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Client {i+1} Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(filename.replace('.png', f'_loss.png'))
        plt.close()


def plot_global_model_performance(global_accuracy, client_accuracies, save_path, title='Global Model vs Client Models Performance'):
    plt.figure()
    client_ids = range(1, len(client_accuracies) + 1)
    plt.bar(client_ids, client_accuracies, label='Client Model Accuracies')
    plt.axhline(y=global_accuracy, color='r', linestyle='--', label='Global Model Accuracy')
    plt.xlabel('Client ID')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix (Non-Normalized)'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_normalized_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix (Normalized)'):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_feature_distribution(feature_data, feature_name, save_path):
    plt.figure()
    sns.histplot(feature_data, kde=True)
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def plot_anomaly_detection(data, anomalies, save_path):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    plt.figure(figsize=(10, 7))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], label='Normal Data')
    plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', label='Anomalies')
    plt.title('Anomaly Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_client_data_distribution(X_train, y_train, feature_indices, save_path, title='Client Data Distribution'):
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train[y_train == 0, feature_indices[0]], X_train[y_train == 0, feature_indices[1]], label='Class 0')
    plt.scatter(X_train[y_train == 1, feature_indices[0]], X_train[y_train == 1, feature_indices[1]], label='Class 1', color='r')
    plt.title(title)
    plt.xlabel(f'Feature {feature_indices[0] + 1}')
    plt.ylabel(f'Feature {feature_indices[1] + 1}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_client_model_performance(history, save_path, title='Client Model Performance'):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()
