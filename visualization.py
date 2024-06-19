# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def plot_training_history(histories, save_path):
    plt.figure()
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Client {i+1} Training Accuracy')
        plt.plot(history.history['val_accuracy'], label=f'Client {i+1} Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()

def plot_global_model_performance(global_accuracy, client_accuracies, save_path):
    plt.figure()
    client_ids = range(1, len(client_accuracies) + 1)
    plt.bar(client_ids, client_accuracies, label='Client Model Accuracies')
    plt.axhline(y=global_accuracy, color='r', linestyle='--', label='Global Model Accuracy')
    plt.xlabel('Client ID')
    plt.ylabel('Accuracy')
    plt.title('Global Model vs Client Models Performance')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
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

def plot_client_data_distribution(X_train, y_train, save_path):
    plt.figure()
    sns.histplot(X_train[:, 0], kde=True, label='Feature 1')
    sns.histplot(X_train[:, 1], kde=True, label='Feature 2')
    plt.title('Client Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_client_model_performance(history, save_path):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Client Model Performance')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()
