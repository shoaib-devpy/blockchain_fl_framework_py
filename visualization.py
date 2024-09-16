import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler

def plot_training_history(histories, filename, title):
    for i, history in enumerate(histories):
        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{title} - Client {i+1} Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(filename.replace('.png', f'_client_{i+1}_accuracy.png'))
        plt.close()

        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{title} - Client {i+1} Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(filename.replace('.png', f'_client_{i+1}_loss.png'))
        plt.close()

def plot_global_model_performance(global_accuracy, client_accuracies, save_path, title):
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

def plot_confusion_matrix(y_true, y_pred, save_path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Non-Normalized) - {title}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_normalized_confusion_matrix(y_true, y_pred, save_path, title):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Confusion Matrix (Normalized) - {title}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_feature_distribution(feature_data, feature_name, save_path, title):
    plt.figure()
    sns.histplot(feature_data, kde=True)
    plt.title(f'Distribution of {feature_name} - {title}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def plot_anomaly_detection(data, anomalies, save_path, title):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    anomalies_scaled = scaler.transform(anomalies)
    plt.figure(figsize=(10, 7))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], label='Normal Data')
    plt.scatter(anomalies_scaled[:, 0], anomalies_scaled[:, 1], color='r', label='Anomalies')
    plt.title(f'Anomaly Detection - {title}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_client_data_distribution(X_train, y_train, feature_indices, save_path, client_id, title):
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train[y_train == 0, feature_indices[0]], X_train[y_train == 0, feature_indices[1]], label='Class 0')
    plt.scatter(X_train[y_train == 1, feature_indices[0]], X_train[y_train == 1, feature_indices[1]], label='Class 1', color='r')
    plt.title(f'Client {client_id} Data Distribution - {title}')
    plt.xlabel(f'Feature {feature_indices[0] + 1}')
    plt.ylabel(f'Feature {feature_indices[1] + 1}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_client_model_performance(history, save_path, client_id, title):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Client {client_id} Model Performance - {title}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()

def plot_blockchain_visualization(blockchain, save_path, title):
    block_indices = [block.index for block in blockchain.chain]
    block_hashes = [block.hash for block in blockchain.chain]
    block_data = [block.data for block in blockchain.chain]

    plt.figure(figsize=(15, 10))
    plt.plot(block_indices, block_hashes, marker='o', linestyle='-', color='b')
    plt.title(f'{title} - Block Hashes Over Time')
    plt.xlabel('Block Index')
    plt.ylabel('Block Hash')
    plt.grid(True)
    plt.savefig(save_path.replace('.png', '_hashes.png'))
    plt.close()

    plt.figure(figsize=(15, 10))
    plt.plot(block_indices, block_data, marker='x', linestyle='--', color='r')
    plt.title(f'{title} - Block Data Over Time')
    plt.xlabel('Block Index')
    plt.ylabel('Block Data')
    plt.grid(True)
    plt.savefig(save_path.replace('.png', '_data.png'))
    plt.close()

def plot_pbft_visualization(pbft_data, save_path, title):
    plt.figure(figsize=(15, 10))
    plt.plot(pbft_data)
    plt.title(f'{title} - PBFT Node States')
    plt.xlabel('Node')
    plt.ylabel('State')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_algorithm_performance(algorithm_name, global_accuracy, client_accuracies, save_path):
    plt.figure()
    client_ids = range(1, len(client_accuracies) + 1)
    plt.bar(client_ids, client_accuracies, label='Client Model Accuracies')
    plt.axhline(y=global_accuracy, color='r', linestyle='--', label='Global Model Accuracy')
    plt.xlabel('Client ID')
    plt.ylabel('Accuracy')
    plt.title(f'Algorithm Performance - {algorithm_name}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_metrics(metrics_dict, save_path, title):
    model_names = list(metrics_dict.keys())
    metrics = ['TP', 'FP', 'FN', 'TN', 'Recall', 'Precision', 'F1-Score', 'AUC']

    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=(10, 7))
        values = [metrics_dict[model][metric] for model in model_names]
        plt.bar(model_names, values)
        plt.title(f'{metric} Comparison - {title}')
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.savefig(save_path.replace('.png', f'_{metric.lower()}.png'))
        plt.close()

def plot_roc_curve(y_true, y_pred, save_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_pred, save_path, title):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(save_path)
    plt.close()

# New function to plot overall accuracy and loss
def plot_accuracy_loss(history, save_path, title):
    # Check if history is a list (multiple histories)
    if isinstance(history, list):
        plt.figure()
        for i, hist in enumerate(history):
            plt.plot(hist.history['accuracy'], label=f'Client {i+1} Training Accuracy')
            plt.plot(hist.history['val_accuracy'], label=f'Client {i+1} Validation Accuracy')
        plt.title(f'{title} - Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(save_path.replace('.png', '_accuracy.png'))
        plt.close()

        plt.figure()
        for i, hist in enumerate(history):
            plt.plot(hist.history['loss'], label=f'Client {i+1} Training Loss')
            plt.plot(hist.history['val_loss'], label=f'Client {i+1} Validation Loss')
        plt.title(f'{title} - Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(save_path.replace('.png', '_loss.png'))
        plt.close()

    # If it's a single history object
    else:
        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{title} - Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(save_path.replace('.png', '_accuracy.png'))
        plt.close()

        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{title} - Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(save_path.replace('.png', '_loss.png'))
        plt.close()
