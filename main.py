import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import logging
import os
from utils.data_preprocessing import load_and_preprocess_data, partition_data_for_clients, prepare_client_data
from federated_learning.client import FederatedClient
from federated_learning.aggregator import CentralAggregator
from blockchain.blockchain import Blockchain
from blockchain.smart_contract import SmartContract
from security.anomaly_detection import AnomalyDetection
from security.adversarial_training import AdversarialTraining
from visualization import (plot_training_history, plot_global_model_performance, 
                           plot_confusion_matrix, plot_normalized_confusion_matrix, 
                           plot_feature_distribution, plot_anomaly_detection, 
                           plot_client_data_distribution, plot_client_model_performance)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import numpy as np
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directory for client data if it does not exist
os.makedirs('data', exist_ok=True)

# Load and preprocess data
df = load_and_preprocess_data('data/credit_card_2023.csv')

# Partition data for clients
n_clients = 5
client_data = partition_data_for_clients(df, n_clients)

# Save client-specific datasets
for i, client_df in enumerate(client_data):
    client_df.to_csv(f'data/client_{i+1}_data.csv', index=False)

# Prepare client-specific datasets
client_partitions = prepare_client_data(client_data)

# Define a simple model
def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize federated learning clients
clients = [FederatedClient((X_train, y_train), create_model(X_train.shape[1])) for (X_train, X_test, y_train, y_test) in client_partitions]

# Train local models with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
logger.info("Starting training for federated clients.")
histories = []
for client_id, client in enumerate(clients):
    X_train, X_test, y_train, y_test = client_partitions[client_id]
    logger.info(f"Training client {client_id + 1}")
    history = client.train_local_model(epochs=5, callbacks=[early_stopping], validation_data=(X_test, y_test))
    histories.append(history)
    logger.info(f"Client {client_id + 1} training completed.")

# Prepare models and training data for aggregation
client_models_data = [{'model': client.model, 'train_data': client.train_data} for client in clients]

# Aggregation and evaluation for FedAvg
aggregator_fedavg = CentralAggregator(create_model(X_train.shape[1]))
logger.info("Starting model aggregation using FedAvg.")
global_model_fedavg = aggregator_fedavg.aggregate_models(client_models_data, method='fedavg')
logger.info("Model aggregation using FedAvg completed.")

# FedAdam Aggregator
class FederatedAdamAggregator(CentralAggregator):
    def aggregate_models(self, client_models_data, method='fedadam', beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        if method == 'fedadam':
            num_clients = len(client_models_data)
            client_weights = [client['model'].get_weights() for client in client_models_data]

            aggregated_weights = []
            m = [0] * len(client_weights[0])
            v = [0] * len(client_weights[0])

            for weights in zip(*client_weights):
                weight_sum = np.sum(weights, axis=0)
                aggregated_weights.append(weight_sum / num_clients)

            for i, weight in enumerate(aggregated_weights):
                m[i] = beta_1 * m[i] + (1 - beta_1) * weight
                v[i] = beta_2 * v[i] + (1 - beta_2) * (weight ** 2)
                m_hat = m[i] / (1 - beta_1)
                v_hat = v[i] / (1 - beta_2)
                aggregated_weights[i] = m_hat / (np.sqrt(v_hat) + epsilon)

            self.global_model.set_weights(aggregated_weights)
        return self.global_model

# Use FederatedAdamAggregator for aggregation
aggregator_fedadam = FederatedAdamAggregator(create_model(X_train.shape[1]))
logger.info("Starting model aggregation using FedAdam.")
global_model_fedadam = aggregator_fedadam.aggregate_models(client_models_data, method='fedadam')
logger.info("Model aggregation using FedAdam completed.")

# Hybrid Model Aggregation
def hybrid_aggregate(client_models_data, initial_model, iterations=3):
    # Perform initial FedAvg aggregation
    aggregator_fedavg = CentralAggregator(initial_model)
    global_model = aggregator_fedavg.aggregate_models(client_models_data, method='fedavg')
    
    # Perform alternating FedAdam and FedAvg
    for _ in range(iterations):
        aggregator_fedadam = FederatedAdamAggregator(global_model)
        global_model = aggregator_fedadam.aggregate_models(client_models_data, method='fedadam')
        aggregator_fedavg = CentralAggregator(global_model)
        global_model = aggregator_fedavg.aggregate_models(client_models_data, method='fedavg')
    
    return global_model

# Perform hybrid aggregation
initial_model = create_model(X_train.shape[1])
logger.info("Starting hybrid model aggregation.")
global_model_hybrid = hybrid_aggregate(client_models_data, initial_model)
logger.info("Hybrid model aggregation completed.")

# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return tp, fp, fn, tn, recall, precision, f1

# Validate global models on the validation set of the first client (or any chosen validation set)
X_train, X_test, y_train, y_test = client_partitions[0]

# Evaluate FedAvg model
y_pred_fedavg = (global_model_fedavg.predict(X_test) > 0.5).astype("int32")
tp, fp, fn, tn, recall, precision, f1 = calculate_metrics(y_test, y_pred_fedavg)
logger.info(f'FedAvg Validation Metrics: TP={tp}, FP={fp}, FN={fn}, TN={tn}, Recall={recall:.2f}, Precision={precision:.2f}, F1-Score={f1:.2f}')

# Evaluate FedAdam model
y_pred_fedadam = (global_model_fedadam.predict(X_test) > 0.5).astype("int32")
tp, fp, fn, tn, recall, precision, f1 = calculate_metrics(y_test, y_pred_fedadam)
logger.info(f'FedAdam Validation Metrics: TP={tp}, FP={fp}, FN={fn}, TN={tn}, Recall={recall:.2f}, Precision={precision:.2f}, F1-Score={f1:.2f}')

# Evaluate Hybrid model
y_pred_hybrid = (global_model_hybrid.predict(X_test) > 0.5).astype("int32")
tp, fp, fn, tn, recall, precision, f1 = calculate_metrics(y_test, y_pred_hybrid)
logger.info(f'Hybrid Validation Metrics: TP={tp}, FP={fp}, FN={fn}, TN={tn}, Recall={recall:.2f}, Precision={precision:.2f}, F1-Score={f1:.2f}')

# Plot training histories
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)
plot_training_history(histories, os.path.join(plot_dir, 'training_history.png'))

# Plot FedAvg vs FedAdam vs Hybrid performance
client_accuracies_fedavg = [client.model.evaluate(client_partitions[i][1], client_partitions[i][3])[1] for i, client in enumerate(clients)]
client_accuracies_fedadam = [client.model.evaluate(client_partitions[i][1], client_partitions[i][3])[1] for i, client in enumerate(clients)]
client_accuracies_hybrid = [client.model.evaluate(client_partitions[i][1], client_partitions[i][3])[1] for i, client in enumerate(clients)]

plot_global_model_performance(global_model_fedavg.evaluate(X_test, y_test)[1], client_accuracies_fedavg, os.path.join(plot_dir, 'global_model_performance_fedavg.png'))
plot_global_model_performance(global_model_fedadam.evaluate(X_test, y_test)[1], client_accuracies_fedadam, os.path.join(plot_dir, 'global_model_performance_fedadam.png'))
plot_global_model_performance(global_model_hybrid.evaluate(X_test, y_test)[1], client_accuracies_hybrid, os.path.join(plot_dir, 'global_model_performance_hybrid.png'))

# Plot confusion matrices
plot_confusion_matrix(y_test, y_pred_fedavg, os.path.join(plot_dir, 'confusion_matrix_fedavg.png'))
plot_confusion_matrix(y_test, y_pred_fedadam, os.path.join(plot_dir, 'confusion_matrix_fedadam.png'))
plot_confusion_matrix(y_test, y_pred_hybrid, os.path.join(plot_dir, 'confusion_matrix_hybrid.png'))
plot_normalized_confusion_matrix(y_test, y_pred_fedavg, os.path.join(plot_dir, 'normalized_confusion_matrix_fedavg.png'))
plot_normalized_confusion_matrix(y_test, y_pred_fedadam, os.path.join(plot_dir, 'normalized_confusion_matrix_fedadam.png'))
plot_normalized_confusion_matrix(y_test, y_pred_hybrid, os.path.join(plot_dir, 'normalized_confusion_matrix_hybrid.png'))

# Blockchain and security steps remain the same
# Initialize blockchain
blockchain = Blockchain()
logger.info("Blockchain initialized.")

# Initialize smart contract and record model updates
smart_contract = SmartContract()
for client in clients:
    smart_contract.verify_and_record(client.model.get_weights())
logger.info("Model updates recorded on blockchain.")

# Print blockchain and smart contract ledger for verification
blockchain.print_chain()
smart_contract.print_ledger()

# Initialize security modules
anomaly_detection = AnomalyDetection()
anomaly_detection.train(X_train)
adversarial_training = AdversarialTraining(global_model_fedavg)  # Assuming using the FedAvg model for adversarial training
adversarial_training.apply_adversarial_training(X_train, y_train)
logger.info("Security enhancements applied.")

# Evaluate the global model again after security enhancements
y_pred_enhanced = (global_model_fedavg.predict(X_test) > 0.5).astype("int32")
tp, fp, fn, tn, recall, precision, f1 = calculate_metrics(y_test, y_pred_enhanced)
logger.info(f'Test Metrics after security enhancements: TP={tp}, FP={fp}, FN={fn}, TN={tn}, Recall={recall:.2f}, Precision={precision:.2f}, F1-Score={f1:.2f}')

# Visualizations for security enhancements
plot_confusion_matrix(y_test, y_pred_enhanced, os.path.join(plot_dir, 'confusion_matrix_enhanced.png'))
plot_normalized_confusion_matrix(y_test, y_pred_enhanced, os.path.join(plot_dir, 'normalized_confusion_matrix_enhanced.png'))

# Plot feature distribution for each feature
feature_names = [f'Feature {i+1}' for i in range(X_train.shape[1])]
for i, feature_name in enumerate(feature_names):
    plot_feature_distribution(X_train[:, i], feature_name, os.path.join(plot_dir, f'feature_distribution_{i+1}.png'))

anomalies = X_train[anomaly_detection.detect(X_train) == -1]
plot_anomaly_detection(X_train, anomalies, os.path.join(plot_dir, 'anomaly_detection.png'))

# New visualizations for client data and performance
feature_indices = [0, 1]  # Indices of features to visualize (adjust as needed)
for i, (X_train, X_test, y_train, y_test) in enumerate(client_partitions):
    plot_client_data_distribution(X_train, y_train, feature_indices, os.path.join(plot_dir, f'client_{i+1}_data_distribution.png'))
    plot_client_model_performance(histories[i], os.path.join(plot_dir, f'client_{i+1}_model_performance.png'))

# Function to retrieve and print block details
def retrieve_and_print_block_details(blockchain, block_hash, block_index):
    block_by_hash = blockchain.get_block_by_hash(block_hash)
    block_by_index = blockchain.get_block_by_index(block_index)

    if block_by_hash:
        logger.info(f"Block by hash: {block_by_hash}")
    else:
        logger.info(f"No block found with hash: {block_hash}")

    if block_by_index:
        logger.info(f"Block by index: {block_by_index}")
    else:
        logger.info(f"No block found with index: {block_index}")

# Retrieve and print block details for verification
block_hash = blockchain.chain[-1].hash  # Replace with the hash of the block you want to retrieve
block_index = blockchain.chain[-1].index  # Replace with the index of the block you want to retrieve
retrieve_and_print_block_details(blockchain, block_hash, block_index)
