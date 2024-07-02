import logging
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

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

from pbft.pbft_node import PBFTNode
from pbft.network import Network

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directory for client data if it does not exist
os.makedirs('data', exist_ok=True)
logger.info("Directory for client data created.")

# Load and preprocess data
df = load_and_preprocess_data('data/credit_card_2023.csv')
logger.info("Data loaded and preprocessed.")

# Partition data for clients
n_clients = 10
client_data = partition_data_for_clients(df, n_clients)
logger.info(f"Data partitioned for {n_clients} clients.")

# Save client-specific datasets
for i, client_df in enumerate(client_data):
    client_df.to_csv(f'data/client_{i+1}_data.csv', index=False)
logger.info("Client-specific datasets saved.")

# Prepare client-specific datasets
client_partitions = prepare_client_data(client_data)
logger.info("Client-specific datasets prepared.")

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
logger.info("Federated learning clients initialized.")

# Train local models with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
logger.info("Starting training for federated clients.")
histories = []
for client_id, client in enumerate(clients):
    X_train, X_test, y_train, y_test = client_partitions[client_id]
    logger.info(f"Training client {client_id + 1}")
    history = client.train_local_model(epochs=10, callbacks=[early_stopping], validation_data=(X_test, y_test))
    histories.append(history)
    logger.info(f"Client {client_id + 1} training completed.")

# Apply anomaly detection and adversarial training
anomaly_detection = AnomalyDetection()
anomaly_detection.train(X_train)
logger.info("Anomaly detection model trained.")

# Initializing model before adversarial training to avoid undefined variable issue
global_model_fedavg = create_model(X_train.shape[1])
adversarial_training = AdversarialTraining(global_model_fedavg)  # Assuming using the FedAvg model for adversarial training
adversarial_training.apply_adversarial_training(X_train, y_train)
logger.info("Adversarial training applied.")

# Prepare models and training data for aggregation
client_models_data = [{'model': client.model, 'train_data': client.train_data} for client in clients]
logger.info("Models and training data prepared for aggregation.")

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
logger.info("Plot directory created.")
for i, history in enumerate(histories):
    plot_training_history([history], os.path.join(plot_dir, f'client_{i+1}_training_history.png'), title=f'Client {i+1} Training History')
logger.info("Training histories plotted.")

# Plot global model performance
client_accuracies_fedavg = [client.model.evaluate(client_partitions[i][1], client_partitions[i][3])[1] for i, client in enumerate(clients)]
client_accuracies_fedadam = [client.model.evaluate(client_partitions[i][1], client_partitions[i][3])[1] for i, client in enumerate(clients)]
client_accuracies_hybrid = [client.model.evaluate(client_partitions[i][1], client_partitions[i][3])[1] for i, client in enumerate(clients)]

plot_global_model_performance(global_model_fedavg.evaluate(X_test, y_test)[1], client_accuracies_fedavg, os.path.join(plot_dir, 'global_model_performance_fedavg.png'), title='Global Model vs Client Models Performance (FedAvg)')
plot_global_model_performance(global_model_fedadam.evaluate(X_test, y_test)[1], client_accuracies_fedadam, os.path.join(plot_dir, 'global_model_performance_fedadam.png'), title='Global Model vs Client Models Performance (FedAdam)')
plot_global_model_performance(global_model_hybrid.evaluate(X_test, y_test)[1], client_accuracies_hybrid, os.path.join(plot_dir, 'global_model_performance_hybrid.png'), title='Global Model vs Client Models Performance (Hybrid)')
logger.info("Global model performance plotted.")

# Plot confusion matrices
plot_confusion_matrix(y_test, y_pred_fedavg, os.path.join(plot_dir, 'confusion_matrix_fedavg.png'), title='Confusion Matrix (FedAvg)')
plot_confusion_matrix(y_test, y_pred_fedadam, os.path.join(plot_dir, 'confusion_matrix_fedadam.png'), title='Confusion Matrix (FedAdam)')
plot_confusion_matrix(y_test, y_pred_hybrid, os.path.join(plot_dir, 'confusion_matrix_hybrid.png'), title='Confusion Matrix (Hybrid)')
plot_normalized_confusion_matrix(y_test, y_pred_fedavg, os.path.join(plot_dir, 'normalized_confusion_matrix_fedavg.png'), title='Normalized Confusion Matrix (FedAvg)')
plot_normalized_confusion_matrix(y_test, y_pred_fedadam, os.path.join(plot_dir, 'normalized_confusion_matrix_fedadam.png'), title='Normalized Confusion Matrix (FedAdam)')
plot_normalized_confusion_matrix(y_test, y_pred_hybrid, os.path.join(plot_dir, 'normalized_confusion_matrix_hybrid.png'), title='Normalized Confusion Matrix (Hybrid)')
logger.info("Confusion matrices plotted.")

# Define load_blockchain and save_blockchain functions
def load_blockchain():
    if os.path.exists('blockchain.pkl'):
        with open('blockchain.pkl', 'rb') as f:
            blockchain = pickle.load(f)
            blockchain.initialize_pbft_nodes()  # Ensure PBFT nodes are initialized
    else:
        blockchain = Blockchain()
        blockchain.create_genesis_block()
        save_blockchain(blockchain)
    return blockchain

def save_blockchain(blockchain):
    with open('blockchain.pkl', 'wb') as f:
        pickle.dump(blockchain, f)
logger.info("Blockchain functions defined.")

# Blockchain and security steps
# Initialize blockchain
blockchain = load_blockchain()
logger.info("Blockchain initialized.")

# Add a new block to the blockchain to ensure there is more than one block
blockchain.add_block("New Block Data")
save_blockchain(blockchain)
logger.info("New block added to blockchain.")

# Initialize smart contract and record model updates
smart_contract = SmartContract()
for client in clients:
    smart_contract.verify_and_record(client.model.get_weights())
logger.info("Model updates recorded on blockchain.")

# Print blockchain and smart contract ledger for verification
blockchain.print_chain()
smart_contract.print_ledger()

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
latest_block_hash = blockchain.get_latest_block().hash  # Retrieve the latest block's hash
genesis_block_index = 0  # Retrieve the genesis block by index
retrieve_and_print_block_details(blockchain, latest_block_hash, genesis_block_index)

# Evaluate the global model again after security enhancements
y_pred_enhanced = (global_model_fedavg.predict(X_test) > 0.5).astype("int32")
tp, fp, fn, tn, recall, precision, f1 = calculate_metrics(y_test, y_pred_enhanced)
logger.info(f'Test Metrics after security enhancements: TP={tp}, FP={fp}, FN={fn}, TN={tn}, Recall={recall:.2f}, Precision={precision:.2f}, F1-Score={f1:.2f}')

# PBFT integration
# Initialize network and nodes
network = Network()
nodes = [PBFTNode(i, network) for i in range(n_clients)]
logger.info("PBFT network and nodes initialized.")

# Add nodes to network
for node in nodes:
    network.add_node(node)
logger.info("Nodes added to PBFT network.")

# Add blocks using PBFT
for i in range(n_clients):
    nodes[i].pre_prepare(f"Client {i+1} data block")
logger.info("Blocks added using PBFT.")

# Visualizations for security enhancements
plot_confusion_matrix(y_test, y_pred_enhanced, os.path.join(plot_dir, 'confusion_matrix_enhanced.png'), title='Confusion Matrix (Enhanced)')
plot_normalized_confusion_matrix(y_test, y_pred_enhanced, os.path.join(plot_dir, 'normalized_confusion_matrix_enhanced.png'), title='Normalized Confusion Matrix (Enhanced)')

# Plot feature distribution for each feature
feature_names = [f'Feature {i+1}' for i in range(X_train.shape[1])]
for i, feature_name in enumerate(feature_names):
    plot_feature_distribution(X_train[:, i], feature_name, os.path.join(plot_dir, f'feature_distribution_{i+1}.png'), title=f'Feature Distribution {feature_name} (Enhanced)')
logger.info("Feature distributions plotted.")

anomalies = X_train[anomaly_detection.detect_anomalies(X_train) == -1]
plot_anomaly_detection(X_train, anomalies, os.path.join(plot_dir, 'anomaly_detection.png'), title='Anomaly Detection (Enhanced)')
logger.info("Anomaly detection plotted.")

# New visualizations for client data and performance
feature_indices = [0, 1]  # Indices of features to visualize (adjust as needed)
for i, (X_train, X_test, y_train, y_test) in enumerate(client_partitions):
    plot_client_data_distribution(X_train, y_train, feature_indices, os.path.join(plot_dir, f'client_{i+1}_data_distribution.png'), client_id=i+1, title=f'Client {i+1} Data Distribution')
    plot_client_model_performance(histories[i], os.path.join(plot_dir, f'client_{i+1}_model_performance.png'), client_id=i+1, title=f'Client {i+1} Model Performance')
logger.info("Client data and performance visualizations plotted.")
