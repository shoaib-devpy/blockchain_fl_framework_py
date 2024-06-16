# main.py
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import logging
import os
from utils.data_preprocessing import load_and_preprocess_data
from federated_learning.client import FederatedClient
from federated_learning.aggregator import CentralAggregator
from blockchain.blockchain import Blockchain
from blockchain.smart_contract import SmartContract
from security.anomaly_detection import AnomalyDetection
from security.adversarial_training import AdversarialTraining
from visualization import plot_training_history, plot_global_model_performance, plot_confusion_matrix, plot_feature_distribution, plot_anomaly_detection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/credit_card_2023.csv')

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define a simple model
def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),  # Corrected this line to pass shape as a tuple
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize federated learning clients
clients = [FederatedClient((X_train, y_train), create_model(X_train.shape[1])) for _ in range(5)]

# Train local models with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
logger.info("Starting training for federated clients.")
histories = [client.train_local_model(epochs=5, callbacks=[early_stopping], validation_data=(X_val, y_val)) for client in clients]

# Initialize central aggregator and aggregate models
aggregator = CentralAggregator(create_model(X_train.shape[1]))
global_model = aggregator.aggregate_models([client.model for client in clients])
logger.info("Model aggregation completed.")

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
adversarial_training = AdversarialTraining(global_model)
adversarial_training.apply_adversarial_training(X_train, y_train)
logger.info("Security enhancements applied.")

# Evaluate the global model
loss, accuracy = global_model.evaluate(X_test, y_test)
logger.info(f'Test Accuracy: {accuracy:.2f}')

# Create directory for plots
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

# Visualizations
client_accuracies = [client.model.evaluate(X_test, y_test)[1] for client in clients]
plot_training_history(histories, os.path.join(plot_dir, 'training_history.png'))
plot_global_model_performance(accuracy, client_accuracies, os.path.join(plot_dir, 'global_model_performance.png'))
plot_confusion_matrix(y_test, (global_model.predict(X_test) > 0.5).astype("int32"), os.path.join(plot_dir, 'confusion_matrix.png'))

# Plot feature distribution for each feature
feature_names = [f'Feature {i+1}' for i in range(X_train.shape[1])]
for i, feature_name in enumerate(feature_names):
    plot_feature_distribution(X_train[:, i], feature_name, os.path.join(plot_dir, f'feature_distribution_{i+1}.png'))

anomalies = X_train[anomaly_detection.detect(X_train) == -1]
plot_anomaly_detection(X_train, anomalies, os.path.join(plot_dir, 'anomaly_detection.png'))
