# main.py
import logging
from utils.data_preprocessing import load_and_preprocess_data
from federated_learning.client import FederatedClient
from federated_learning.aggregator import CentralAggregator
from blockchain.blockchain import Blockchain
from blockchain.smart_contract import SmartContract
from security.anomaly_detection import AnomalyDetection
from security.adversarial_training import AdversarialTraining
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

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
client_models = [
    client.train_local_model(epochs=10, callbacks=[early_stopping], validation_data=(X_val, y_val))
    for client in clients
]

# Initialize central aggregator and aggregate models
aggregator = CentralAggregator(create_model(X_train.shape[1]))
global_model = aggregator.aggregate_models(client_models)
logger.info("Model aggregation completed.")

# Initialize blockchain
blockchain = Blockchain()
logger.info("Blockchain initialized.")

# Initialize smart contract and record model updates
smart_contract = SmartContract()
for model in client_models:
    smart_contract.verify_and_record(model.get_weights())
logger.info("Model updates recorded on blockchain.")

# Initialize security modules
anomaly_detection = AnomalyDetection()
anomaly_detection.train(X_train)
adversarial_training = AdversarialTraining(global_model)
adversarial_training.apply_adversarial_training(X_train, y_train)
logger.info("Security enhancements applied.")

# Evaluate the global model
loss, accuracy = global_model.evaluate(X_test, y_test)
logger.info(f'Test Accuracy: {accuracy:.2f}')
