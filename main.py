# main.py
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

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/credit_card_2023.csv')

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
client_models = [client.train_local_model(callbacks=[early_stopping]) for client in clients]

# Initialize central aggregator and aggregate models
aggregator = CentralAggregator(create_model(X_train.shape[1]))
global_model = aggregator.aggregate_models(client_models)

# Initialize blockchain
blockchain = Blockchain()

# Initialize smart contract and record model updates
smart_contract = SmartContract()
for model in client_models:
    smart_contract.verify_and_record(model.get_weights())

# Initialize security modules
anomaly_detection = AnomalyDetection()
anomaly_detection.train(X_train)
adversarial_training = AdversarialTraining(global_model)
adversarial_training.apply_adversarial_training(X_train, y_train)

# Evaluate the global model
loss, accuracy = global_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')
