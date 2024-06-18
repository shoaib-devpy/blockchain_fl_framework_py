import os
from diagrams import Diagram, Cluster
from diagrams.custom import Custom
from diagrams.onprem.client import Client
from diagrams.onprem.compute import Server
from diagrams.programming.flowchart import Action

# Add Graphviz bin directory to PATH
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

with Diagram("Federated Learning and Blockchain Process", show=False):
    
    with Cluster("Data Preprocessing"):
        load_data = Custom("Load Data", "./icons/load_data.png")
        preprocess_data = Custom("Preprocess Data", "./icons/preprocess_data.png")
        load_data >> preprocess_data

    with Cluster("Federated Learning"):
        clients = [Server("Client 1"),
                   Server("Client 2"),
                   Server("Client 3"),
                   Server("Client 4"),
                   Server("Client 5")]
        train_local_models = Action("Train Local Models")
        aggregate_models = Action("Aggregate Models")
        
        preprocess_data >> train_local_models
        train_local_models >> clients
        clients >> aggregate_models

    with Cluster("Blockchain"):
        blockchain = Custom("Blockchain", "./icons/blockchain.png")
        smart_contract = Custom("Smart Contract", "./icons/smart_contract.png")
        
        aggregate_models >> blockchain
        blockchain >> smart_contract

    with Cluster("Security"):
        anomaly_detection = Custom("Anomaly Detection", "./icons/anomaly_detection.png")
        adversarial_training = Custom("Adversarial Training", "./icons/adversarial_training.png")
        
        aggregate_models >> anomaly_detection
        anomaly_detection >> adversarial_training

    with Cluster("Evaluation"):
        evaluate_model = Action("Evaluate Model")
        visualize_results = Custom("Visualize Results", "./icons/visualize_results.png")
        
        adversarial_training >> evaluate_model
        evaluate_model >> visualize_results
