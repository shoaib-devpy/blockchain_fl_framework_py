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
        clean_data = Custom("Clean Data", "./icons/clean_data.png")
        normalize_data = Custom("Normalize Data", "./icons/normalize_data.png")
        feature_engineering = Custom("Feature Engineering", "./icons/feature_engineering.png")
        preprocess_data = Custom("Preprocess Data", "./icons/preprocess_data.png")
        partition_data = Custom("Partition Data", "./icons/partition_data.png")
        
        load_data >> clean_data >> normalize_data >> feature_engineering >> preprocess_data >> partition_data

    with Cluster("Federated Learning"):
        client1 = Server("Client 1")
        client2 = Server("Client 2")
        client3 = Server("Client 3")
        client4 = Server("Client 4")
        client5 = Server("Client 5")
        train_local_model_client1 = Action("Train Local Model (Client 1)")
        train_local_model_client2 = Action("Train Local Model (Client 2)")
        train_local_model_client3 = Action("Train Local Model (Client 3)")
        train_local_model_client4 = Action("Train Local Model (Client 4)")
        train_local_model_client5 = Action("Train Local Model (Client 5)")
        aggregate_models = Action("Aggregate Models")
        
        partition_data >> train_local_model_client1
        partition_data >> train_local_model_client2
        partition_data >> train_local_model_client3
        partition_data >> train_local_model_client4
        partition_data >> train_local_model_client5

        train_local_model_client1 >> client1
        train_local_model_client2 >> client2
        train_local_model_client3 >> client3
        train_local_model_client4 >> client4
        train_local_model_client5 >> client5

        client1 >> aggregate_models
        client2 >> aggregate_models
        client3 >> aggregate_models
        client4 >> aggregate_models
        client5 >> aggregate_models

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
        calculate_metrics = Action("Calculate Metrics")
        tp = Custom("True Positives (TP)", "./icons/tp.png")
        fp = Custom("False Positives (FP)", "./icons/fp.png")
        fn = Custom("False Negatives (FN)", "./icons/fn.png")
        tn = Custom("True Negatives (TN)", "./icons/tn.png")
        recall = Custom("Recall", "./icons/recall.png")
        precision = Custom("Precision", "./icons/precision.png")
        f1_score = Custom("F1-Score", "./icons/f1_score.png")
        visualize_results = Custom("Visualize Results", "./icons/visualize_results.png")
        
        adversarial_training >> evaluate_model >> calculate_metrics
        calculate_metrics >> tp
        calculate_metrics >> fp
        calculate_metrics >> fn
        calculate_metrics >> tn
        calculate_metrics >> recall
        calculate_metrics >> precision
        calculate_metrics >> f1_score
        f1_score >> visualize_results

    with Cluster("Visualization"):
        plot_training_history = Custom("Plot Training History", "./icons/plot_training_history.png")
        plot_global_model_performance = Custom("Plot Global Model Performance", "./icons/plot_global_model_performance.png")
        plot_confusion_matrix = Custom("Plot Confusion Matrix", "./icons/plot_confusion_matrix.png")
        plot_normalized_confusion_matrix = Custom("Plot Normalized Confusion Matrix", "./icons/plot_normalized_confusion_matrix.png")
        plot_feature_distribution = Custom("Plot Feature Distribution", "./icons/plot_feature_distribution.png")
        plot_anomaly_detection = Custom("Plot Anomaly Detection", "./icons/plot_anomaly_detection.png")
        plot_client_data_distribution = Custom("Plot Client Data Distribution", "./icons/plot_client_data_distribution.png")
        plot_client_model_performance = Custom("Plot Client Model Performance", "./icons/plot_client_model_performance.png")
        
        visualize_results >> plot_training_history
        visualize_results >> plot_global_model_performance
        visualize_results >> plot_confusion_matrix
        visualize_results >> plot_normalized_confusion_matrix
        visualize_results >> plot_feature_distribution
        visualize_results >> plot_anomaly_detection
        visualize_results >> plot_client_data_distribution
        visualize_results >> plot_client_model_performance
