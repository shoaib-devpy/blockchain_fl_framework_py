# federated_learning/aggregator.py
import numpy as np
from tensorflow.keras.models import clone_model

class CentralAggregator:
    def __init__(self, model):
        self.global_model = model

    def aggregate_models(self, client_models, method='fedavg'):
        if method == 'fedavg':
            return self.fedavg(client_models)
        elif method == 'fedsgd':
            return self.fedsgd(client_models)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def fedavg(self, client_models):
        global_weights = self.global_model.get_weights()
        num_clients = len(client_models)
        
        # Initialize the global weights to zeros
        new_global_weights = [np.zeros_like(weight) for weight in global_weights]
        
        # Sum all the client model weights
        for model in client_models:
            client_weights = model.get_weights()
            for i, weight in enumerate(client_weights):
                new_global_weights[i] += weight / num_clients
        
        # Set the global model weights to the averaged weights
        self.global_model.set_weights(new_global_weights)
        return self.global_model

    def fedsgd(self, client_models):
        # Assuming equal data size for each client for simplicity
        total_gradient = [np.zeros_like(weight) for weight in self.global_model.get_weights()]

        for model in client_models:
            gradients = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
            for i, grad in enumerate(gradients):
                total_gradient[i] += grad

        avg_gradient = [grad / len(client_models) for grad in total_gradient]
        
        self.global_model.optimizer.apply_gradients(zip(avg_gradient, self.global_model.trainable_weights))
        return self.global_model
