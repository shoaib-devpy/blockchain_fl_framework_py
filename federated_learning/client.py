# federated_learning/client.py

from tensorflow.keras.models import Sequential

class FederatedClient:
    def __init__(self, data, model: Sequential):
        self.X, self.y = data
        self.model = model

    def train_local_model(self, epochs, callbacks=None, validation_data=None):
        history = self.model.fit(
            self.X, 
            self.y, 
            epochs=epochs, 
            validation_data=validation_data, 
            callbacks=callbacks, 
            verbose=1
        )
        return history

# federated_learning/aggregator.py

import numpy as np
from tensorflow.keras.models import Sequential

class CentralAggregator:
    def __init__(self, global_model: Sequential):
        self.global_model = global_model

    def aggregate_models(self, client_models, method='fedavg'):
        if method == 'fedavg':
            return self.fedavg(client_models)
        elif method == 'fedsgd':
            return self.fedsgd(client_models)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def fedavg(self, client_models):
        global_weights = client_models[0].get_weights()
        new_global_weights = [np.zeros_like(weight) for weight in global_weights]

        for client_model in client_models:
            client_weights = client_model.get_weights()
            for i, weight in enumerate(client_weights):
                new_global_weights[i] += weight / len(client_models)

        self.global_model.set_weights(new_global_weights)
        return self.global_model

    def fedsgd(self, client_models):
        global_weights = client_models[0].get_weights()
        new_global_weights = [np.zeros_like(weight) for weight in global_weights]

        for client_model in client_models:
            client_weights = client_model.get_weights()
            for i, weight in enumerate(client_weights):
                new_global_weights[i] += weight / len(client_models)

        self.global_model.set_weights(new_global_weights)
        return self.global_model
