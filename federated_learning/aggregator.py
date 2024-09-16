import numpy as np

class CentralAggregator:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate_models(self, client_models_data, method='fedavg'):
        if method == 'fedavg':
            num_clients = len(client_models_data)
            client_weights = [client['model'].get_weights() for client in client_models_data]

            averaged_weights = []
            for weights in zip(*client_weights):
                averaged_weights.append(np.mean(weights, axis=0))

            self.global_model.set_weights(averaged_weights)
        return self.global_model

# New FedProx Aggregator
class FedProxAggregator(CentralAggregator):
    def __init__(self, global_model, mu=0.1):
        super().__init__(global_model)
        self.mu = mu  # Proximal term

    def aggregate_models(self, client_models_data, method='fedprox'):
        if method == 'fedprox':
            num_clients = len(client_models_data)
            global_weights = self.global_model.get_weights()
            client_weights = [client['model'].get_weights() for client in client_models_data]

            # Aggregate the models with FedProx logic
            prox_weights = []
            for client_w in client_weights:
                prox_client_w = [(client_w[i] + self.mu * global_weights[i]) / (1 + self.mu)
                                 for i in range(len(client_w))]
                prox_weights.append(prox_client_w)

            averaged_weights = []
            for weights in zip(*prox_weights):
                averaged_weights.append(np.mean(weights, axis=0))

            self.global_model.set_weights(averaged_weights)
        return self.global_model
