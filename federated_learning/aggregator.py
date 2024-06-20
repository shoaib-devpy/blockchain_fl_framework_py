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
