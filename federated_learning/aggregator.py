# federated_learning/aggregator.py
from tensorflow.keras.models import Model

class CentralAggregator:
    def __init__(self, model: Model):
        self.global_model = model

    def aggregate_models(self, client_models):
        new_weights = [model.get_weights() for model in client_models]
        avg_weights = [
            sum(layer_weights) / len(layer_weights) for layer_weights in zip(*new_weights)
        ]
        self.global_model.set_weights(avg_weights)
        return self.global_model
