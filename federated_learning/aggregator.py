import numpy as np
import tensorflow as tf
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
        global_weights = client_models[0]['model'].get_weights()
        new_global_weights = [np.zeros_like(weight) for weight in global_weights]

        for client_model in client_models:
            client_weights = client_model['model'].get_weights()
            for i, weight in enumerate(client_weights):
                new_global_weights[i] += weight / len(client_models)

        self.global_model.set_weights(new_global_weights)
        return self.global_model

    def fedsgd(self, client_models):
        total_gradient = [np.zeros_like(weight) for weight in self.global_model.trainable_weights]

        # Assuming all client models are compiled with the same input shape and loss
        for client_model in client_models:
            model = client_model['model']
            train_data = client_model['train_data']
            for X_batch, y_batch in train_data:
                with tf.GradientTape() as tape:
                    predictions = model(X_batch, training=True)
                    loss = model.compute_loss(X_batch, y_batch, predictions)
                gradients = tape.gradient(loss, self.global_model.trainable_weights)
                gradients = [grad.numpy() if grad is not None else np.zeros_like(weight) for grad, weight in zip(gradients, self.global_model.trainable_weights)]  # Convert to numpy arrays
                for i, grad in enumerate(gradients):
                    total_gradient[i] += grad

        avg_gradient = [grad / len(client_models) for grad in total_gradient]

        optimizer = self.global_model.optimizer
        optimizer.apply_gradients(zip(avg_gradient, self.global_model.trainable_weights))
        return self.global_model
