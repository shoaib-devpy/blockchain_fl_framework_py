import tensorflow as tf
from tensorflow.keras.models import Sequential

class FederatedClient:
    def __init__(self, data, model: Sequential):
        self.X, self.y = data
        self.model = model
        self.train_data = tf.data.Dataset.from_tensor_slices((self.X, self.y)).batch(32)  # Example batch size

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
