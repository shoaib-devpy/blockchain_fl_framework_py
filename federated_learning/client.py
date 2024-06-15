# federated_learning/client.py
class FederatedClient:
    def __init__(self, data, model):
        self.data = data
        self.model = model

    def train_local_model(self, epochs=1, callbacks=None, validation_data=None):
        X_train, y_train = self.data
        self.model.fit(X_train, y_train, epochs=epochs, callbacks=callbacks, validation_data=validation_data)
        return self.model
