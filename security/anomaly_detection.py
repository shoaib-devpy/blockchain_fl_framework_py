# security/anomaly_detection.py

from sklearn.ensemble import IsolationForest

class AnomalyDetection:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1)

    def train(self, X_train):
        self.model.fit(X_train)

    def detect_anomalies(self, X):
        return self.model.predict(X)
