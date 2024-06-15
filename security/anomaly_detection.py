# security/anomaly_detection.py
from sklearn.ensemble import IsolationForest

class AnomalyDetection:
    def __init__(self):
        self.detector = IsolationForest(contamination=0.1)

    def train(self, X):
        self.detector.fit(X)

    def detect(self, X):
        return self.detector.predict(X)
