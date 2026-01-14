# domain/ports/classifier.py
from abc import ABC, abstractmethod


class TextClassifier(ABC):
    @abstractmethod
    def train(self, train_data, val_data):
        pass

    @abstractmethod
    def predict(self, texts):
        pass

    @abstractmethod
    def predict_proba(self, texts):
        pass
