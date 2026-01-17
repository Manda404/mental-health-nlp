import numpy as np
from typing import List
from mh_nlp.domain.ports.classifier import TextClassifier
from mh_nlp.domain.entities.document import Document
import tensorflow as tf
from tensorflow.keras.models import Sequential
from mh_nlp.application.dto.dataset_dto import DatasetDTO
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

class KerasCNNClassifier(TextClassifier):
    """Implémentation CNN via TensorFlow/Keras."""
    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int, num_labels: int, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels
        
        self.model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_labels, activation='softmax' if num_labels > 2 else 'sigmoid')
        ])
        
        loss = 'sparse_categorical_crossentropy' if num_labels > 2 else 'binary_crossentropy'
        self.model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    def train(self, train_data: DatasetDTO, validation_data: DatasetDTO) -> None:
        self.tokenizer.fit(train_data.documents)
        x_train = self.tokenizer.tokenize(train_data.documents)
        y_train = np.array(train_data.labels)
        
        x_val = self.tokenizer.tokenize(validation_data.documents)
        y_val = np.array(validation_data.labels)

        # Utilise la barre de progression native de Keras (verbose=1)
        self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=10,
            batch_size=32,
            verbose=1
        )

    def predict(self, documents: List[Document]) -> List[int]:
        probs = self.predict_proba(documents)
        return np.argmax(probs, axis=1).tolist()

    def predict_proba(self, documents: List[Document]):
        sequences = self.tokenizer.tokenize(documents)
        return self.model.predict(sequences, verbose=0)