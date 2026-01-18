# tests/integration/test_cnn_keras_classifier.py

from mh_nlp.infrastructure.models.cnn_keras_classifier import CnnKerasClassifier
from mh_nlp.infrastructure.nlp.keras_tokenizer import KerasTextTokenizer
from mh_nlp.domain.entities.document import Document


def test_cnn_keras_predict_runs():
    tokenizer = KerasTextTokenizer(
        vocab_size=1000,
        max_length=50
    )

    classifier = CnnKerasClassifier(
        tokenizer=tokenizer,
        vocab_size=1000,
        max_length=50,
        num_labels=3
    )

    documents = [
        Document("I feel stressed and tired"),
        Document("Today is a good day")
    ]

    predictions = classifier.predict(documents)

    assert isinstance(predictions, list)
    assert len(predictions) == 2