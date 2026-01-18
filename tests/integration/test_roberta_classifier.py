# tests/integration/test_roberta_classifier.py

import torch

from mh_nlp.infrastructure.models.roberta_classifier import RobertaClassifier
from mh_nlp.infrastructure.nlp.hf_tokenizer import HuggingFaceTokenizer
from mh_nlp.domain.entities.document import Document


def test_roberta_predict_runs():
    tokenizer = HuggingFaceTokenizer(
        model_name="roberta-base",
        max_length=128
    )

    classifier = RobertaClassifier(
        model_name="roberta-base",
        num_labels=3,
        tokenizer=tokenizer,
        device=torch.device("cpu")
    )

    documents = [
        Document("I feel extremely stressed and anxious")
    ]

    predictions = classifier.predict(documents)

    assert isinstance(predictions, list)
    assert len(predictions) == 1