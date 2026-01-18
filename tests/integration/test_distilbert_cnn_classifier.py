import torch
from mh_nlp.infrastructure.models.distilbert_cnn_classifier import (
    DistilBertCNNClassifier
)
from mh_nlp.infrastructure.nlp.hf_tokenizer import HuggingFaceTokenizer
from mh_nlp.domain.entities.document import Document

def test_distilbert_cnn_predict_runs():
    tokenizer = HuggingFaceTokenizer(
        model_name="distilbert-base-uncased",
        max_length=100
    )

    classifier = DistilBertCNNClassifier(
        num_labels=3,
        tokenizer=tokenizer,
        device=torch.device("cpu")
    )

    docs = [Document("I feel anxious today")]

    preds = classifier.predict(docs)

    assert isinstance(preds, list)
    assert len(preds) == 1 