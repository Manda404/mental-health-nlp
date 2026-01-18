import torch
import yaml
import os

from mh_nlp.infrastructure.nlp.hf_tokenizer import HuggingFaceTokenizer
from mh_nlp.infrastructure.nlp.keras_tokenizer import KerasTextTokenizer

# Importations corrigées selon nos fichiers précédents
from mh_nlp.infrastructure.models.distilbert_cnn_classifier import HybridClassifier
from mh_nlp.infrastructure.models.roberta_classifier import RobertaClassifier
from mh_nlp.infrastructure.models.cnn_classifier import KerasCNNClassifier
from mh_nlp.infrastructure.models.distilbert_classifier import DistilBertClassifier

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_tokenizer(config: dict):
    model_type = config["model"]["type"] # Utilise 'type' pour plus de clarté

    if model_type in ["distilbert", "distilbert_cnn"]:
        return HuggingFaceTokenizer(
            model_name="distilbert-base-uncased",
            max_length=config["tokenizer"]["max_length"]
        )
    if model_type == "roberta":
        return HuggingFaceTokenizer(
            model_name="roberta-base",
            max_length=config["tokenizer"]["max_length"]
        )
    if model_type == "cnn_keras":
        return KerasTextTokenizer(
            vocab_size=config["tokenizer"]["vocab_size"],
            max_length=config["tokenizer"]["max_length"]
        )
    raise ValueError(f"Unknown model type: {model_type}")

def build_classifier(config: dict, tokenizer):
    """La fonction qui manquait à ton script"""
    model_type = config["model"]["type"]
    num_labels = config["model"]["num_labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "distilbert":
        return DistilBertClassifier("distilbert-base-uncased", num_labels, tokenizer, device)
    
    if model_type == "distilbert_cnn":
        return HybridClassifier("distilbert-base-uncased", num_labels, tokenizer, device)
    
    if model_type == "roberta":
        return RobertaClassifier("roberta-base", num_labels, tokenizer, device)
    
    if model_type == "cnn_keras":
        return KerasCNNClassifier(
            vocab_size=config["tokenizer"]["vocab_size"],
            embedding_dim=128,
            max_length=config["tokenizer"]["max_length"],
            num_labels=num_labels,
            tokenizer=tokenizer
        )
    raise ValueError(f"Classifier non supporté: {model_type}")

def build_system(config_path: str):
    config = load_config(config_path)
    tokenizer = build_tokenizer(config)
    classifier = build_classifier(config, tokenizer)

    return classifier, tokenizer, config