import torch
import torch.nn.functional as F
from typing import List
from transformers import RobertaForSequenceClassification
from torch.utils.data import DataLoader
from mh_nlp.domain.ports.classifier import TextClassifier
from mh_nlp.domain.entities.document import Document
from mh_nlp.infrastructure.models.distilbert_classifier import MentalHealthDataset
from mh_nlp.infrastructure.training.torch_trainer import TorchTrainer
from mh_nlp.application.dto.dataset_dto import DatasetDTO

class RobertaClassifier(TextClassifier):
    """Adaptateur pour RoBERTa (plus robuste sur le contexte long)."""
    def __init__(self, model_name: str, num_labels: int, tokenizer, device):
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(device)
        self.tokenizer = tokenizer
        self.device = device

    def train(self, train_data: DatasetDTO, validation_data: DatasetDTO) -> None:
        train_enc = self.tokenizer.tokenize(train_data.documents)
        dataset = MentalHealthDataset(train_enc, train_data.labels)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        trainer = TorchTrainer(self.model, optimizer, loss_fn, self.device)
        trainer.train(loader, epochs=3)

    def predict(self, documents: List[Document]) -> List[int]:
        logits = self._get_logits(documents)
        return torch.argmax(logits, dim=1).tolist()

    def predict_proba(self, documents: List[Document]):
        logits = self._get_logits(documents)
        return F.softmax(logits, dim=1).tolist()

    def _get_logits(self, documents: List[Document]):
        self.model.eval()
        enc = self.tokenizer.tokenize(documents)
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in enc.items()}
            return self.model(**inputs).logits