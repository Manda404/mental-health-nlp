from typing import List

import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification

from mh_nlp.domain.entities.document import Document
from mh_nlp.domain.ports.classifier import TextClassifier
from mh_nlp.infrastructure.training.torch_trainer import TorchTrainer
from mh_nlp.application.dto.dataset_dto import DatasetDTO

class MentalHealthDataset(Dataset):
    """Dataset interne pour mapper les tenseurs et les labels."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # On retourne un tuple (inputs, labels) comme attendu par ton TorchTrainer
        inputs = {k: v[idx] for k, v in self.encodings.items()}
        label = torch.tensor(self.labels[idx])
        return inputs, label

    def __len__(self):
        return len(self.labels)

class DistilBertClassifier(TextClassifier):
    """
    Implémentation DistilBERT respectant le port TextClassifier.
    Utilise le TorchTrainer pour la boucle d'entraînement.
    """
    def __init__(self, model_name: str, num_labels: int, tokenizer, device):
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(device)
        self.tokenizer = tokenizer
        self.device = device

    def train(self, train_data: DatasetDTO, validation_data: DatasetDTO) -> None:
        """Prépare les DataLoaders et délègue au TorchTrainer."""
        train_enc = self.tokenizer.tokenize(train_data.documents)
        train_dataset = MentalHealthDataset(train_enc, train_data.labels)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        val_loader = None
        if validation_data:
            val_enc = self.tokenizer.tokenize(validation_data.documents)
            val_dataset = MentalHealthDataset(val_enc, validation_data.labels)
            val_loader = DataLoader(val_dataset, batch_size=16)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        trainer = TorchTrainer(self.model, optimizer, loss_fn, self.device)
        trainer.train(train_loader, epochs=3, val_loader=val_loader)

    def predict(self, documents: List[Document]) -> List[int]:
        """Prédit les IDs de classes pour une liste de Documents."""
        logits = self._get_logits(documents)
        return torch.argmax(logits, dim=1).tolist()

    def predict_proba(self, documents: List[Document]) -> List[List[float]]:
        """Retourne les probabilités après Softmax."""
        logits = self._get_logits(documents)
        return softmax(logits, dim=1).tolist()

    def _get_logits(self, documents: List[Document]):
        self.model.eval()
        enc = self.tokenizer.tokenize(documents)
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**inputs)
            # DistilBertForSequenceClassification retourne un objet avec un attribut .logits
            return outputs.logits if hasattr(outputs, 'logits') else outputs