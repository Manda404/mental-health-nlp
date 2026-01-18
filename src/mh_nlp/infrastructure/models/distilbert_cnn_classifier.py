import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import DistilBertModel
from torch.utils.data import DataLoader
from mh_nlp.domain.ports.classifier import TextClassifier
from mh_nlp.domain.entities.document import Document
from mh_nlp.infrastructure.models.distilbert_classifier import MentalHealthDataset
from mh_nlp.infrastructure.training.torch_trainer import TorchTrainer
from sklearn.utils.class_weight import compute_class_weight


class HybridBertCNN(nn.Module):
    """Modèle hybride : BERT pour le sens global, CNN pour les n-grammes émotionnels."""
    def __init__(self, model_name, num_labels, n_filters=100, filter_sizes=[3, 4, 5]):
        super(HybridBertCNN, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, 768)) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, num_labels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # On extrait la dernière couche cachée : [batch, seq_len, 768]
        embedded = outputs.last_hidden_state.unsqueeze(1) 
        # Convolution + ReLU + MaxPool
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # Concatenation des différentes tailles de filtres
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class HybridClassifier(TextClassifier):
    def __init__(self, model_name: str, num_labels: int, tokenizer, device):
        self.model = HybridBertCNN(model_name, num_labels).to(device)
        self.tokenizer = tokenizer
        self.device = device

    def train(self, train_data, validation_data) -> None:
        train_enc = self.tokenizer.tokenize(train_data.documents)
        train_dataset = MentalHealthDataset(train_enc, train_data.labels)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        val_loader = None
        if validation_data:
            val_enc = self.tokenizer.tokenize(validation_data.documents)
            val_dataset = MentalHealthDataset(val_enc, validation_data.labels)
            val_loader = DataLoader(val_dataset, batch_size=16)

        weights = compute_class_weight(
            "balanced",
            classes=np.unique(train_data.labels),
            y=train_data.labels
        )

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights).to(self.device)
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        #loss_fn = torch.nn.CrossEntropyLoss()
        trainer = TorchTrainer(self.model, optimizer, loss_fn, self.device)
        trainer.train(train_loader=train_loader, epochs=3, val_loader=val_loader)

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
            return self.model(enc['input_ids'].to(self.device), 
                             enc['attention_mask'].to(self.device))