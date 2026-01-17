from typing import Optional

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm


class TorchTrainer:
    """
    Trainer générique PyTorch avec support de tqdm et logging via loguru.
    """
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        logger.info(f"Trainer initialisé sur {device}")

    def train(self, train_loader: DataLoader, epochs: int, val_loader: Optional[DataLoader] = None):
        self.model.to(self.device)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                # Unpacking du tuple (inputs, labels)
                inputs_dict, labels = batch
                
                # Envoi sur le device
                inputs = {k: v.to(self.device) for k, v in inputs_dict.items()}
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                
                # Support pour les modèles HuggingFace (retournent un objet) et Custom (retournent des logits)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = self.loss_fn(logits, labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = total_loss / len(train_loader)
            if val_loader:
                val_loss = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

    def evaluate(self, data_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                inputs_dict, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs_dict.items()}
                labels = labels.to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                total_loss += self.loss_fn(logits, labels).item()
        return total_loss / len(data_loader)