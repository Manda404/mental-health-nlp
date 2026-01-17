from typing import List

from loguru import logger
from transformers import AutoTokenizer

from mh_nlp.domain.entities.document import Document
from mh_nlp.domain.ports.tokenizer import Tokenizer


class HuggingFaceTokenizer(Tokenizer):
    """
    Implémentation du port Tokenizer utilisant la bibliothèque HuggingFace Transformers.

    Cette classe permet de charger n'importe quel tokenizer pré-entraîné 
    (DistilBERT, RoBERTa, BERT, etc.) et de transformer des documents du domaine 
    en tenseurs exploitables par les modèles de Deep Learning.
    """

    def __init__(self, model_name: str, max_length: int = 512):
        """
        Initialise le tokenizer HuggingFace.

        Args:
            model_name (str): Nom du modèle sur le Hub HuggingFace (ex: 'distilbert-base-uncased').
            max_length (int): Longueur maximale des séquences (padding/truncation).
        
        Raises:
            RuntimeError: Si le modèle ne peut pas être chargé depuis le Hub ou le cache.
        """
        self.model_name = model_name
        self.max_length = max_length
        
        try:
            logger.info(f"Chargement du tokenizer HF : {model_name} (max_length={max_length})")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.success(f"Tokenizer {model_name} chargé avec succès.")
        except Exception as e:
            logger.critical(f"Impossible de charger le tokenizer '{model_name}': {e}")
            raise RuntimeError("Échec d'initialisation du tokenizer HuggingFace") from e

    def tokenize(self, documents: List[Document]):
        """
        Transforme une liste de documents en dictionnaire de tenseurs PyTorch.

        Applique systématiquement le padding et la troncature selon la configuration
        définie à l'initialisation.

        Args:
            documents (List[Document]): Liste d'entités Documents à transformer.

        Returns:
            BatchEncoding: Objet contenant les 'input_ids', 'attention_mask', etc.
        
        Raises:
            ValueError: Si la liste de documents est vide.
        """
        if not documents:
            logger.warning("Tokenize appelé avec une liste de documents vide.")
            raise ValueError("La liste de documents à tokeniser ne peut pas être vide.")

        texts = [doc.text for doc in documents]
        
        logger.debug(f"Tokenisation en cours pour {len(texts)} documents (Modèle: {self.model_name})")
        
        try:
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Log de la forme du tenseur résultant pour le monitoring
            shape = encoded_input["input_ids"].shape
            logger.debug(f"Tokenisation terminée. Forme du tenseur : {list(shape)}")
            
            return encoded_input

        except Exception as e:
            logger.error(f"Erreur lors de la tokenisation du lot : {e}")
            raise RuntimeError("Échec du processus de tokenisation HuggingFace") from e