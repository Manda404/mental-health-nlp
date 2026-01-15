import re
from typing import List
from loguru import logger
from mh_nlp.domain.entities.document import Document
from mh_nlp.domain.ports.fast_cleaning_engine import FastCleaningEngine

class FastTextCleaner:
    """
    Service Domaine optimisé pour le traitement massif (Batch) de documents.

    Ce service est conçu pour transformer une collection d'entités Document 
    en une liste de chaînes de caractères nettoyées, prêtes pour l'entraînement.
    """

    # Patterns pré-compilés pour la performance globale
    _RE_NOISE = re.compile(r"http\S+|www\S+|@\w+|#\w+|[^a-z\s]")
    _RE_SPACES = re.compile(r"\s+")

    def __init__(self, engine: FastCleaningEngine):
        """
        Initialise le service avec un moteur supportant le traitement par lot.
        """
        self.engine = engine
        logger.debug(f"FastTextCleaner (Batch Only) prêt avec {engine.__class__.__name__}")

    def _apply_regex(self, text: str) -> str:
        """Nettoyage Regex atomique (Minuscules -> Bruit -> Espaces)."""
        text = text.lower()
        # Remplacer par un espace " " au lieu de "" évite de coller les mots
        text = self._RE_NOISE.sub(" ", text)
        return self._RE_SPACES.sub(" ", text).strip()

    def clean_batch(self, documents: List[Document]) -> List[str]:
        """
        Nettoie un lot de documents en une seule passe orchestrée.
        
        Cette méthode garantit la conservation de l'ordre original et gère 
        les documents vides sans rompre l'alignement des données.

        Args:
            documents (List[Document]): Liste d'entités issues du Repository.

        Returns:
            List[str]: Liste de textes nettoyés, de même longueur que l'entrée.
        """
        if not documents:
            return []

        # 1. Pré-nettoyage Regex (Opération CPU Python rapide)
        # On conserve les emplacements vides pour ne pas décaler les labels
        pre_cleaned = [
            self._apply_regex(doc.text) if not doc.is_empty() else "" 
            for doc in documents
        ]
        
        try:
            # 2. Traitement NLP Batch (Délégation à l'engine optimisé)
            # C'est ici que nlp.pipe de spaCy ou le batch de NLTK intervient
            token_batches = self.engine.process_batch(pre_cleaned)
            
            # 3. Reconstruction des textes (Flattening du résultat de l'engine)
            return [" ".join(tokens).strip() for tokens in token_batches]

        except Exception as exc:
            # En cas d'erreur critique de l'engine, on renvoie le travail déjà fait par Regex
            logger.error(f"Échec du traitement Batch NLP : {exc}")
            return pre_cleaned