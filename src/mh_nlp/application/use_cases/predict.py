from typing import List

from loguru import logger

from mh_nlp.domain.entities.document import Document
from mh_nlp.domain.ports.classifier import TextClassifier


class PredictUseCase:
    """
    Cas d’usage : Réaliser des prédictions sur de nouveaux textes.

    Ce service transforme des chaînes de caractères brutes en entités Document
    du domaine avant de solliciter le classifieur pour obtenir les labels.

    Attributes:
        classifier (TextClassifier): Le modèle entraîné utilisé pour l'inférence.
    """

    def __init__(self, classifier: TextClassifier):
        """
        Initialise le service de prédiction.

        Args:
            classifier (TextClassifier): Implémentation du moteur de classification.
        """
        self.classifier = classifier
        logger.debug(f"PredictUseCase initialisé avec le moteur : {type(classifier).__name__}")

    def execute(self, texts: List[str]) -> List[int]:
        """
        Transforme les textes bruts et prédit leurs classes respectives.

        Args:
            texts (List[str]): Liste de messages ou documents à classifier.

        Returns:
            List[int]: Liste des identifiants de classes prédits.

        Raises:
            ValueError: Si la liste de textes est vide.
        """
        if not texts:
            logger.warning("PredictUseCase appelé avec une liste de textes vide.")
            return []

        num_texts = len(texts)
        logger.info(f"Lancement de l'inférence sur {num_texts} document(s).")

        try:
            # 1. Conversion en objets du Domaine
            # On encapsule les strings pour garantir la cohérence avec le reste du système
            documents = [Document(text) for text in texts]
            
            # 2. Inférence via le classifieur
            predictions = self.classifier.predict(documents)
            
            logger.success(f"Inférence terminée avec succès pour {num_texts} documents.")
            return predictions

        except Exception as err:
            logger.error(f"Échec critique lors de la prédiction : {err}")
            # On propage l'erreur pour informer la couche supérieure (API/UI)
            raise RuntimeError("Le service de classification n'a pas pu traiter la demande.") from err