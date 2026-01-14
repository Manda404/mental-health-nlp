from abc import ABC, abstractmethod
from typing import List

from mh_nlp.domain.entities.document import Document


class TextClassifier(ABC):
    """
    Interface métier pour tout classifieur de texte.

    Pourquoi :
    - Découpler le domaine des frameworks ML
    - Permettre de changer de modèle sans casser le système
    """

    @abstractmethod
    def train(self, train_data, validation_data) -> None:
        """
        Entraîne le modèle.
        """
        pass

    @abstractmethod
    def predict(self, documents: List[Document]) -> List[int]:
        """
        Prédit la classe la plus probable.
        """
        pass

    @abstractmethod
    def predict_proba(self, documents: List[Document]):
        """
        Retourne les probabilités par classe.
        """
        pass
