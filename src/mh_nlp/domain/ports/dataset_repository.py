# domain/ports/dataset_repository.py
from abc import ABC, abstractmethod
from typing import List, Tuple

from mh_nlp.domain.entities.document import Document
from mh_nlp.domain.entities.label import Label


class DatasetRepository(ABC):
    """
    Port (Interface) définissant comment les données externes entrent dans le domaine.

    Règle d'or : Le repository est responsable de la conversion des données
    externes (CSV, SQL) en Entités du domaine avant de les retourner.
    """

    @abstractmethod
    def load(self) -> List[Tuple[Document, Label]]:
        """
        Charge la source de données et renvoie une collection d'objets typés.

        Returns:
            List[Tuple[Document, Label]]: Une liste de paires (Document, Label).
        """
        pass
