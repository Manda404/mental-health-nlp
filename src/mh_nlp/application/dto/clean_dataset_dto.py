from dataclasses import dataclass
from typing import List

from mh_nlp.domain.entities.document import Document


@dataclass(frozen=True)
class CleanedDatasetDTO:
    """
    Objet de transfert de données pour un dataset nettoyé.

    Pourquoi un DTO :
    - Évite de renvoyer des entités brutes si elles contiennent des données inutiles.
    - Stabilise le contrat entre l'Application et l'Interface (Notebook/API).

    NB: DTO représentant un dataset prêt à être utilisé par un modèle.

    Pourquoi :
    - Découpler pandas / numpy du reste du système
    - Rendre les use cases testables
    """

    documents: List[Document]
    labels: List[int]
    total_processed: int


