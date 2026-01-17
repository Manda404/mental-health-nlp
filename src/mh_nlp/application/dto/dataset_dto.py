from dataclasses import dataclass
from typing import List

from mh_nlp.domain.entities.document import Document


@dataclass(frozen=True)
class DatasetDTO:
    """
    Data Transfer Object représentant un dataset textuel prêt à être utilisé.

    Rôle :
    - Transporter les données entre les couches Application → Infrastructure
    - Fournir une structure stable et explicite
    - Éviter toute dépendance aux frameworks (pandas, torch, numpy, etc.)

    Ce DTO est volontairement :
    - simple
    - immuable
    - sans logique métier
    """

    documents: List[Document]
    labels: List[int]
    total_records: int