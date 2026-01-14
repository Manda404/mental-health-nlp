from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    """
    Représente une classe cible du point de vue métier.

    Pourquoi :
    - Séparer la notion métier ('Anxiety') de son encodage numérique
    - Éviter les dépendances à sklearn dans le domain
    """

    name: str
    index: int
