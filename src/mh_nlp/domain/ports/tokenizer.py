from abc import ABC, abstractmethod
from typing import Any, List

from loguru import logger

from mh_nlp.domain.entities.document import Document


class Tokenizer(ABC):
    """
    Port (Interface) métier pour la tokenisation de texte.

    Cette abstraction permet de découpler la logique métier des bibliothèques 
    externes (HuggingFace, Keras, spaCy). Elle garantit que n'importe quel 
    moteur de tokenisation peut être injecté dans les cas d'usage tant qu'il 
    respecte ce contrat.

    Pourquoi cette interface :
    - Interchangeabilité : Passer de BERT à un CNN Keras sans modifier les Use Cases.
    - Testabilité : Créer facilement des Mocks/Stubs pour les tests unitaires.
    - Standardisation : Forcer l'utilisation de la liste d'entités 'Document'.
    """

    @abstractmethod
    def tokenize(self, documents: List[Document]) -> Any:
        """
        Transforme une collection de documents en une structure exploitable par un modèle.

        Args:
            documents (List[Document]): Liste des entités documents du domaine.

        Returns:
            Any: Le format de sortie dépend de l'implémentation (Tenseurs PT, 
                 Matrices Numpy, Listes de tokens, etc.).
        
        Note aux implémenteurs:
            Pensez à loguer les métriques de sortie (dimensions des tenseurs, 
            nombre de tokens) en niveau 'DEBUG'.
        """
        # On peut ajouter un log ici même si c'est une méthode abstraite 
        # pour tracer quel type de tokenizer est sollicité.
        logger.trace(f"Appel du tokenizer : {self.__class__.__name__}")
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} Interface>"