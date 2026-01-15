from abc import ABC, abstractmethod
from typing import List

class CleaningEngine(ABC):
    """
    Interface abstraite (Port) définissant le contrat pour les moteurs de traitement.

    Pourquoi ce port :
    - Découple le domaine des bibliothèques lourdes (spaCy/NLTK).
    - Permet d'interchanger l'implémentation technique sans modifier la logique métier.
    - Définit une signature stricte pour assurer la cohérence des données sortantes.
    """

    @abstractmethod
    def process_text(self, text: str) -> List[str]:
        """
        Transforme une seule chaîne en liste de tokens (lemmes/racines).

        Args:
            text (str): Le texte pré-nettoyé (souvent par Regex).

        Returns:
            List[str]: Une liste de mots clés nettoyés.
        """
        pass