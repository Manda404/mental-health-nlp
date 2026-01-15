from abc import ABC, abstractmethod
from typing import List

class FastCleaningEngine(ABC):
    """
    Interface abstraite (Port) définissant le contrat pour les moteurs de traitement.

    Pourquoi ce port :
    - Découple le domaine des bibliothèques lourdes (spaCy/NLTK).
    - Permet d'interchanger l'implémentation technique sans modifier la logique métier.
    - Définit une signature stricte pour assurer la cohérence des données sortantes.
    """

    @abstractmethod
    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Transforme une liste de chaînes en une liste de listes de tokens.
        
        Pourquoi cette méthode :
        - Performance : Permet d'utiliser les optimisations multi-flux de spaCy (nlp.pipe).
        - Efficacité : Réduit l'overhead des appels répétés dans les couches supérieures.
        
        Args:
            texts (List[str]): Liste de textes pré-nettoyés.
            
        Returns:
            List[List[str]]: Liste contenant les tokens pour chaque document.
        """
        pass