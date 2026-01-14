from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """
    Représente un document textuel du point de vue métier.

    Pourquoi cette classe :
    - Centraliser la notion de 'texte' dans le domaine
    - Éviter de manipuler des strings brutes partout
    - Faciliter la validation, le nettoyage et les tests
    """

    text: str

    def is_empty(self) -> bool:
        """
        Vérifie si le document est vide ou ne contient que des espaces.

        Pourquoi :
        - Éviter les documents invalides dans le pipeline
        - Remplacer les checks dispersés dans le notebook
        """
        return not self.text.strip()
