import re

from loguru import logger

from mh_nlp.domain.entities.document import Document
from mh_nlp.domain.ports.cleaningengine import CleaningEngine


class TextCleaner:
    """
    Service Domaine orchestrant la politique de nettoyage des textes.

    Responsabilités :
    1. Validation métier (Document) : Vérification de l'intégrité des données d'entrée.
    2. Normalisation métier (Regex) : Suppression du bruit (URLs, handles, caractères spéciaux)
       via des patterns pré-compilés pour maximiser la performance.
    3. Délégation linguistique (CleaningEngine) : Traitement avancé (stop-words, lemmatisation).

    Optimisation :
    - Utilise des expressions régulières pré-compilées au niveau de la classe pour éviter
      le overhead de compilation à chaque appel de la méthode clean().
    - Réduit les copies de chaînes en mémoire en fusionnant les patterns de suppression.
    """

    # Patterns pré-compilés pour la performance
    # Supprime URLs, mentions (@), hashtags (#) et tout ce qui n'est pas a-z ou espace
    _RE_NOISE = re.compile(r"http\S+|www\S+|@\w+|#\w+|[^a-z\s]")
    # Normalise les espaces multiples
    _RE_SPACES = re.compile(r"\s+")

    def __init__(self, engine: CleaningEngine):
        self.engine = engine
        logger.debug(f"TextCleaner initialisé avec {engine.__class__.__name__}")

    def clean(self, document: Document) -> str:
        """
        Nettoie un Document et retourne un texte normalisé prêt pour le ML.

        Args:
            document (Document): L'objet document contenant le texte brut.

        Returns:
            str: Le texte nettoyé, tokenisé par le moteur et reformaté en chaîne.
        """
        if document.is_empty():
            logger.warning("Document vide reçu → nettoyage ignoré")
            return ""

        # ----------------------------
        # Phase 1 — Normalisation métier (Optimisée)
        # ----------------------------
        # Passage en minuscule unique
        text = document.text.lower()

        # Suppression du bruit en une seule passe sur les patterns combinés
        text = self._RE_NOISE.sub("", text)

        # Nettoyage final des espaces blancs
        text = self._RE_SPACES.sub(" ", text).strip()

        # ----------------------------
        # Phase 2 — Traitement linguistique
        # ----------------------------
        try:
            tokens = self.engine.process_text(text)
            return " ".join(tokens).strip()

        except Exception as exc:
            logger.error(
                f"Erreur moteur linguistique : {exc} — fallback sur texte normalisé"
            )
            return text
