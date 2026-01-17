import re

from loguru import logger

from mh_nlp.domain.entities.document import Document


class TextCleaner:
    """
    Service de domaine orchestrant le nettoyage de surface des documents.
    
    Responsabilités :
    1. Nettoyage technique (HTML, URLs, Mentions).
    2. Normalisation de la casse (Lowercase).
    3. Préservation du sens (conservation des négations et de l'intensité).
    4. Filtrage des débris de caractères isolés.
    """

    # Patterns pré-compilés pour une performance maximale
    _RE_HTML = re.compile(r'<.*?>')
    _RE_URLS = re.compile(r"https?://\S+|www\S+")
    _RE_MENTIONS = re.compile(r"@\w+|#\w+")
    # On garde l'apostrophe pour ne pas casser "can't", "don't", "i'm"
    _RE_SPECIAL_CHARS = re.compile(r"[^a-zA-Z\s']")
    _RE_SPACES = re.compile(r"\s+")

    def clean(self, document: Document) -> str:
        """
        Nettoie le texte en profondeur sans altérer les mots porteurs de sens.
        
        Args:
            document (Document): L'entité contenant le texte brut.
            
        Returns:
            str: Texte nettoyé et normalisé pour le Deep Learning.
        """
        if not document or document.is_empty():
            logger.warning("Document vide ou nul reçu → nettoyage ignoré.")
            return ""

        # 1. Suppression du HTML (évite les résidus type 'p' ou 'br')
        text = self._RE_HTML.sub(' ', document.text)

        # 2. Normalisation de la casse
        text = text.lower()

        # 3. Suppression du bruit Web (URLs, @Mentions, #Hashtags)
        text = self._RE_URLS.sub('', text)
        text = self._RE_MENTIONS.sub('', text)

        # 4. Suppression des caractères spéciaux (sauf lettres et apostrophes)
        text = self._RE_SPECIAL_CHARS.sub(' ', text)

        # 5. Filtrage fin et gestion des mots isolés
        words = text.split()
        # On garde les mots > 1 lettre OU le pronom 'i' (crucial en anglais)
        # On ne filtre PAS les stop-words ici pour garder 'not', 'very', etc.
        # cleaned_words = [w for w in words if len(w) > 1 or w == 'i']
        cleaned_words = [w for w in words if len(w) > 1]
        
        text = " ".join(cleaned_words)

        # 6. Nettoyage final des espaces blancs
        return self._RE_SPACES.sub(" ", text).strip()