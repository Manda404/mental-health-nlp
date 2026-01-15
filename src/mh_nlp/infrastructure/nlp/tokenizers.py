# mh_nlp/infrastructure/nlp/tokenizers.py
from typing import List

from loguru import logger

from mh_nlp.domain.ports.cleaning_engine import CleaningEngine


class SpacyTokenizerAdapter(CleaningEngine):
    """
    Adaptateur utilisant spaCy pour une lemmatisation précise.
    
    Contrairement à NLTK, spaCy utilise des modèles statistiques pour identifier 
    la racine des mots (lemme) en fonction de leur rôle grammatical (POS tagging).
    
    Note: spaCy ne supporte pas nativement le 'stemming' car il privilégie 
    la précision linguistique du 'lemmatizing'.
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialise l'adaptateur spaCy.

        Args:
            model (str): Le nom du modèle linguistique spaCy à charger. 
                         Exemples : 'en_core_web_sm' (anglais), 'fr_core_news_sm' (français).
                         Le modèle doit être préalablement téléchargé via spacy download.
        """
        import spacy
        try:
            # On désactive les composants inutiles pour la vitesse (NER et Parser)
            self.nlp = spacy.load(model, disable=["parser", "ner"])
            logger.info(f"SpacyTokenizerAdapter : Modèle '{model}' chargé avec succès.")
        except OSError as e:
            logger.error(f"SpacyTokenizerAdapter : Modèle '{model}' introuvable.")
            # Use 'from e' to link the new exception to the original cause
            raise OSError(f"Exécuter : python -m spacy download {model}") from e


    def process_text(self, text: str) -> List[str]:
        """
        Traite le texte via le pipeline spaCy.
        Exclut les stop-words et les espaces.
        """
        doc = self.nlp(text)
        return [t.lemma_ for t in doc if not t.is_stop and not t.is_space]


class NltkTokenizerAdapter(CleaningEngine):
    """
    Adaptateur utilisant NLTK pour le stemming ou la lemmatisation légère.
    
    Idéal pour des traitements rapides ou lorsque le 'stemming' (réduction brutale 
    à la racine) est requis pour augmenter le rappel (recall).
    """
    
    def __init__(self, method: str = "lemmatize"):
        """
        Args:
            method (str): 'lemmatize' pour la forme canonique, 'stem' pour la racine.
        """
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        
        self.method = method
        try:
            self.stop_words = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
        except LookupError:
            logger.warning("NltkTokenizerAdapter : Téléchargement des ressources nécessaires...")
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            self.stop_words = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
        
        logger.info(f"NltkTokenizerAdapter : Initialisé en mode '{method}'.")

    def process_text(self, text: str) -> List[str]:
        """
        Découpe le texte par espaces et applique la méthode choisie.
        """
        words = text.split()
        if self.method == "stem":
            return [self.stemmer.stem(w) for w in words if w not in self.stop_words]
        return [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]