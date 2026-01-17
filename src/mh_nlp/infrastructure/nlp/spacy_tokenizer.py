from typing import List

import spacy
from loguru import logger
from tqdm import tqdm  # Import de la barre de progression

from mh_nlp.domain.entities.document import Document
from mh_nlp.domain.ports.tokenizer import Tokenizer


class SpacyTokenizer(Tokenizer):
    """
    Implémentation spaCy du port Tokenizer avec suivi de progression.
    
    Optimisé pour le prétraitement linguistique massif via nlp.pipe.
    """

    def __init__(
        self, 
        model_name: str = "en_core_web_sm", 
        lemmatize: bool = True,  # Mis à False pour garder le sens (not, can't)
        remove_stop: bool = True, # Mis à False pour la santé mentale
        batch_size: int = 256
    ):
        """
        Initialise le moteur spaCy.

        Args:
            model_name (str): Modèle spaCy (ex: en_core_web_sm).
            lemmatize (bool): Active/Désactive la réduction au lemme.
            remove_stop (bool): Active/Désactive le filtrage des mots vides.
            batch_size (int): Taille des lots pour le traitement parallèle.
        """
        self.lemmatize = lemmatize
        self.remove_stop = remove_stop
        self.batch_size = batch_size
        self.model_name = model_name
        
        try:
            logger.info(f"Initialisation de SpacyTokenizer : {model_name}")
            # On désactive le 'ner' (entités) et 'parser' (syntaxe) pour aller plus vite
            self.nlp = spacy.load(model_name, disable=["ner", "parser"])
            logger.success(f"Modèle spaCy '{model_name}' prêt.")
        except OSError as e:
            logger.critical(f"Modèle spaCy '{model_name}' non trouvé.")
            raise RuntimeError("Infrastructure NLP indisponible") from e

    def tokenize(self, documents: List[Document]) -> List[List[str]]:
        """
        Tokenise les documents avec une barre de progression tqdm.

        Args:
            documents (List[Document]): Liste des documents à traiter.

        Returns:
            List[List[str]]: Liste de mots par document.
        """
        if not documents:
            return []

        texts = [doc.text for doc in documents]
        total_docs = len(texts)
        
        logger.info(f"Tokenisation de {total_docs} documents (Batch: {self.batch_size})")

        tokenized_docs = []

        try:
            # On enveloppe nlp.pipe avec tqdm pour voir la barre de progression
            # total=total_docs est nécessaire car nlp.pipe est un générateur
            for doc in tqdm(
                self.nlp.pipe(texts, batch_size=self.batch_size), 
                total=total_docs, 
                desc="NLP Processing", # desc="Batch Lemmatization (spaCy)")
                unit="doc"
            ):
                
                # Filtrage (Stop words / Ponctuation)
                if self.remove_stop:
                    tokens = [t for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
                else:
                    # On garde au moins le filtrage des espaces/ponctuation pure si besoin
                    tokens = [t for t in doc if not t.is_space]

                # Extraction du texte ou du lemme
                processed = [t.lemma_ if self.lemmatize else t.text for t in tokens]
                tokenized_docs.append(processed)

            return tokenized_docs

        except Exception as e:
            logger.error(f"Erreur durant le traitement spaCy : {e}")
            raise RuntimeError("Échec de la tokenisation") from e