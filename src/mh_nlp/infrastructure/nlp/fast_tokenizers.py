# mh_nlp/infrastructure/nlp/fast_tokenizers.py 

from typing import List
from loguru import logger
from tqdm import tqdm

from mh_nlp.domain.ports.fast_cleaning_engine import FastCleaningEngine

class FastSpacyTokenizerAdapter(FastCleaningEngine):
    """
    Adaptateur haute performance utilisant spaCy pour la lemmatisation.
    
    Cette implémentation est optimisée pour le traitement de gros volumes de données
    en utilisant les pipelines natifs de spaCy (nlp.pipe) et une barre de 
    progression visuelle pour le monitoring.
    """
    
    def __init__(self, model: str = "en_core_web_sm", batch_size: int = 1000):
        """
        Initialise le moteur spaCy avec des composants optimisés.

        Args:
            model (str): Le modèle linguistique à charger (ex: 'en_core_web_sm').
            batch_size (int): Nombre de textes traités simultanément dans le buffer.
                              Augmenter cette valeur améliore la vitesse sur les gros datasets.
        """
        import spacy
        self.batch_size = batch_size
        try:
            # 'disable' retire les composants lourds non nécessaires à la lemmatisation simple.
            # Cela réduit drastiquement la consommation de RAM et de CPU.
            self.nlp = spacy.load(model, disable=["parser", "ner"])
            logger.info(f"SpacyTokenizerAdapter : Modèle '{model}' prêt (Batch: {batch_size}).")
        except OSError:
            logger.error(f"SpacyTokenizerAdapter : Modèle '{model}' non trouvé localement.")
            raise OSError(f"Veuillez télécharger le modèle avec : python -m spacy download {model}")

    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Traite une liste massive de textes de manière optimisée.

        Utilise nlp.pipe pour minimiser les passages dans le moteur Global Interpreter Lock (GIL)
        de Python et tqdm pour fournir un feedback visuel sur l'avancement.

        Args:
            texts (List[str]): Collection de textes à traiter.
        Returns:
            List[List[str]]: Liste de listes de tokens lemmatisés.
        """
        logger.info(f"Démarrage du traitement par lots ({len(texts)} lignes).")
        
        processed_docs = []
        
        # tqdm wrap le générateur nlp.pipe pour calculer le temps restant estimé (ETA).
        for doc in tqdm(self.nlp.pipe(texts, batch_size=self.batch_size), 
                        total=len(texts), 
                        desc="Batch Lemmatization (spaCy)"):
            
            # Extraction des lemmes avec filtrage intégré
            tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_space]
            processed_docs.append(tokens)
            
        logger.success("Traitement massif terminé avec succès.")
        return processed_docs