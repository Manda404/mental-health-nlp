from typing import List

import numpy as np
from loguru import logger
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer as KTokenizer

from mh_nlp.domain.entities.document import Document
from mh_nlp.domain.ports.tokenizer import Tokenizer


class KerasTextTokenizer(Tokenizer):
    """
    Implémentation du port Tokenizer utilisant TensorFlow/Keras.
    
    Adapté pour les architectures de type CNN ou LSTM, ce tokenizer convertit 
    les textes en séquences d'entiers de longueur fixe (padding/truncation).
    """

    def __init__(self, vocab_size: int, max_length: int):
        """
        Initialise le tokenizer Keras.

        Args:
            vocab_size (int): Nombre maximum de mots à conserver dans le dictionnaire.
            max_length (int): Longueur fixe des séquences en sortie.
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.tokenizer = KTokenizer(num_words=vocab_size, oov_token="<OOV>")
        
        logger.info(
            f"KerasTextTokenizer initialisé (Vocab target: {vocab_size}, Max length: {max_length})"
        )

    def fit(self, documents: List[Document]) -> None:
        """
        Apprend le dictionnaire de mots à partir d'un corpus de documents.

        Args:
            documents (List[Document]): Documents d'entraînement pour construire l'index.
        """
        if not documents:
            logger.warning("Tentative de 'fit' sur une liste de documents vide.")
            return

        texts = [doc.text for doc in documents]
        logger.info(f"Apprentissage du vocabulaire sur {len(texts)} documents...")
        
        self.tokenizer.fit_on_texts(texts)
        
        # Calcul de la taille réelle du vocabulaire trouvé
        found_vocab_size = len(self.tokenizer.word_index)
        logger.success(f"Vocabulaire appris : {found_vocab_size} mots uniques trouvés.")

    def tokenize(self, documents: List[Document]) -> np.ndarray:
        """
        Transforme les documents en séquences numériques paddées.

        Args:
            documents (List[Document]): Liste de documents à transformer.

        Returns:
            np.ndarray: Matrice de forme (nb_documents, max_length).
        """
        if not documents:
            logger.warning("Tokenize appelé avec une liste vide.")
            return np.array([])

        texts = [doc.text for doc in documents]
        
        try:
            # 1. Conversion texte -> listes d'entiers
            sequences = self.tokenizer.texts_to_sequences(texts)
            
            # 2. Application du padding (post-padding par défaut pour les CNN)
            padded_sequences = pad_sequences(
                sequences, 
                maxlen=self.max_length, 
                padding="post",
                truncating="post"
            )
            
            logger.debug(
                f"Séquences générées : {padded_sequences.shape} (Type: {padded_sequences.dtype})"
            )
            return padded_sequences

        except Exception as e:
            logger.error(f"Erreur lors de la transformation Keras : {e}")
            raise RuntimeError("Échec de la tokenisation/padding Keras") from e