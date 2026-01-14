from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger

from mh_nlp.domain.entities.document import Document
from mh_nlp.domain.entities.label import Label
from mh_nlp.domain.ports.dataset_repository import DatasetRepository


class KaggleDatasetRepository(DatasetRepository):
    """
    Adaptateur concret transformant les fichiers CSV Kaggle en Entités Domaine.

    Pourquoi cette classe :
    - Elle contient la connaissance technique de la structure du CSV (noms des colonnes).
    - Elle gère le "mapping" entre les labels textuels et les index numériques.
    """

    def __init__(self, csv_path: str, label_mapping: Optional[dict] = None):
        """
        Args:
            csv_path (str): Chemin vers le fichier source.
            label_mapping (dict): Dictionnaire de correspondance (ex: {"Anxiety": 0}).
        """
        self.csv_path = csv_path
        self.label_mapping = label_mapping or {}

    def load(self) -> List[Tuple[Document, Label]]:
        """
        Lit le CSV et effectue la transformation vers le langage du Domaine.
        """
        logger.info(f"Démarrage du chargement des données depuis : {self.csv_path}")

        try:
            # 1. Lecture technique via pandas
            df = pd.read_csv(self.csv_path)

            # 2. Nettoyage technique immédiat
            df = df.drop(columns=["Unnamed: 0"], errors="ignore")

            entities = []

            # 3. Transformation : Ligne de DataFrame -> (Document, Label)
            for _, row in df.iterrows():
                # On s'assure que le texte est bien une chaîne de caractères
                doc = Document(text=str(row["statement"]))

                # Récupération du label et de son index associé
                label_name = str(row["status"])
                label_idx = self.label_mapping.get(label_name, -1)  # -1 si inconnu

                lbl = Label(name=label_name, index=label_idx)

                entities.append((doc, lbl))

            logger.success(
                f"Infrastructure : {len(entities)} paires (Document, Label) créées."
            )
            return entities

        except FileNotFoundError:
            logger.error(
                f"Fichier CSV introuvable au chemin spécifié : {self.csv_path}"
            )
            raise
        except Exception as e:
            logger.critical(
                f"Erreur lors de la conversion des données en Entités : {str(e)}"
            )
            raise
