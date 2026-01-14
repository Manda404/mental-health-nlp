from loguru import logger

from mh_nlp.application.dto.clean_dataset_dto import CleanedDatasetDTO
from mh_nlp.domain.ports.dataset_repository import DatasetRepository
from mh_nlp.domain.services.text_cleaner import TextCleaner


class BuildCleanDatasetUseCase:
    """
    Cas d'usage : Transformer les données brutes de Kaggle en un dataset NLP propre.
    
    Pourquoi ce Use Case :
    - Centralise l'orchestration : Chargement -> Nettoyage -> Formatage.
    - Est totalement indépendant du modèle ML final (DistilBERT ou CNN).
    - Permet de tester toute la chaîne de préparation sans lancer d'entraînement.
    """

    def __init__(self, repository: DatasetRepository, cleaner: TextCleaner):
        """
        Injection des dépendances.
        
        Note : On injecte l'interface DatasetRepository, ce qui permet 
        d'utiliser ce Use Case avec n'importe quelle source de données.
        """
        self.repository = repository
        self.cleaner = cleaner

    def execute(self) -> CleanedDatasetDTO:
        """
        Exécute la logique métier de préparation des données.
        """
        logger.info("Application : Démarrage du Use Case BuildCleanDataset.")

        # 1. Chargement des données (via l'Infrastructure)
        raw_entities = self.repository.load()
        
        cleaned_texts = []
        final_labels = []

        # 2. Orchestration du nettoyage
        for doc, label in raw_entities:

            # On retire les lignes donc les labels sont inconnus
            if label == -1:
                continue

            # On utilise notre service de domaine
            cleaned_text = self.cleaner.clean(doc)
            
            # On ne garde que les documents qui ont encore du contenu après nettoyage
            if cleaned_text:
                cleaned_texts.append(cleaned_text)
                final_labels.append(label.index)

        logger.success(f"Application : Dataset construit. {len(cleaned_texts)} documents valides conservés.")

        # 3. Retour sous forme de DTO
        return CleanedDatasetDTO(
            documents=cleaned_texts,
            labels=final_labels,
            total_processed=len(cleaned_texts)
        )