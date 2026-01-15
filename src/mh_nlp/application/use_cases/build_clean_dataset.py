from typing import Union, List
from loguru import logger

from mh_nlp.application.dto.clean_dataset_dto import CleanedDatasetDTO
from mh_nlp.domain.ports.dataset_repository import DatasetRepository
from mh_nlp.domain.services.text_cleaner import TextCleaner
from mh_nlp.domain.services.fast_text_cleaner import FastTextCleaner

class BuildCleanDatasetUseCase:
    """
    Cas d'usage orchestrant la transformation de données brutes en un dataset NLP normalisé.
    
    Responsabilités :
    1. Orchestration : Coordonne le chargement via le Repository et le nettoyage via le Service.
    2. Filtrage métier : Écarte les données corrompues ou les labels invalides (-1).
    3. Optimisation de flux : Sélectionne dynamiquement la stratégie de nettoyage (Batch vs Unitaire)
       selon les capacités du moteur injecté pour maximiser le débit (throughput).
    
    Avantages de cette approche :
    - Découplage total : Le pipeline ne connaît pas la source des données (CSV, SQL, API).
    - Préparation au ML : Produit un DTO immuable prêt pour la tokenisation DistilBERT.
    """

    def __init__(self, repository: DatasetRepository, cleaner: Union[TextCleaner, FastTextCleaner]):
        """
        Initialise le Use Case avec ses dépendances contractuelles.
        
        Args:
            repository (DatasetRepository): Port d'accès aux données (Inversion de dépendance).
            cleaner (Union[TextCleaner, FastTextCleaner]): Service de domaine pour le nettoyage.
        """
        self.repository = repository
        self.cleaner = cleaner

    def execute(self) -> CleanedDatasetDTO:
        """
        Exécute le pipeline de préparation complet.
        
        Algorithme :
        1. Extraction des entités depuis la source de données.
        2. Premier filtrage sur l'intégrité des labels.
        3. Nettoyage vectorisé (si FastTextCleaner) pour réduire le temps de calcul (O(n)).
        4. Post-filtrage des documents devenus vides après lemmatisation.
        
        Returns:
            CleanedDatasetDTO: Conteneur de données nettoyées et synchronisées.
        """
        logger.info("Application : Lancement du pipeline de nettoyage de données.")

        # --- 1. CHARGEMENT ET PRÉ-FILTRAGE ---
        # On charge tout en mémoire (ou par itérateur selon l'implémentation du repo)
        raw_data = self.repository.load()
        
        # Filtrage des labels invalides avant traitement pour économiser du CPU
        valid_pairs = [(doc, lbl) for doc, lbl in raw_data if lbl.index != -1]
        
        if not valid_pairs:
            logger.warning("Aucune donnée valide trouvée après filtrage des labels.")
            return CleanedDatasetDTO(documents=[], labels=[], total_processed=0)

        docs_to_process = [p[0] for p in valid_pairs]
        labels_to_process = [p[1].index for p in valid_pairs]

        # --- 2. STRATÉGIE DE NETTOYAGE ---
        # On vérifie la capacité de traitement par lot (Batch)
        # On privilégie clean_batch pour spaCy/NLTK afin d'optimiser le GIL Python
        if isinstance(self.cleaner, FastTextCleaner):
            logger.info(f"Utilisation du mode Batch ({self.cleaner.__class__.__name__})")
            cleaned_texts = self.cleaner.clean_batch(docs_to_process)
        else:
            logger.info("Utilisation du mode Unitaire (Fallback)")
            cleaned_texts = [self.cleaner.clean(d) for d in docs_to_process]

        # --- 3. POST-FILTRAGE ET SYNCHRONISATION ---
        # On élimine les textes vides (ex: texte qui ne contenait que des stop-words)
        # tout en gardant l'alignement strict avec les labels.
        final_texts = []
        final_labels = []

        for text, label_idx in zip(cleaned_texts, labels_to_process):
            if text.strip():
                final_texts.append(text)
                final_labels.append(label_idx)

        logger.success(
            f"Dataset prêt : {len(final_texts)} documents conservés "
            f"({len(docs_to_process) - len(final_texts)} écartés car vides)."
        )

        return CleanedDatasetDTO(
            documents=final_texts,
            labels=final_labels,
            total_processed=len(final_texts)
        )