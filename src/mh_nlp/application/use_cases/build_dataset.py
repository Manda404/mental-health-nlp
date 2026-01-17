
from loguru import logger

from mh_nlp.application.dto.dataset_dto import DatasetDTO
from mh_nlp.domain.entities.document import Document
from mh_nlp.domain.ports.dataset_repository import DatasetRepository
from mh_nlp.domain.services.text_cleaner import TextCleaner
 

class BuildCleanDatasetUseCase:
    """
    Orchestrateur du pipeline de préparation des données.
    
    Ce cas d'usage fait le pont entre le stockage (Repository) et les règles de 
    transformation (Service). Il garantit que les données sortantes sont 
    prêtes pour l'entraînement ou l'inférence sans risque d'erreur de format.
    """

    def __init__(self, repository: DatasetRepository, cleaner: TextCleaner):
        """
        Initialise le cas d'usage avec ses ports et services.

        Args:
            repository (DatasetRepository): Port permettant d'extraire les données brutes.
            cleaner (TextCleaner): Service domaine chargé de la normalisation textuelle 
                                  (suppression stop-words, ponctuation, etc.).
        """
        self.repository = repository
        self.cleaner = cleaner  # Correction : stockage de la dépendance

    def execute(self) -> DatasetDTO:
        """
        Pilote le processus de nettoyage de bout en bout.

        Déroulement :
        1. Chargement des paires (Document, Label) depuis le repository.
        2. Exclusion des données ayant un label de classe -1 (données non étiquetées).
        3. Normalisation textuelle via le service TextCleaner.
        4. Suppression des documents dont le contenu est devenu vide après nettoyage.

        Returns:
            DatasetDTO: Objet de transfert de données contenant les textes propres et leurs labels.
        
        Raises:
            Exception: Si une erreur survient durant le chargement ou le traitement.
        """
        logger.info("Démarrage du pipeline BuildCleanDataset.")

        # --- 1. ACQUISITION ET NETTOYAGE DES LABELS ---
        # Le repository renvoie typiquement une liste de Tuple[Document, LabelEntity]
        raw_data = self.repository.load()
        
        # On élimine les labels invalides (-1) dès le départ pour ne pas 
        # gaspiller de ressources CPU sur le nettoyage de textes inutilisables.
        valid_pairs = [(doc, lbl) for doc, lbl in raw_data if lbl.index != -1]
        
        if not valid_pairs:
            logger.warning("Pipeline interrompu : aucune donnée valide après filtrage des labels.")
            return DatasetDTO(documents=[], labels=[], total_records=0)
        
        docs_to_process = [p[0] for p in valid_pairs]
        labels_to_process = [p[1].index for p in valid_pairs]

        # --- 2. TRAITEMENT TEXTUEL (LOGIQUE DOMAINE) ---
        # Ici, on délègue au service TextCleaner qui contient les expressions régulières, émojis, etc...
        logger.debug(f"Traitement unitaire de {len(docs_to_process)} documents.")
        cleaned_texts = [self.cleaner.clean(doc) for doc in docs_to_process]

        # --- 3. SYNCHRONISATION ET RECONSTITUTION DES ENTITÉS DU DOMAINE ---
        
        final_documents = []
        final_labels = []

        # On itère simultanément sur les textes nettoyés et les labels d'origine.
        # Le 'strict=True' est notre garde-fou : il garantit qu'aucun décalage 
        # n'est possible entre une donnée et son étiquette.
        for text, label_idx in zip(cleaned_texts, labels_to_process, strict=True):
            
            # Condition de garde : on ne conserve que les documents ayant une 
            # substance sémantique après nettoyage.
            if text and text.strip():
                # RE-MATÉRIALISATION : On recrée une entité Document propre
                # pour le domaine à partir de la chaîne de caractères traitée.
                final_documents.append(Document(text))
                
                # ALIGNEMENT : On associe le label à l'entité nouvellement créée.
                final_labels.append(label_idx)

        # Calcul de la perte de volume pour le monitoring de la qualité du corpus
        dropped_count = len(docs_to_process) - len(final_documents)
        
        logger.success(
            f"Pipeline terminé : {len(final_documents)} entités synchronisées. "
            f"({dropped_count} supprimées car vides)."
        )

        # Encapsulation dans le DTO pour transfert à la couche Application/ML
        return DatasetDTO(
            documents=final_documents,
            labels=final_labels,
            total_records=len(final_documents)
        )