from loguru import logger

from mh_nlp.application.dto.dataset_dto import DatasetDTO
from mh_nlp.domain.ports.classifier import TextClassifier


class TrainModelUseCase:
    """
    Cas d’usage : Orchestrer l'entraînement d'un modèle de classification.

    Cette classe sert de pont entre la logique applicative et les implémentations
    spécifiques de modèles (Scikit-Learn, PyTorch, etc.), permettant de changer
    de moteur d'entraînement sans modifier le reste de l'application.

    Attributes:
        classifier (TextClassifier): Le port (interface) vers le modèle à entraîner.
    """

    def __init__(self, classifier: TextClassifier):
        """
        Initialise le cas d'usage avec un classifieur spécifique.

        Args:
            classifier (TextClassifier): Une implémentation concrète de l'interface
                de classification (ex: SpacyClassifier, SklearnClassifier).
        """
        self.classifier = classifier
        logger.debug(f"TrainModelUseCase initialisé avec {type(classifier).__name__}")

    def execute(self, train_data: DatasetDTO, val_data: DatasetDTO) -> None:
        """
        Lance le processus d'entraînement et de validation.

        Cette méthode prépare le monitoring, délègue l'apprentissage au moteur
        choisi et valide la réussite de l'opération.

        Args:
            train_data (DatasetDTO): Données étiquetées pour l'apprentissage.
            val_data (DatasetDTO): Données pour le monitoring de l'overfitting.
        
        Raises:
            ValueError: Si les jeux de données sont vides.
            RuntimeError: En cas d'échec technique pendant l'entraînement.
        """
        # Vérification pré-entraînement
        if train_data.total_records == 0:
            logger.error("Tentative d'entraînement avec un jeu de données vide.")
            raise ValueError("Le jeu d'entraînement (train_data) ne contient aucun document.")

        logger.info(f"Démarrage de l'entraînement : {train_data.total_records} docs (train) | "
                    f"{val_data.total_records} docs (val)")

        try:
            # Délégation au port de domaine
            self.classifier.train(train_data, val_data)
            
            logger.success(
                f"Modèle entraîné avec succès via {self.classifier.__class__.__name__}."
            )
            
        except Exception as err:
            logger.critical(f"Erreur fatale lors de l'entraînement du modèle : {err}")
            # On relève l'erreur pour la gestion au niveau de l'API/Interface
            raise RuntimeError("Échec de la phase d'apprentissage du modèle.") from err