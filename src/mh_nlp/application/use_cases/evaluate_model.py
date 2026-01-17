from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score

from mh_nlp.application.dto.dataset_dto import DatasetDTO
from mh_nlp.domain.ports.classifier import TextClassifier


class EvaluateModelUseCase:
    """
    Cas d’usage : évaluer un modèle de classification.

    Optimisé pour les datasets déséquilibrés en utilisant des métriques 
    pondérées (weighted) et des rapports par classe.
    """

    def __init__(self, classifier: TextClassifier):
        self.classifier = classifier

    def execute(self, dataset: DatasetDTO) -> dict:
        """
        Calcule les performances du modèle sur un jeu de données.

        Returns:
            dict: Dictionnaire contenant l'accuracy, le f1_score et le rapport détaillé.
        """
        logger.info(f"Évaluation du modèle sur {len(dataset.documents)} documents...")

        # Inférence
        predictions = self.classifier.predict(dataset.documents)
        y_true = dataset.labels

        # Calcul des métriques
        # 'weighted' calcule la moyenne pondérée par le nombre d'échantillons de chaque classe
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average='weighted')
        
        # Le rapport détaillé (Précision/Recall/F1 par classe)
        report = classification_report(y_true, predictions, output_dict=True)

        logger.success(f"Évaluation terminée | Accuracy: {accuracy:.4f} | F1-Score (weighted): {f1:.4f}")

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "detailed_report": report
        }