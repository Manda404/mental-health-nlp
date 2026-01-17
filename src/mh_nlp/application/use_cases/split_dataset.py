from typing import Dict

from loguru import logger
from sklearn.model_selection import train_test_split

from mh_nlp.application.dto.dataset_dto import DatasetDTO


class SplitDatasetUseCase:
    """
    Cas d’usage : découper un dataset en train / validation / test.

    Responsabilités :
    - Définir une stratégie de split stable (random_state).
    - Garantir la stratification des labels pour conserver la distribution des classes.
    - Calculer dynamiquement les ratios pour obtenir des tailles de sets cohérentes.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialise les paramètres de découpage.

        Args:
            test_size (float): Proportion du jeu de test (ex: 0.2 pour 20%).
            val_size (float): Proportion du jeu de validation par rapport au total.
            random_state (int): Graine pour la reproductibilité des résultats.
        """
        if test_size + val_size >= 1.0:
            raise ValueError("La somme de test_size et val_size doit être inférieure à 1.0")

        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def execute(self, dataset: DatasetDTO) -> Dict[str, DatasetDTO]:
        """
        Exécute le découpage stratifié du dataset.

        Args:
            dataset (DatasetDTO): Le dataset complet à diviser.

        Returns:
            Dict[str, DatasetDTO]: Dictionnaire contenant les clés 'train', 'val', 'test'.
        """
        documents = dataset.documents
        labels = dataset.labels
        
        logger.info(
            f"Découpage du dataset (Total: {len(documents)}) | "
            f"Cible: Test={self.test_size}, Val={self.val_size}"
        )

        # ---------------------------------------------------------
        # Étape 1 : Isoler le jeu de TEST
        # On utilise des minuscules (x_temp, x_test) pour satisfaire Ruff (N806)
        # ---------------------------------------------------------
        x_temp, x_test, y_temp, y_test = train_test_split(
            documents,
            labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.random_state,
            shuffle=True,
        )

        # ---------------------------------------------------------
        # Étape 2 : Séparer TRAIN et VALIDATION
        # Calcul du ratio relatif pour maintenir les proportions globales
        # ---------------------------------------------------------
        relative_val_size = self.val_size / (1.0 - self.test_size)

        x_train, x_val, y_train, y_val = train_test_split(
            x_temp,
            y_temp,
            test_size=relative_val_size,
            stratify=y_temp,
            random_state=self.random_state,
            shuffle=True,
        )

        logger.success(
            f"Split réussi : Train={len(y_train)} | Val={len(y_val)} | Test={len(y_test)}"
        )

        return {
            "train": DatasetDTO(documents=x_train, labels=y_train, total_records=len(y_train)),
            "val": DatasetDTO(documents=x_val, labels=y_val, total_records=len(y_val)),
            "test": DatasetDTO(documents=x_test, labels=y_test, total_records=len(y_test)),
        }