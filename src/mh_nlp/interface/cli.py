"""
Module d'interface en ligne de commande (CLI).

Ce module sert de point d'entrée pour l'utilisation de l'application hors Notebook.
Il orchestre la création des composants via les Factories et délègue l'exécution 
aux Use Cases de la couche Application.

---------------------------------------------------------------------
EXEMPLES D'UTILISATION AVEC POETRY :
---------------------------------------------------------------------
Préalable : Assurez-vous d'avoir configuré [tool.poetry.scripts] dans pyproject.toml
et lancé 'poetry install'.

1. Entraîner DistilBERT pendant 5 époques :
   $ poetry run mh-nlp train --model distilbert --epochs 5

2. Entraîner le modèle hybride (BERT+CNN) :
   $ poetry run mh-nlp train --model distilbert_cnn

3. Prédire la classe pour une phrase simple :
   $ poetry run mh-nlp predict --model distilbert --texts "I feel very anxious today"

4. Prédire pour plusieurs phrases simultanément :
   $ poetry run mh-nlp predict --model roberta --texts "Happy" "Sad" "I can't sleep"

Note : Si vous n'avez pas configuré le script dans pyproject.toml, utilisez :
   $ poetry run python -m mh_nlp.interface.cli [command] [args]
---------------------------------------------------------------------
"""

import argparse
from loguru import logger
from typing import List, Tuple
import torch

from mh_nlp.application.use_cases.train_model import TrainModelUseCase
from mh_nlp.application.use_cases.predict_text import PredictTextUseCase
from mh_nlp.infrastructure.config.model_factory import ModelFactory
from mh_nlp.infrastructure.config.tokenizer_factory import TokenizerFactory


# ---------------------------------------------------------------------
# UTILS / CONFIGURATION
# ---------------------------------------------------------------------

def _setup_infrastructure(model_type: str) -> Tuple[object, object, torch.device]:
    """
    Initialise les composants techniques nécessaires (modèle et tokenizer).
    
    Pourquoi cette fonction privée : 
    Évite la duplication du code de setup entre 'train' et 'predict'.
    
    Args:
        model_type (str): Le nom du modèle choisi dans le CLI.
        
    Returns:
        Tuple: (classifier, tokenizer, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation du hardware : {device}")

    # Utilisation des Factories pour découpler la CLI de l'implémentation précise
    tokenizer = TokenizerFactory.create(model_type)
    classifier = ModelFactory.create(
        model_name=model_type,
        tokenizer=tokenizer,
        device=device,
    )
    
    return classifier, tokenizer, device


# ---------------------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Définit la structure des commandes et arguments du CLI.

    Le CLI utilise des 'subparsers' pour séparer proprement les modes 'train' (apprentissage)
    et 'predict' (utilisation du modèle).

    Returns:
        argparse.ArgumentParser: L'objet parser configuré.
    """
    parser = argparse.ArgumentParser(
        description="Mental Health NLP Project - Interface de ligne de commande"
    )

    # Création du groupe de sous-commandes (train, predict)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Commandes disponibles"
    )

    # --- Sous-commande: TRAIN ---
    train_parser = subparsers.add_parser(
        "train",
        help="Entraîner un modèle de classification de texte"
    )
    train_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["distilbert", "distilbert_cnn", "roberta", "cnn"],
        help="Type d'architecture de modèle à entraîner"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=3, # Réduit par défaut pour les Transformers
        help="Nombre d'itérations d'entraînement (époques)"
    )

    # --- Sous-commande: PREDICT ---
    predict_parser = subparsers.add_parser(
        "predict",
        help="Prédire la santé mentale à partir de textes bruts"
    )
    predict_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["distilbert", "distilbert_cnn", "roberta", "cnn"],
        help="Modèle à charger pour l'inférence"
    )
    predict_parser.add_argument(
        "--texts",
        type=str,
        nargs="+", # Permet d'envoyer plusieurs phrases séparées par des espaces
        required=True,
        help="Liste de textes à classifier (ex: 'I feel sad' 'I am okay')"
    )

    return parser


# ---------------------------------------------------------------------
# COMMAND HANDLERS
# ---------------------------------------------------------------------

def handle_train(args: argparse.Namespace):
    """
    Orchestre le processus d'entraînement.

    Note d'architecture : 
    Le handler ne contient aucune logique de calcul. Il instancie le Use Case 
    et lui passe les 'Ports' (le classifier) nécessaires.
    """
    logger.info(f"Démarrage de l'entraînement pour le modèle : {args.model}")

    # Initialisation infra
    classifier, _, _ = _setup_infrastructure(args.model)

    # Injection du classifier dans le Use Case
    use_case = TrainModelUseCase(classifier=classifier)
    
    # Exécution de la logique métier
    # Note: On suppose ici que execute() accepte epochs et gère le chargement data
    use_case.execute(epochs=args.epochs)

    logger.success("Entraînement terminé avec succès")


def handle_predict(args: argparse.Namespace):
    """
    Orchestre le processus de prédiction.
    """
    logger.info(f"Chargement du modèle {args.model} pour prédiction...")

    # Initialisation infra
    classifier, _, _ = _setup_infrastructure(args.model)

    # Inversion de dépendance : on passe l'implémentation au Use Case
    use_case = PredictTextUseCase(classifier=classifier)

    # Récupération des prédictions (liste d'entiers ou labels)
    predictions = use_case.execute(args.texts)

    print("\n" + "="*50)
    print(" RÉSULTATS DE LA PRÉDICTION")
    print("="*50)
    for text, pred in zip(args.texts, predictions):
        print(f" TEXTE      : {text}")
        print(f" PRÉDICTION : {pred}")
        print("-" * 50)


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------

def main():
    """
    Point d'entrée principal de l'application CLI.
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            handle_train(args)
        elif args.command == "predict":
            handle_predict(args)
    except Exception as e:
        logger.error(f"Une erreur fatale est survenue : {e}")
        exit(1)


if __name__ == "__main__":
    main()