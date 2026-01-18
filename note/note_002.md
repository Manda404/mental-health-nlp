# Clean Architecture in Mental Health NLP
# Clean Architecture pour un Projet NLP en Santé Mentale

## Introduction

Cette architecture suit les principes de la **Clean Architecture** (Robert C. Martin) appliqués à un projet de classification NLP pour la détection de problèmes de santé mentale. L'objectif est de créer un système maintenable, testable et évolutif en séparant clairement les responsabilités.

### Principe fondamental : La règle de dépendance

Les dépendances pointent toujours **vers l'intérieur** :
```
Interface → Infrastructure → Application → Domain
   (UI)      (Frameworks)    (Use Cases)   (Métier)
```

Le **domaine** (cœur métier) ne connaît rien des frameworks, bases de données ou interfaces utilisateur. C'est l'infrastructure qui implémente les contrats définis par le domaine.

---

## Vue d'ensemble des couches

### 🎯 Domain (Cœur métier)
**Rôle** : Contient la logique métier pure, indépendante de toute technologie.

**Composants** :
- **Entities** : Objets métier (Document, Label)
- **Ports** : Interfaces que l'infrastructure devra implémenter
- **Services** : Logique métier réutilisable (nettoyage de texte, validation)

**Interactions** : Ne dépend de RIEN. Les autres couches dépendent de lui.

---

### 🔄 Application (Orchestration)
**Rôle** : Implémente les cas d'usage en orchestrant le domaine.

**Composants** :
- **Use Cases** : Scénarios métier (entraîner un modèle, prédire, évaluer)
- **DTOs** : Objets de transfert de données entre couches

**Interactions** :
- ⬇️ Utilise le **Domain** (entities, ports, services)
- ⬆️ Est appelée par l'**Infrastructure** ou l'**Interface**

---

### 🔧 Infrastructure (Implémentations)
**Rôle** : Implémente les ports du domaine avec des technologies concrètes.

**Composants** :
- **Data** : Adaptateurs pour Kaggle, CSV, bases de données
- **NLP** : Implémentations HuggingFace, Keras pour la tokenisation
- **Models** : Modèles ML concrets (DistilBERT, CNN, RoBERTa)
- **Training** : Framework d'entraînement (PyTorch, TensorFlow)

**Interactions** :
- ⬇️ Implémente les interfaces du **Domain**
- ⬇️ Est injectée dans l'**Application** (dependency injection)

---

### 🖥️ Interface (Points d'entrée)
**Rôle** : Expose le système au monde extérieur.

**Composants** :
- **CLI** : Interface en ligne de commande
- *(Futur : API REST, interface web)*

**Interactions** :
- ⬇️ Appelle l'**Application** (use cases)
- ⬇️ Instancie et injecte l'**Infrastructure**

---

## Structure détaillée avec commentaires

```text
mental-health-nlp/
│
├── data/                                    # 💾 Données du projet (gitignored)
│   ├── raw/                                # Données brutes téléchargées (Kaggle, etc.)
│   ├── processed/                          # Datasets nettoyés et tokenisés
│   └── external/                           # Sources externes (APIs, scraping)
│
├── notebooks/                              # 📊 Exploration et expérimentation (prototypage)
│   ├── 01_explore_dataset.ipynb           # EDA : distribution des labels, longueur textes
│   ├── 02_build_dataset.ipynb             # Préparation : nettoyage, équilibrage
│   ├── 03_train_distilbert.ipynb          # Expérimentation DistilBERT
│   ├── 04_train_distilbert_cnn.ipynb      # Expérimentation architecture hybride
│   ├── 05_train_cnn.ipynb                 # Baseline CNN classique
│   └── 06_inference.ipynb                 # Tests de prédiction et analyse erreurs
│
├── configs/                                # ⚙️ Configuration externalisée (YAML)
│   ├── dataset.yaml                       # Sources, chemins, ratio train/val/test
│   ├── distilbert.yaml                    # Learning rate, batch size, epochs, etc.
│   ├── distilbert_cnn.yaml                # Config architecture hybride
│   ├── cnn.yaml                           # Hyperparamètres CNN (filters, kernel_size)
│   └── roberta.yaml                       # Config RoBERTa (fine-tuning)
│
├── src/
│   └── mh_nlp/                            # 📦 Package principal
│       │
│       ├── domain/                         # 🎯 COUCHE DOMAINE (Clean Core)
│       │   │                              # → Logique métier pure
│       │   │                              # → ZÉRO dépendance externe
│       │   │                              # → Définit les règles métier
│       │   │
│       │   ├── entities/                  # Objets métier riches
│       │   │   ├── document.py            # class Document: texte, id, metadata
│       │   │   │                          # → Méthodes : validate(), clean()
│       │   │   └── label.py               # class Label: nom, id, description
│       │   │                              # → Enum des catégories (Depression, Anxiety...)
│       │   │
│       │   ├── ports/                     # 🔌 Interfaces (Design by Contract)
│       │   │   │                          # → Définissent "QUOI" sans "COMMENT"
│       │   │   ├── tokenizer.py           # Protocol Tokenizer: tokenize(text) -> tokens
│       │   │   ├── classifier.py          # Protocol Classifier: predict(), train()
│       │   │   └── dataset_repository.py  # Protocol Repository: load(), save()
│       │   │
│       │   └── services/                  # Services du domaine (logique réutilisable)
│       │       └── text_cleaner.py        # clean_text(text) -> cleaned_text
│       │                                  # → Règles métier : normalisation, stopwords
│       │
│       ├── application/                    # 🔄 COUCHE APPLICATION (Use Cases)
│       │   │                              # → Orchestration du domaine
│       │   │                              # → Dépend du Domain (entities, ports)
│       │   │                              # → Indépendante de l'implémentation
│       │   │
│       │   ├── use_cases/                 # Scénarios métier (user stories)
│       │   │   ├── build_dataset.py       # UC: Charger, nettoyer, sauvegarder dataset
│       │   │   │                          # → Utilise DatasetRepository (port)
│       │   │   ├── split_dataset.py       # UC: Diviser en train/val/test (stratifié)
│       │   │   ├── train_model.py         # UC: Entraîner modèle + sauvegarder checkpoints
│       │   │   │                          # → Utilise Classifier (port)
│       │   │   ├── evaluate_model.py      # UC: Calculer métriques (F1, accuracy, etc.)
│       │   │   └── predict.py             # UC: Prédire label pour nouveau texte
│       │   │
│       │   └── dto/                       # Data Transfer Objects (immutables)
│       │       ├── prediction_result.py   # Résultat prédiction (label, confidence)
│       │       └── training_metrics.py    # Métriques d'entraînement (loss, accuracy)
│       │
│       ├── infrastructure/                 # 🔧 COUCHE INFRASTRUCTURE (Implémentations)
│       │   │                              # → Implémente les ports du Domain
│       │   │                              # → Contient les détails techniques
│       │   │                              # → Dépendances externes autorisées
│       │   │
│       │   ├── data/                      # Adaptateurs pour sources de données
│       │   │   └── kaggle_repository.py   # Implémente DatasetRepository avec Kaggle API
│       │   │                              # → Peut être remplacé par CSVRepository, etc.
│       │   │
│       │   ├── nlp/                       # Adaptateurs pour outils NLP
│       │   │   ├── hf_tokenizer.py        # Implémente Tokenizer avec Transformers
│       │   │   │                          # → AutoTokenizer.from_pretrained()
│       │   │   └── keras_tokenizer.py     # Implémente Tokenizer avec Keras
│       │   │                              # → Tokenizer(num_words=10000)
│       │   │
│       │   ├── models/                    # Implémentations des modèles ML
│       │   │   ├── distilbert_classifier.py     # DistilBertForSequenceClassification
│       │   │   ├── distilbert_cnn_classifier.py # Hybrid: DistilBERT + CNN layers
│       │   │   ├── roberta_classifier.py        # RoBERTa fine-tuning
│       │   │   └── cnn_classifier.py            # CNN classique (baseline)
│       │   │   # → Tous implémentent l'interface Classifier
│       │   │
│       │   └── training/                  # Infrastructure d'entraînement
│       │       └── torch_trainer.py       # Boucle d'entraînement PyTorch
│       │                                  # → Callbacks, early stopping, logging
│       │
│       ├── interface/                      # 🖥️ COUCHE INTERFACE (Points d'entrée)
│       │   │                              # → Exposition du système
│       │   │                              # → Instancie et injecte les dépendances
│       │   │
│       │   └── cli.py                     # CLI avec Typer ou argparse
│       │       # Commandes :
│       │       # - train --model distilbert --config configs/distilbert.yaml
│       │       # - predict --text "I feel anxious" --model saved_model/
│       │       # - evaluate --model saved_model/ --test-data data/test.csv
│       │
│       └── utils/                         # 🛠️ Utilitaires transversaux
│           ├── seed.py                    # set_seed(42) pour reproductibilité
│           └── logging.py                 # Configuration logger (format, niveau)
│
├── tests/                                  # 🧪 Tests automatisés (TDD)
│   ├── unit/                              # Tests unitaires (domaine isolé)
│   │   ├── test_entities.py              # Test Document, Label (validation)
│   │   └── test_services.py              # Test TextCleaner (mock dependencies)
│   │
│   ├── integration/                       # Tests d'intégration (use cases)
│   │   ├── test_train_use_case.py        # Test entraînement avec mock repository
│   │   └── test_predict_use_case.py      # Test prédiction end-to-end
│   │
│   └── e2e/                               # Tests end-to-end (scénarios réels)
│       └── test_full_pipeline.py         # Test dataset → train → predict
│
├── .github/workflows/                      # 🚀 CI/CD avec GitHub Actions
│   ├── ci.yml                             # Lint (ruff), tests (pytest), coverage
│   └── release.yml                        # Build package, publish to PyPI
│
├── pyproject.toml                         # 📋 Configuration projet (Poetry/setuptools)
│   # Dependencies:
│   # - transformers, torch (infrastructure)
│   # - pydantic (domain entities)
│   # - pytest, pytest-cov (tests)
│
└── README.md                              # 📖 Documentation projet
```

---

## Flux d'exécution (exemple : Entraîner un modèle)

```
┌─────────────┐
│   CLI       │  1. Utilisateur lance : python -m mh_nlp train --model distilbert
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│   Interface (cli.py)        │  2. Parse arguments, charge config YAML
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│   Infrastructure                        │  3. Instancie les adaptateurs :
│   - KaggleRepository (data)             │     - repository = KaggleRepository()
│   - HFTokenizer (nlp)                   │     - tokenizer = HFTokenizer("distilbert-base")
│   - DistilBertClassifier (models)       │     - model = DistilBertClassifier()
└──────┬──────────────────────────────────┘
       │
       │  4. Injection de dépendances
       ▼
┌─────────────────────────────┐
│   Application               │  5. Exécute le use case :
│   TrainModelUseCase         │     use_case = TrainModelUseCase(
│                             │         repository=repository,
│                             │         tokenizer=tokenizer,
│                             │         classifier=model
│                             │     )
│                             │     use_case.execute(config)
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   Domain                    │  6. Utilise les services :
│   - TextCleaner             │     - Nettoie les textes
│   - Document entities       │     - Valide les documents
│   - Label entities          │     - Encode les labels
└─────────────────────────────┘
```

---

## Avantages de cette architecture

### ✅ Testabilité
- Le domaine est testable sans PyTorch, HuggingFace ou Kaggle
- Les use cases sont testables avec des mocks (repositories, classifiers)
- Tests pyramide : beaucoup d'unit tests, moins d'integration, peu d'e2e

### ✅ Flexibilité
- Changer de DistilBERT → RoBERTa : modifier uniquement l'infrastructure
- Remplacer Kaggle par une API custom : un seul fichier à changer
- Ajouter une interface web : créer `interface/web.py` sans toucher au reste

### ✅ Maintenabilité
- Séparation claire des responsabilités
- Code métier isolé des frameworks (moins de couplage)
- Plus facile à comprendre et à onboarder de nouveaux développeurs

### ✅ Évolutivité
- Ajouter de nouveaux modèles : implémenter l'interface `Classifier`
- Ajouter de nouvelles sources de données : implémenter `DatasetRepository`
- Migration progressive (ex: PyTorch → JAX) sans réécrire tout le code