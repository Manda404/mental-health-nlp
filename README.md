```text
mental-health-nlp/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_build_dataset.ipynb
│   ├── 03_train_distilbert.ipynb
│   ├── 04_train_distilbert_cnn.ipynb
│   ├── 05_train_cnn.ipynb
│   └── 06_inference.ipynb
├── configs/
│   ├── dataset.yaml
│   ├── distilbert.yaml
│   ├── distilbert_cnn.yaml
│   ├── cnn.yaml
│   └── roberta.yaml
├── src/
│   └── mh_nlp/
│       ├── domain/
│       │   ├── entities/
│       │   │   ├── document.py
│       │   │   └── label.py
│       │   ├── ports/
│       │   │   ├── tokenizer.py
│       │   │   ├── classifier.py
│       │   │   └── dataset_repository.py
│       │   └── services/
│       │       └── text_cleaner.py
│       ├── application/
│       │   ├── use_cases/
│       │   │   ├── build_dataset.py
│       │   │   ├── split_dataset.py
│       │   │   ├── train_model.py
│       │   │   ├── evaluate_model.py
│       │   │   └── predict.py
│       │   └── dto/
│       ├── infrastructure/
│       │   ├── data/
│       │   │   └── kaggle_repository.py
│       │   ├── nlp/
│       │   │   ├── hf_tokenizer.py
│       │   │   └── keras_tokenizer.py
│       │   ├── models/
│       │   │   ├── distilbert_classifier.py
│       │   │   ├── distilbert_cnn_classifier.py
│       │   │   ├── roberta_classifier.py
│       │   │   └── cnn_classifier.py
│       │   └── training/
│       │       └── torch_trainer.py
│       ├── interface/
│       │   └── cli.py
│       └── utils/
│           ├── seed.py
│           └── logging.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── .github/workflows/
│   ├── ci.yml
│   └── release.yml
├── pyproject.toml
└── README.md
```