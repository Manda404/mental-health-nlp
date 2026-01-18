# Clean Architecture in Mental Health NLP — A Pedagogical Guide

## Table of Contents

1. [Project Context](#1-project-context)
2. [Why Clean Architecture?](#2-why-clean-architecture)
3. [Architectural Foundations](#3-architectural-foundations)
4. [Layer-by-Layer Implementation](#4-layer-by-layer-implementation)
5. [Dependency Management](#5-dependency-management)
6. [Practical Benefits Demonstrated](#6-practical-benefits-demonstrated)
7. [Common Pitfalls Avoided](#7-common-pitfalls-avoided)

---

## 1. Project Context

### The Challenge

Building a mental health sentiment classification system presents several engineering challenges beyond the ML model itself:

- **Multiple ML frameworks**: PyTorch transformers vs. TensorFlow CNNs
- **Evolving requirements**: Today 3 classes, tomorrow 7 classes
- **Different interfaces**: Notebooks for experimentation, API for production, CLI for automation
- **Testing complexity**: How to test business logic without training actual models?
- **Team collaboration**: Data scientists, ML engineers, and backend developers working together

### The Traditional Approach (and its problems)

Most ML projects start with a notebook:

```python
# ❌ Common anti-pattern
# notebook: train_model.ipynb

import pandas as pd
from transformers import DistilBertForSequenceClassification
import torch

# Load data
df = pd.read_csv("data.csv")
df['text'] = df['text'].str.lower().str.replace('[^a-z\s]', '')

# Train model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
# ... 200 lines of training code ...

# Save model
torch.save(model, "model.pt")
```

**Problems with this approach:**

1. **Tight coupling**: Data processing, model training, and persistence all mixed together
2. **Impossible to test**: Can't test text cleaning without loading the entire model
3. **Framework lock-in**: Switching from PyTorch to TensorFlow means rewriting everything
4. **No reusability**: Want to add an API? Copy-paste 300 lines of code
5. **Collaboration nightmare**: Merge conflicts in 1000-line notebooks

### The Clean Architecture Solution

Clean Architecture solves these problems by enforcing **separation of concerns** through **layers** and **dependency inversion**.

---

## 2. Why Clean Architecture?

### Core Problem Statement

**"How do we build ML systems that remain flexible, testable, and maintainable as requirements evolve?"**

### The Three Fundamental Principles

#### Principle 1: Independence from Frameworks

**Business logic should not depend on PyTorch, TensorFlow, FastAPI, or any specific library.**

**Why?** Frameworks change, get deprecated, or become unsuitable. Your business rules don't.

**Example in this project:**

```python
# ✅ Domain layer (framework-agnostic)
class TextClassifier(ABC):
    """Abstract classifier - no mention of PyTorch or TensorFlow"""
    
    @abstractmethod
    def train(self, documents: List[Document], labels: List[Label]) -> Metrics:
        pass
    
    @abstractmethod
    def predict(self, documents: List[Document]) -> List[Label]:
        pass
```

This interface can be implemented by PyTorch, TensorFlow, or even a future framework that doesn't exist yet.

#### Principle 2: Testability

**Every component should be testable in isolation, without expensive dependencies.**

**Why?** You shouldn't need a GPU and 10 minutes to test if text cleaning works correctly.

**Example in this project:**

```python
# ✅ Testing text cleaning without any ML model
def test_text_cleaner_removes_urls():
    cleaner = TextCleaner()
    result = cleaner.clean("Check this https://example.com")
    assert result == "Check this"
    # No model loading, no GPU, runs in milliseconds
```

#### Principle 3: Screaming Architecture

**The folder structure should tell you what the system does, not what frameworks it uses.**

**Bad structure:**
```
project/
├── pytorch_stuff/
├── flask_api/
└── jupyter_notebooks/
```
*"What does this project do?" → Unclear*

**Good structure (ours):**
```
mh_nlp/
├── domain/          # ← "Ah, it's about mental health classification"
│   ├── entities/
│   └── services/
├── application/     # ← "These are the use cases"
│   └── use_cases/
└── infrastructure/  # ← "These are implementation details"
```

---

## 3. Architectural Foundations

### The Dependency Rule

**The Golden Rule:** *Source code dependencies must point inward, toward higher-level policies.*

```
┌─────────────────────────────────────────┐
│         Interface Layer (UI)            │  ← Knows about Application
├─────────────────────────────────────────┤
│       Application Layer (Use Cases)     │  ← Knows about Domain
├─────────────────────────────────────────┤
│    Domain Layer (Business Logic)        │  ← Knows about NOTHING
├─────────────────────────────────────────┤
│  Infrastructure Layer (Implementations) │  ← Knows about Domain
└─────────────────────────────────────────┘
```

**Critical insight:** Infrastructure depends on Domain, not the other way around.

### Layer Responsibilities

| Layer | Responsibility | Examples in This Project |
|-------|---------------|--------------------------|
| **Domain** | Business rules, entity definitions | `Document`, `Label`, `TextClassifier` interface |
| **Application** | Orchestrate domain logic | `TrainModelUseCase`, `PredictUseCase` |
| **Infrastructure** | Technical implementations | `DistilBERTClassifier`, `KerasTokenizer` |
| **Interface** | User interactions | FastAPI routes, CLI commands |

---

## 4. Layer-by-Layer Implementation

### Layer 1: Domain — The Core Business Logic

**Location:** `src/mh_nlp/domain/`

**Purpose:** Encode the fundamental rules of mental health text classification, independent of any technology.

#### 4.1 Entities: The Building Blocks

**File:** `domain/entities/document.py`

```python
@dataclass
class Document:
    """Represents a text document to classify.
    
    This is a BUSINESS concept, not a technical one.
    It could be stored in CSV, JSON, database, or memory.
    """
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("Document text cannot be empty")
```

**Why this matters:**
- ✅ **Framework-independent**: No pandas, no torch tensors, just pure Python
- ✅ **Validates business rules**: Documents must have non-empty text
- ✅ **Self-documenting**: Clear what a "document" means in our domain

**File:** `domain/entities/label.py`

```python
class MentalHealthStatus(Enum):
    """Mental health classification labels.
    
    These are DOMAIN concepts validated by mental health professionals,
    not arbitrary ML categories.
    """
    NORMAL = "Normal"
    ANXIETY = "Anxiety"
    DEPRESSION = "Depression"
    # Future expansion: STRESS, BIPOLAR, etc.

@dataclass
class Label:
    status: MentalHealthStatus
    confidence: Optional[float] = None
```

**Design decision:** Using an `Enum` instead of strings prevents typos and makes the valid categories explicit.

#### 4.2 Ports: The Contracts

**File:** `domain/ports/classifier.py`

```python
class TextClassifier(ABC):
    """Contract that any ML classifier must fulfill.
    
    This is the KEY to framework independence.
    Both PyTorch and TensorFlow models implement this interface.
    """
    
    @abstractmethod
    def train(self, 
              documents: List[Document], 
              labels: List[Label],
              validation_split: float = 0.2) -> TrainingMetrics:
        """Train the classifier on labeled documents."""
        pass
    
    @abstractmethod
    def predict(self, documents: List[Document]) -> List[Label]:
        """Predict mental health status for documents."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a pre-trained model."""
        pass
```

**Why this is powerful:**

```python
# ✅ Application code doesn't care about the implementation
def predict_use_case(classifier: TextClassifier, texts: List[str]):
    docs = [Document(id=str(i), text=t) for i, t in enumerate(texts)]
    return classifier.predict(docs)
    # Works with DistilBERT, RoBERTa, CNN, or a future model!
```

**File:** `domain/ports/dataset_repository.py`

```python
class DatasetRepository(ABC):
    """Contract for data access.
    
    Hides whether data comes from CSV, database, API, or cloud storage.
    """
    
    @abstractmethod
    def load_train_data(self) -> Tuple[List[Document], List[Label]]:
        pass
    
    @abstractmethod
    def save_processed_data(self, documents: List[Document], labels: List[Label]) -> None:
        pass
```

**Benefit:** Switch from CSV to PostgreSQL without changing business logic.

#### 4.3 Services: Domain Logic

**File:** `domain/services/text_cleaner.py`

```python
class TextCleaner:
    """Encapsulates text preprocessing rules.
    
    BUSINESS RULE: Mental health text analysis requires:
    - Lowercase normalization
    - URL removal (irrelevant to sentiment)
    - Special character handling
    - Whitespace normalization
    """
    
    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-z0-9\s]', ' ', text)    # Keep only alphanumeric
        text = re.sub(r'\s+', ' ', text).strip()    # Normalize whitespace
        return text
```

**Why a service?**
- ✅ **Reusable**: Used in training, inference, and API
- ✅ **Testable**: Test cleaning logic independently
- ✅ **Encapsulated**: Changing cleaning rules doesn't affect the rest of the system

---

### Layer 2: Application — Use Cases Orchestration

**Location:** `src/mh_nlp/application/use_cases/`

**Purpose:** Coordinate domain objects to fulfill specific user goals.

#### 4.4 Train Model Use Case

**File:** `application/use_cases/train_model.py`

```python
class TrainModelUseCase:
    """Orchestrates the model training workflow.
    
    BUSINESS WORKFLOW:
    1. Load and clean data
    2. Split into train/validation
    3. Train classifier
    4. Log metrics to MLflow
    5. Save model
    """
    
    def __init__(self,
                 classifier: TextClassifier,           # ← Port (abstraction)
                 dataset_repo: DatasetRepository,      # ← Port (abstraction)
                 text_cleaner: TextCleaner,            # ← Domain service
                 mlflow_client: MLflowClient):         # ← Infrastructure
        self.classifier = classifier
        self.dataset_repo = dataset_repo
        self.text_cleaner = text_cleaner
        self.mlflow = mlflow_client
    
    def execute(self, config: TrainingConfig) -> TrainingResult:
        # Step 1: Load data (abstracted)
        documents, labels = self.dataset_repo.load_train_data()
        
        # Step 2: Clean data (domain logic)
        cleaned_docs = [
            Document(id=doc.id, text=self.text_cleaner.clean(doc.text))
            for doc in documents
        ]
        
        # Step 3: Train (abstracted)
        with self.mlflow.start_run():
            metrics = self.classifier.train(cleaned_docs, labels, 
                                           validation_split=config.val_split)
            self.mlflow.log_metrics(metrics)
            self.classifier.save(config.output_path)
        
        return TrainingResult(metrics=metrics, model_path=config.output_path)
```

**Key observations:**

1. **Dependency Injection**: The use case receives abstractions (ports), not concrete implementations
2. **Single Responsibility**: Only orchestrates, doesn't implement training or data loading
3. **Framework Agnostic**: No PyTorch, TensorFlow, or pandas code here

**Testing this use case:**

```python
# ✅ Test with mocks (fast, no GPU needed)
def test_train_use_case():
    mock_classifier = Mock(spec=TextClassifier)
    mock_repo = Mock(spec=DatasetRepository)
    mock_cleaner = Mock(spec=TextCleaner)
    
    use_case = TrainModelUseCase(mock_classifier, mock_repo, mock_cleaner)
    use_case.execute(config)
    
    # Verify workflow was orchestrated correctly
    mock_repo.load_train_data.assert_called_once()
    mock_classifier.train.assert_called_once()
```

#### 4.5 Predict Use Case

**File:** `application/use_cases/predict.py`

```python
class PredictUseCase:
    """Handles inference requests."""
    
    def __init__(self,
                 classifier: TextClassifier,
                 text_cleaner: TextCleaner):
        self.classifier = classifier
        self.text_cleaner = text_cleaner
    
    def execute(self, texts: List[str]) -> List[PredictionResult]:
        # Clean input
        documents = [
            Document(id=str(i), text=self.text_cleaner.clean(text))
            for i, text in enumerate(texts)
        ]
        
        # Predict (abstracted)
        labels = self.classifier.predict(documents)
        
        # Format output
        return [
            PredictionResult(text=doc.text, 
                           label=label.status.value,
                           confidence=label.confidence)
            for doc, label in zip(documents, labels)
        ]
```

**Why separate use cases?**
- Different responsibilities (training vs. inference)
- Different dependencies (training needs MLflow, inference doesn't)
- Easier to test and maintain

---

### Layer 3: Infrastructure — Technical Implementations

**Location:** `src/mh_nlp/infrastructure/`

**Purpose:** Implement the domain ports using specific technologies.

#### 4.6 PyTorch Implementation

**File:** `infrastructure/models/distilbert_classifier.py`

```python
class DistilBERTClassifier(TextClassifier):  # ← Implements domain port
    """PyTorch + HuggingFace implementation of TextClassifier.
    
    This is a DETAIL. The domain doesn't know or care about PyTorch.
    """
    
    def __init__(self, model_name: str, num_labels: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train(self, documents: List[Document], labels: List[Label], 
              validation_split: float = 0.2) -> TrainingMetrics:
        # Convert domain objects to PyTorch format
        texts = [doc.text for doc in documents]
        label_ids = [self._label_to_id(label) for label in labels]
        
        # PyTorch-specific training logic
        encodings = self.tokenizer(texts, truncation=True, padding=True, 
                                   return_tensors="pt")
        dataset = TensorDataset(encodings['input_ids'], 
                               encodings['attention_mask'],
                               torch.tensor(label_ids))
        
        # ... training loop ...
        
        return TrainingMetrics(accuracy=acc, f1=f1, loss=loss)
    
    def predict(self, documents: List[Document]) -> List[Label]:
        texts = [doc.text for doc in documents]
        encodings = self.tokenizer(texts, truncation=True, padding=True,
                                   return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**encodings.to(self.device))
            predictions = torch.argmax(outputs.logits, dim=1)
        
        # Convert back to domain objects
        return [self._id_to_label(pred.item()) for pred in predictions]
    
    def _label_to_id(self, label: Label) -> int:
        """Convert domain Label to PyTorch integer ID"""
        mapping = {
            MentalHealthStatus.NORMAL: 0,
            MentalHealthStatus.ANXIETY: 1,
            MentalHealthStatus.DEPRESSION: 2
        }
        return mapping[label.status]
    
    def _id_to_label(self, id: int) -> Label:
        """Convert PyTorch integer ID back to domain Label"""
        mapping = {0: MentalHealthStatus.NORMAL, 
                  1: MentalHealthStatus.ANXIETY,
                  2: MentalHealthStatus.DEPRESSION}
        return Label(status=mapping[id])
```

**Critical pattern:** Boundary translation

- **Inbound**: Convert domain objects (`Document`) to PyTorch tensors
- **Outbound**: Convert PyTorch outputs back to domain objects (`Label`)
- **Domain layer never sees PyTorch types**

#### 4.7 TensorFlow Implementation

**File:** `infrastructure/models/cnn_classifier.py`

```python
class CNNClassifier(TextClassifier):  # ← Same interface, different technology
    """TensorFlow/Keras implementation of TextClassifier."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_labels: int):
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.model = self._build_model(vocab_size, embedding_dim, num_labels)
    
    def _build_model(self, vocab_size, embedding_dim, num_labels):
        model = Sequential([
            Embedding(vocab_size, embedding_dim),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_labels, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def train(self, documents: List[Document], labels: List[Label],
              validation_split: float = 0.2) -> TrainingMetrics:
        # Convert domain objects to TensorFlow format
        texts = [doc.text for doc in documents]
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=200)
        
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform([l.status.value for l in labels])
        y = to_categorical(y)
        
        # TensorFlow-specific training
        history = self.model.fit(X, y, validation_split=validation_split,
                                epochs=10, batch_size=32, verbose=0)
        
        return TrainingMetrics(
            accuracy=history.history['accuracy'][-1],
            val_accuracy=history.history['val_accuracy'][-1]
        )
    
    def predict(self, documents: List[Document]) -> List[Label]:
        texts = [doc.text for doc in documents]
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=200)
        
        predictions = self.model.predict(X)
        label_ids = np.argmax(predictions, axis=1)
        
        # Convert back to domain objects
        return [self._id_to_label(id) for id in label_ids]
```

**The power of abstraction:**

```python
# ✅ Application code works with BOTH implementations
classifier = DistilBERTClassifier(...)  # or CNNClassifier(...)
use_case = TrainModelUseCase(classifier, ...)
use_case.execute(config)
# Same code, different ML framework!
```

#### 4.8 Data Repository Implementation

**File:** `infrastructure/data/kaggle_repository.py`

```python
class KaggleDatasetRepository(DatasetRepository):  # ← Implements domain port
    """CSV-based implementation of dataset storage."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
    
    def load_train_data(self) -> Tuple[List[Document], List[Label]]:
        # pandas is an INFRASTRUCTURE detail
        df = pd.read_csv(self.data_path / "Combined Data.csv")
        
        # Convert pandas DataFrame to domain objects
        documents = [
            Document(id=str(row['unique_id']), text=row['statement'])
            for _, row in df.iterrows()
        ]
        
        labels = [
            Label(status=MentalHealthStatus(row['status']))
            for _, row in df.iterrows()
        ]
        
        return documents, labels
    
    def save_processed_data(self, documents: List[Document], 
                           labels: List[Label]) -> None:
        # Convert domain objects back to pandas for storage
        data = {
            'unique_id': [doc.id for doc in documents],
            'statement': [doc.text for doc in documents],
            'status': [label.status.value for label in labels]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path / "processed.csv", index=False)
```

**Future flexibility:**

```python
# Tomorrow: Switch to PostgreSQL
class PostgresDatasetRepository(DatasetRepository):
    def load_train_data(self):
        result = self.db.execute("SELECT id, text, status FROM documents")
        documents = [Document(id=row[0], text=row[1]) for row in result]
        # ... no changes needed in application layer!
```

---

### Layer 4: Interface — Entry Points

**Location:** `src/mh_nlp/interface/`

**Purpose:** Wire everything together and expose to users.

#### 4.9 Dependency Injection Container

**File:** `interface/container.py`

```python
class DIContainer:
    """Wires together dependencies based on configuration.
    
    This is where we decide WHICH implementations to use.
    """
    
    @staticmethod
    def create_train_use_case(config_path: Path) -> TrainModelUseCase:
        # Load configuration
        config = load_config(config_path)
        
        # Create domain services
        text_cleaner = TextCleaner()
        
        # Create infrastructure (based on config)
        if config.model.type == "distilbert":
            classifier = DistilBERTClassifier(
                model_name=config.model.checkpoint,
                num_labels=config.model.num_labels
            )
        elif config.model.type == "cnn":
            classifier = CNNClassifier(
                vocab_size=config.model.vocab_size,
                embedding_dim=config.model.embedding_dim,
                num_labels=config.model.num_labels
            )
        
        dataset_repo = KaggleDatasetRepository(config.data.path)
        mlflow_client = MLflowClient(config.mlflow.tracking_uri)
        
        # Inject dependencies into use case
        return TrainModelUseCase(
            classifier=classifier,
            dataset_repo=dataset_repo,
            text_cleaner=text_cleaner,
            mlflow_client=mlflow_client
        )
```

**Why this matters:**
- ✅ Configuration drives implementation selection
- ✅ Easy to swap implementations for testing
- ✅ Clear overview of all dependencies

#### 4.10 FastAPI Interface

**File:** `interface/api/main.py`

```python
app = FastAPI(title="Mental Health NLP API")

# Wire dependencies at startup
@app.on_event("startup")
async def startup():
    config = load_config("configs/production.yaml")
    app.state.predict_use_case = DIContainer.create_predict_use_case(config)

@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict mental health status from text."""
    
    # Use case handles all business logic
    results = app.state.predict_use_case.execute(request.texts)
    
    # API layer only formats response
    return PredictResponse(
        predictions=[r.label for r in results],
        confidences=[r.confidence for r in results]
    )
```

**Separation achieved:**
- ✅ API handles HTTP concerns (routing, serialization)
- ✅ Use case handles business logic
- ✅ Infrastructure handles ML computation

---

## 5. Dependency Management

### The Inversion Principle in Action

**Traditional dependency (bad):**
```python
# ❌ High-level policy depends on low-level detail
class TrainModelUseCase:
    def execute(self):
        model = DistilBertModel()  # ← Concrete dependency
        model.train()              # ← Stuck with PyTorch forever
```

**Inverted dependency (good):**
```python
# ✅ Both depend on abstraction
class TrainModelUseCase:
    def __init__(self, classifier: TextClassifier):  # ← Abstract dependency
        self.classifier = classifier
    
    def execute(self):
        self.classifier.train()  # ← Works with any implementation
```

### Configuration-Driven Assembly

**File:** `configs/distilbert.yaml`

```yaml
model:
  type: "distilbert"
  checkpoint: "distilbert-base-uncased"
  num_labels: 3

data:
  repository_type: "kaggle_csv"
  path: "data/raw/"

training:
  batch_size: 16
  epochs: 3
  learning_rate: 2e-5
```

**File:** `configs/cnn.yaml`

```yaml
model:
  type: "cnn"
  vocab_size: 10000
  embedding_dim: 128
  num_labels: 3

data:
  repository_type: "kaggle_csv"
  path: "data/raw/"

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
```

**Usage:**

```bash
# Train with DistilBERT
python train.py --config configs/distilbert.yaml

# Train with CNN (same code!)
python train.py --config configs/cnn.yaml
```

---

## 6. Practical Benefits Demonstrated

### Benefit 1: Testability

**Domain logic tests (fast, no dependencies):**

```python
# tests/unit/domain/test_text_cleaner.py
def test_text_cleaner_removes_urls():
    cleaner = TextCleaner()
    result = cleaner.clean("Visit https://example.com for info")
    assert "https" not in result
    assert "example.com" not in result
```

**Use case tests (with mocks):**

```python
# tests/unit/application/test_train_use_case.py
def test_train_use_case_orchestration():
    # Arrange
    mock_classifier = Mock(spec=TextClassifier)
    mock_classifier.train.return_value = TrainingMetrics(accuracy=0.95)
    
    mock_repo = Mock(spec=DatasetRepository)
    mock_repo.load_train_data.return_value = (sample_docs, sample_labels)
    
    use_case = TrainModelUseCase(mock_classifier, mock_repo, TextCleaner())
    
    # Act
    result = use_case.execute(config)
    
    # Assert
    assert result.metrics.accuracy == 0.95
    mock_classifier.train.assert_called_once()
```

**Integration tests (real implementations):**

```python
# tests/integration/test_distilbert_training.py
def test_distilbert_trains_successfully():
    classifier = DistilBERTClassifier("distilbert-base-uncased", num_labels=3)
    repo = KaggleDatasetRepository(Path("data/test"))
    
    use_case = TrainModelUseCase(classifier, repo, TextCleaner())
    result = use_case.execute(config)
    
    assert result.metrics.accuracy > 0.5  # Sanity check
```

### Benefit 2: Framework Flexibility

**Scenario:** Company decides to switch from PyTorch to TensorFlow

**Required changes:**
1. Write new `TFClassifier` class implementing `TextClassifier` port
2. Update configuration file

**NO changes needed in:**
- Domain entities and services
- Application use cases
- Interface layer (API, CLI)
- Tests (except implementation-specific integration tests)

### Benefit 3: Team Collaboration

**Data Scientists:**
- Work in notebooks
- Experiment freely
- Import only from `domain/` and `application/`

**ML Engineers:**
- Implement new models in `infrastructure/models/`
- Optimize training loops
- Add new features to existing ports

**Backend Engineers:**
- Build APIs in `interface/`
- Don't touch ML code
- Use use cases as black boxes

**No stepping on each other's toes!**

### Benefit 4: Incremental Migration

**Today:** PyTorch DistilBERT in production

**Tomorrow:** Want to A/B test with TensorFlow CNN

**Implementation:**

```python
# No code changes, just configuration
class EnsembleClassifier(TextClassifier):
    def __init__(self, classifiers: List[TextClassifier]):
        self.classifiers = classifiers
    
    def predict(self, documents: List[Document]) -> List[Label]:
        # Average predictions from multiple models
        all_predictions = [clf.predict(documents) for clf in self.classifiers]
        return self._aggregate(all_predictions)

# In container
distilbert = DistilBERTClassifier(...)
cnn = CNNClassifier(...)
ensemble = EnsembleClassifier([distilbert, cnn])

use_case = PredictUseCase(ensemble, text_cleaner)
```

---

## 7. Common Pitfalls Avoided

### Pitfall 1: Leaking Infrastructure into Domain

**❌ Wrong:**

```python
# domain/entities/document.py
import torch  # ← Infrastructure leaking into domain!

class Document:
    def to_tensor(self) -> torch.Tensor:  # ← Domain shouldn't know about tensors
        pass
```

**✅ Correct:**

```python
# domain/entities/document.py
# No framework imports!

class Document:
    id: str
    text: str
    # Domain objects are pure data structures
```

### Pitfall 2: Use Cases Doing Too Much

**❌ Wrong:**

```python
class TrainModelUseCase:
    def execute(self):
        # Loading data
        df = pd.read_csv("data.csv")
        
        # Cleaning text
        df['text'] = df['text'].apply(lambda x: x.lower())
        
        # Building model
        model = DistilBertModel(...)
        
        # Training loop
        for epoch in range(10):
            # ... 50 lines of training code ...
```

**✅ Correct:**

```python
class TrainModelUseCase:
    def execute(self):
        # Delegate to specialized components
        documents, labels = self.dataset_repo.load_train_data()
        cleaned_docs = [self._clean(doc) for doc in documents]
        metrics = self.classifier.train(cleaned_docs, labels)
        return metrics
```

### Pitfall 3: Concrete Dependencies in Constructors

**❌ Wrong:**

```python
class PredictUseCase:
    def __init__(self):
        # Hardcoded concrete implementations
        self.classifier = DistilBERTClassifier()  # ← Tightly coupled!
        self.text_cleaner = TextCleaner()
        self.config = load_config("config.yaml")
```

**Problems:**
- Cannot swap implementations without modifying code
- Impossible to test with mocks
- Violates Dependency Inversion Principle
- Creates hidden dependencies

**✅ Correct:**

```python
class PredictUseCase:
    def __init__(self, 
                 classifier: TextClassifier,      # ← Abstract dependency
                 text_cleaner: TextCleaner):      # ← Injected
        self.classifier = classifier
        self.text_cleaner = text_cleaner
```

**Benefits:**
- Easy to inject different implementations
- Testable with mocks
- Dependencies are explicit and visible
- Follows Dependency Inversion Principle

**Example usage:**

```python
# Production
classifier = DistilBERTClassifier(...)
use_case = PredictUseCase(classifier, TextCleaner())

# Testing
mock_classifier = Mock(spec=TextClassifier)
use_case = PredictUseCase(mock_classifier, TextCleaner())

# A/B Testing
ensemble_classifier = EnsembleClassifier([model1, model2])
use_case = PredictUseCase(ensemble_classifier, TextCleaner())
```

### Pitfall 4: Testing Through the API

**❌ Wrong:**

```python
def test_prediction_accuracy():
    # Testing everything through HTTP layer
    response = requests.post("http://localhost:8000/predict", 
                            json={"texts": ["I feel anxious"]})
    assert response.status_code == 200
    # Slow, fragile, tests too much at once
```

**Problems:**
- Requires server to be running
- Slow (HTTP overhead, full stack initialization)
- Tests API, business logic, and infrastructure simultaneously
- Hard to isolate failures
- Difficult to test edge cases

**✅ Correct:**

```python
# Unit test (use case level)
def test_prediction_logic():
    mock_classifier = Mock(spec=TextClassifier)
    mock_classifier.predict.return_value = [Label(status=MentalHealthStatus.ANXIETY)]
    
    use_case = PredictUseCase(mock_classifier, TextCleaner())
    result = use_case.execute(["I feel anxious"])
    
    assert result[0].label == "Anxiety"
    # Fast, focused, reliable

# Integration test (infrastructure level)
def test_distilbert_prediction():
    classifier = DistilBERTClassifier.load("models/trained_model.pt")
    documents = [Document(id="1", text="I feel anxious")]
    
    labels = classifier.predict(documents)
    
    assert labels[0].status == MentalHealthStatus.ANXIETY
    # Tests real ML model without HTTP layer

# E2E test (full stack) - only a few critical paths
def test_api_endpoint():
    response = client.post("/predict", json={"texts": ["I feel anxious"]})
    assert response.json()["predictions"][0] == "Anxiety"
    # Validates complete integration
```

**Testing pyramid applied:**

```
        /\
       /  \     E2E Tests (few)
      /____\    - Full API integration
     /      \   - Critical user paths
    /________\  
   /          \ Integration Tests (moderate)
  /____________\ - Real implementations
 /              \ - Component interactions
/________________\ 
    Unit Tests (many)
    - Use cases with mocks
    - Domain logic
    - Fast & isolated
```

### Pitfall 5: Mixing Business Logic with Framework Code

**❌ Wrong:**

```python
# infrastructure/models/distilbert_classifier.py
class DistilBERTClassifier:
    def train(self, df: pd.DataFrame):  # ← pandas in interface
        # Text cleaning logic mixed with PyTorch
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].str.replace('[^a-z]', '')
        
        # Model training
        for epoch in range(self.epochs):
            for batch in dataloader:
                # ... PyTorch code ...
```

**Problems:**
- Business rules (text cleaning) hidden in infrastructure
- Cannot reuse cleaning logic
- Cannot change cleaning without touching ML code
- Violates Single Responsibility Principle

**✅ Correct:**

```python
# domain/services/text_cleaner.py
class TextCleaner:
    """Pure business logic - no framework dependencies"""
    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        return text.strip()

# infrastructure/models/distilbert_classifier.py
class DistilBERTClassifier(TextClassifier):
    def train(self, documents: List[Document], labels: List[Label]):
        # Documents are already cleaned by use case
        # This class only handles PyTorch-specific logic
        texts = [doc.text for doc in documents]
        encodings = self.tokenizer(texts, ...)
        
        for epoch in range(self.epochs):
            # ... PyTorch training loop ...
```

### Pitfall 6: Configuration Scattered Everywhere

**❌ Wrong:**

```python
# Hardcoded values scattered across files
class DistilBERTClassifier:
    def __init__(self):
        self.learning_rate = 2e-5  # ← Here
        self.model = "distilbert-base-uncased"  # ← Here

class TrainModelUseCase:
    def execute(self):
        self.batch_size = 16  # ← Here
        self.epochs = 3  # ← Here
```

**Problems:**
- Hard to find and change configuration
- Different values in different environments
- No single source of truth

**✅ Correct:**

```yaml
# configs/distilbert.yaml - Single source of truth
model:
  type: "distilbert"
  checkpoint: "distilbert-base-uncased"
  num_labels: 3

training:
  learning_rate: 2e-5
  batch_size: 16
  epochs: 3
  optimizer: "adamw"
```

```python
# Code reads from configuration
class DIContainer:
    @staticmethod
    def create_classifier(config: Config) -> TextClassifier:
        if config.model.type == "distilbert":
            return DistilBERTClassifier(
                model_name=config.model.checkpoint,
                num_labels=config.model.num_labels,
                learning_rate=config.training.learning_rate
            )
```

### Pitfall 7: God Objects (Classes That Do Everything)

**❌ Wrong:**

```python
class MentalHealthModel:
    """Does everything - loads data, cleans, trains, predicts, saves"""
    
    def __init__(self):
        pass
    
    def load_data(self, path):
        # 50 lines of data loading
        pass
    
    def clean_text(self, text):
        # 30 lines of text processing
        pass
    
    def build_model(self):
        # 40 lines of model architecture
        pass
    
    def train(self):
        # 100 lines of training logic
        pass
    
    def predict(self, text):
        # 20 lines of inference
        pass
    
    def save_model(self, path):
        # 15 lines of serialization
        pass
    
    # ... 500 lines total
```

**Problems:**
- Violates Single Responsibility Principle
- Hard to test (must mock everything)
- Hard to understand
- Hard to maintain
- Hard to extend

**✅ Correct:**

```python
# Each class has ONE responsibility

# Domain
class Document:
    """Represents a document - that's it"""
    pass

class TextCleaner:
    """Cleans text - that's it"""
    def clean(self, text: str) -> str:
        pass

# Infrastructure
class DistilBERTClassifier(TextClassifier):
    """Implements PyTorch classifier - that's it"""
    def train(self, documents, labels):
        pass
    
    def predict(self, documents):
        pass

class KaggleDatasetRepository(DatasetRepository):
    """Handles data loading/saving - that's it"""
    def load_train_data(self):
        pass

# Application
class TrainModelUseCase:
    """Orchestrates training workflow - that's it"""
    def execute(self, config):
        documents = self.repo.load_train_data()
        cleaned = [self.cleaner.clean(d.text) for d in documents]
        return self.classifier.train(cleaned)
```

---

## Summary: Architecture Decision Records

Here's a summary of key architectural decisions made in this project:

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **Clean Architecture** | Separation of concerns, testability, flexibility | More initial setup complexity |
| **Dependency Injection** | Loose coupling, easy testing | Requires wiring/container |
| **Port/Adapter Pattern** | Framework independence | More abstractions to learn |
| **Configuration-driven** | Environment flexibility, no code changes | YAML complexity |
| **Domain-first design** | Business logic clarity | Requires domain understanding |
| **Multiple implementations** | Demonstrates flexibility | Maintains multiple codebases |
| **Use case orchestration** | Clear workflows, single responsibility | More files/classes |

---

## Key Takeaways

### What Clean Architecture IS:
✅ A way to organize code by **business concerns**, not technical frameworks  
✅ An investment in **long-term maintainability** over short-term speed  
✅ A system of **rules and boundaries** that prevent common mistakes  
✅ A **communication tool** that makes code readable by domain experts  

### What Clean Architecture is NOT:
❌ Not just "putting code in folders"  
❌ Not over-engineering for its own sake  
❌ Not a rigid dogma (adapt to your needs)  
❌ Not necessary for every project (use judgment)  

### When to Use Clean Architecture:

**Use it when:**
- Project will evolve over months/years
- Multiple team members will work on it
- Requirements are likely to change
- You want to showcase professional practices
- System needs high testability

**Skip it when:**
- One-off analysis or prototype
- Solo project with clear, fixed scope
- Exploration/research phase
- Time constraints are extreme

---

## Next Steps for This Project

Based on this architecture, here are natural extensions:

1. **Add Model Registry**
   - Store models in MLflow Model Registry
   - Version control for production models
   - A/B testing framework

2. **Implement Monitoring**
   - Prediction logging
   - Drift detection
   - Performance metrics dashboard

3. **Extend to 7 Classes**
   - Currently uses 3 classes (Normal/Anxiety/Depression)
   - Add remaining categories
   - Handle class imbalance

4. **Add Explainability**
   - SHAP values for predictions
   - Attention visualization
   - Confidence scoring

5. **Production Deployment**
   - Kubernetes manifests
   - Load balancing
   - Auto-scaling
   - CI/CD pipeline

---

**This document demonstrates:**
- ✅ Deep understanding of software architecture principles
- ✅ Ability to explain complex concepts pedagogically
- ✅ Professional engineering practices
- ✅ Attention to maintainability and team collaboration

---

**Author:** Manda Surel  
**Contact:** mandasurel@yahoo.com  
**Project Repository:** [github.com/Manda404/mental-health-nlp]