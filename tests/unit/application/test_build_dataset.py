from mh_nlp.application.use_cases.build_dataset import BuildDatasetUseCase
from mh_nlp.domain.services.text_cleaner import TextCleaner

def test_build_dataset_filters_and_cleans():
    cleaner = TextCleaner()
    use_case = BuildDatasetUseCase(
        text_cleaner=cleaner,
        selected_classes=["Normal", "Anxiety"],
        label_mapping={"Normal": 0, "Anxiety": 1}
    )

    texts = ["Hello WORLD!!!", "   ", "I am anxious"]
    labels = ["Normal", "Normal", "Anxiety"]

    dataset = use_case.execute(texts, labels)

    assert len(dataset.documents) == 2
    assert dataset.labels == [0, 1]