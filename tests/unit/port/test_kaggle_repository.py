from mh_nlp.infrastructure.data.kaggle_repository import KaggleDatasetRepository

"""
def test_kaggle_repository_load(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("statement,status\nhello,Normal\n")

    repo = KaggleDatasetRepository(str(csv))
    data = repo.load()

    assert data["texts"] == ["hello"]
    assert data["labels"] == ["Normal"]
"""