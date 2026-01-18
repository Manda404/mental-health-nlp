from mh_nlp.interface.container import build_system

def test_build_system_runs():
    classifier, tokenizer, config = build_system("configs/model.yaml")

    assert classifier is not None
    assert tokenizer is not None
    assert "model" in config