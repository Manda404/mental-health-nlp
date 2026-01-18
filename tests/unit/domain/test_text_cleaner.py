from mh_nlp.domain.services.text_cleaner import TextCleaner
from mh_nlp.domain.entities.document import Document

def test_clean_text_removes_urls_and_symbols():
    cleaner = TextCleaner()
    text = "Hello!!! Visit https://example.com #test"
    doc = Document(text)
    cleaned = cleaner.clean(doc)

    assert "http" not in cleaned
    assert "#" not in cleaned