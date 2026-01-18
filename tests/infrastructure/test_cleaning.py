import pytest
import pandas as pd
from mh_nlp.domain.services.text_cleaner import TextCleaner

def test_cleaner_should_remove_emojis():
    cleaner = TextCleaner()
    text_with_emoji = "I feel sad 😭"
    cleaned = cleaner.clean(text_with_emoji)
    
    # On vérifie que l'emoji a disparu et que le texte reste
    assert "😭" not in cleaned
    assert "sad" in cleaned


def test_cleaner_should_handle_dirty_fixtures():
    cleaner = TextCleaner()
    # Test sur la ligne 10 (balise HTML)
    dirty_text = "I <br> cannot stop shaking"
    cleaned = cleaner.clean(dirty_text)
    
    assert "<br>" not in cleaned
    assert cleaned == "i cannot stop shaking" # Si ton cleaner lowercase tout