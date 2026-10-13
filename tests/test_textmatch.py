from lantern_talk.textmatch import normalize_words, word_similarity


def test_normalize() -> None:
    assert normalize_words("Ça va? Grüß dich!") == ["ca", "va", "gruß", "dich"]


def test_similar_transcripts_match() -> None:
    ref = "Hi, and welcome! How can I help you today? I'm happy to answer any question you have."
    heard = "Hi and welcome. How can I help you today? I'm happy to answer any question you have."
    assert word_similarity(ref, heard) > 0.95


def test_hallucination_does_not_match() -> None:
    ref = "Hi, and welcome! How can I help you today? I'm happy to answer any question you have."
    assert word_similarity(ref, "Thank you.") < 0.3


def test_empty() -> None:
    assert word_similarity("", "") == 1.0
    assert word_similarity("hello", "") == 0.0
