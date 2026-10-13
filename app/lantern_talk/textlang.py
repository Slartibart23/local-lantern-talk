"""Tiny language guesser for typed messages (German, English, Spanish, French).

Spoken input gets its language from Whisper; typed input has none. These are
exactly the languages with built-in voices, so a function-word count is enough
to pick the right voice and reply language. Returns ``None`` when unsure.
"""

from __future__ import annotations

import re

_WORDS: dict[str, frozenset[str]] = {
    "de": frozenset("""
        der die das und ist ich du er sie es wir ihr nicht ein eine einen mit auf für von zu
        den dem des sind bin bist hast habe hat was wie wer wo warum wann auch noch aber oder
        wenn dann mir mich dir dich uns euch kann kannst bitte danke hallo ja nein gut sehr
        mal schon jetzt heute morgen gerne erzähl erzähle sag sage mein meine dein deine
    """.split()),
    "en": frozenset("""
        the and is are am i you he she it we they not a an with on for of to in that this
        was were be been have has do does did what how who where why when also but or if
        then me my your our can could would please thanks thank hello hi yes no good very
        just now today tomorrow tell about there here
    """.split()),
    "es": frozenset("""
        el la los las y es son soy eres yo tú él ella nosotros no un una con en por para de
        del al que qué cómo quién dónde por qué cuándo también pero o si me te mi tu muy
        bien hola gracias sí está estás estoy hoy mañana dime cuéntame puedes
    """.split()),
    "fr": frozenset("""
        le la les et est sont suis es je tu il elle nous vous ils elles ne pas un une avec
        sur pour de du des au aux que qui quoi comment où pourquoi quand aussi mais ou si
        me te moi toi mon ma mes ton ta très bien bonjour merci salut oui non c'est
        aujourd'hui demain dis peux
    """.split()),
}

_LETTERS: dict[str, str] = {"de": "äöüß", "es": "ñ¿¡áíóú", "fr": "çœàèêëîïôûù"}

_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def guess_language(text: str) -> str | None:
    """Best guess among de / en / es / fr, or ``None`` if the text is too ambiguous."""
    lowered = text.lower()
    tokens = _TOKEN.findall(lowered)
    if not tokens:
        return None
    scores = {lang: sum(t in words for t in tokens) for lang, words in _WORDS.items()}
    for lang, letters in _LETTERS.items():
        scores[lang] += 2 * sum(lowered.count(ch) for ch in letters)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top == 0 or top == second:
        return None
    if len(tokens) >= 4 and top < 2:
        return None
    return best
