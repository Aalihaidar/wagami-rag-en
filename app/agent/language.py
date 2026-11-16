# cspell:disable -- this module is lists of words from other languages
"""A cheap, deterministic check for a message that is clearly not English (rule C-28).

The demo's menu, FAQ and answers are English only, and a message in another language gets a
fixed reply (R-18). The understanding call can tell languages apart, but it costs about 4,500
tokens, so the obvious cases are caught here first, in code, with no model call:

* **Another script.** A message whose letters are mostly not Latin (Arabic, Cyrillic, Chinese,
  Japanese, Korean, Hebrew, Hindi, Thai, Greek, ...) is not English whatever it says.
* **Function words of a Latin-script language.** French, German, Spanish, Italian, Portuguese,
  Dutch, Turkish, Indonesian, Polish, and Arabic written in Latin letters ("marhaba", "shukran")
  share the alphabet with English, so the evidence is words: each language has a short list of
  very common words that are not English words ("bonjour", "vous", "gracias", "ich", "nicht").
  A message counts as that language when it has at least two of them and more than it has English
  function words. One greeting alone ("hola", "merci", "danke") is enough.

Every list is deliberately short and leaves out a word that is also English ("die", "hat", "no",
"me", "sin", "come", "per", "con", "prix", "carte"), and none of them is a dish or ingredient
name, so an English question about the menu, including one that names a dish from another
language ("do you have yasai gyoza?"), is never caught. A message this check does not catch is
not necessarily English: the understanding call (U-23) decides the rest.
"""

import re
import unicodedata
from collections import Counter

# Runs of letters. Apostrophes, hyphens and digits split a word, so "qu'est-ce" is qu / est / ce.
_WORD = re.compile(r"[^\W\d_]+")


def _words(text: str) -> frozenset[str]:
    """The whitespace-separated words of `text`, as a set."""
    return frozenset(text.split())


# Hits a message needs in one language's list, and the English function words it may not match.
MIN_FOREIGN_WORDS = 2

ENGLISH_WORDS = _words(
    """
    the a an and or but if then than so as is are was were be been am do does did have has had
    i you your he she it we they me my our their his her its this that these those what which who
    whom how when where why can could will would shall should may might must not no yes please
    thanks thank hello hi hey with without for of in on at to from by about any some there here
    too very much many more most less just also only tell show give want need like get got there
    """
)

# Words (lower case, whole words) that mark a language and are not English words.
_GREETINGS: dict[str, frozenset[str]] = {
    "french": frozenset({"bonjour", "bonsoir", "merci"}),
    "german": frozenset({"hallo", "danke", "bitte", "servus", "moin"}),
    "spanish": frozenset({"hola", "gracias", "buenas", "buenos"}),
    "italian": frozenset({"grazie", "buongiorno", "buonasera", "prego"}),
    "portuguese": frozenset({"olá", "ola", "obrigado", "obrigada"}),
    "dutch": frozenset({"hallo", "bedankt", "alstublieft", "dankjewel"}),
    "turkish": frozenset({"merhaba", "selam", "teşekkürler", "tesekkurler", "lütfen"}),
    "indonesian": frozenset({"terima", "kasih", "tolong"}),
    "polish": frozenset({"cześć", "czesc", "dziękuję", "dziekuje", "proszę", "prosze"}),
    "arabic (latin letters)": frozenset(
        {"marhaba", "marhaban", "ahlan", "ahlen", "salam", "salaam", "shukran", "habibi",
         "inshallah", "mashallah", "yalla", "assalamu", "alaikum", "alaykum", "aleykum"}
    ),
}  # fmt: skip

_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "french": _words(
        """
        je tu vous nous est sont les des une avec dans que qui quoi quel quelle quels quelles
        avez avons veux voudrais voulez oui beaucoup plat plats boisson boissons allergie
        allergies combien il elle pas
        """
    ),
    "german": _words(
        """
        ich du sie wir ihr ist sind nicht und der das ein eine einen mit ohne für gibt haben
        habt möchte mochte welche welches wie viel kostet kosten gericht gerichte speisekarte
        getränke getranke guten
        """
    ),
    "spanish": _words(
        """
        tú tu usted ustedes es están estan los las una para pero muy por quiero quisiera
        tienen tiene tienes hay cuánto cuanto cuesta precio carta menú comida bebida bebidas
        alergia alergias qué
        """
    ),
    "italian": _words(
        """
        tu noi voi il gli della delle dei una sono che cosa anche più vorrei avete quanto
        costa prezzo menù piatti piatto bevande glutine allergia allergie senza
        """
    ),
    "portuguese": _words(
        """
        eu você voce vocês voces nós uma são sao tem têm não nao quero queria quanto custa
        preço cardápio cardapio pratos prato bebidas alergia alergias
        """
    ),
    "dutch": _words(
        """
        ik jij wij jullie het een niet welke hoeveel kost heeft hebben zonder voor graag
        menukaart gerechten drankjes
        """
    ),
    "turkish": _words(
        """
        nedir kaç kac kadar fiyat fiyatı menü yemek içecek icecek bir ve için icin ile bu
        evet hayır hayir yok
        """
    ),
    "indonesian": _words(
        """
        saya anda kamu yang dengan tidak tanpa bisa mau ingin apa ada berapa harga makanan
        minuman
        """
    ),
    "polish": _words(
        """
        jest czy nie jakie ile kosztuje dzień dzien dobry jakie proszę prosze
        """
    ),
    "arabic (latin letters)": _words(
        """
        shu kifak kifik ana enta enti ayz ayza badi mafi
        """
    ),
}

_LANGUAGE_WORDS: dict[str, frozenset[str]] = {
    language: _FUNCTION_WORDS[language] | _GREETINGS[language] for language in _GREETINGS
}
_ALL_GREETINGS = frozenset().union(*_GREETINGS.values())


def _letter_script(letter: str) -> str:
    """The script a letter belongs to, as the first word of its Unicode name ("LATIN",
    "ARABIC", "CYRILLIC", "CJK", "HIRAGANA", "HANGUL", ...)."""
    name = unicodedata.name(letter, "")
    return name.split(" ", 1)[0] if name else "UNKNOWN"


def detect_non_english(text: str) -> str | None:
    """Why `text` is clearly not English, or None when it is, or when this check cannot tell.

    The answer is a label for tests and logs: the script ("arabic", "cyrillic", "cjk", ...) or
    the language ("french", "german", ...). See the module docstring for what counts.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None  # numbers, emoji or punctuation only: not a language

    scripts = Counter(_letter_script(ch) for ch in letters)
    latin = scripts.pop("LATIN", 0)
    if scripts and sum(scripts.values()) > latin:
        return scripts.most_common(1)[0][0].lower()

    words = _WORD.findall(text.lower())
    if len(words) == 1 and words[0] in _ALL_GREETINGS:
        language = next(lang for lang, greetings in _GREETINGS.items() if words[0] in greetings)
        return language
    english = sum(1 for word in words if word in ENGLISH_WORDS)
    hits = {
        language: sum(1 for word in words if word in vocabulary)
        for language, vocabulary in _LANGUAGE_WORDS.items()
    }
    language, best = max(hits.items(), key=lambda item: item[1])
    if best >= MIN_FOREIGN_WORDS and best > english:
        return language
    return None
