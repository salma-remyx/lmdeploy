# Copyright (c) OpenMMLab. All rights reserved.
"""Route requests to per-language LoRA adapters by detecting prompt language.

Adapted from "Language-Conditional Dequantization: Recovering What Quantization
Steals from Non-English Languages" (arXiv:2608.11786). The paper's contribution
is a set of *post-hoc, per-language rank-2 LoRA corrections* attached to the
linear layers of an already-quantized model, recovering most of the
quantization-induced quality loss that non-English languages suffer. lmdeploy
already has the execution half of that recipe: the MLoRA path loads multiple
named adapters and ``fused_lora`` selects one per row of the batch. What was
missing is the *selection* half -- a request carrying no explicit
``adapter_name`` always falls back to the base model, so a stack of trained
per-language corrections can never be reached by a plain multilingual request.

This module fills that gap with a dependency-free language identifier over
Unicode script ranges plus a small stop-word table for the Latin- and
Cyrillic-script languages that script alone cannot separate. It is deliberately
parameter-free: the paper's corrections are trained offline; here we only have
to pick the right one at request time.

Usage::

    router = LangAdapterRouter({'fr': 'path/to/fr_lora'})
    router.route('Bonjour, comment allez-vous ?')  # -> 'fr'
    router.route('Hello there')                    # -> None (base model)
"""

import os
import re
import unicodedata

# Script ranges as (start, end, language or family). Han is mapped to 'zh'
# only as a default: see the kana tiebreak in `detect_language`.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0530, 0x058F, 'hy'),  # Armenian
    (0x0590, 0x05FF, 'he'),  # Hebrew
    (0x0600, 0x06FF, 'ar'),  # Arabic
    (0x0900, 0x097F, 'hi'),  # Devanagari
    (0x0980, 0x09FF, 'bn'),  # Bengali
    (0x0A00, 0x0A7F, 'pa'),  # Gurmukhi
    (0x0A80, 0x0AFF, 'gu'),  # Gujarati
    (0x0B00, 0x0B7F, 'or'),  # Oriya
    (0x0B80, 0x0BFF, 'ta'),  # Tamil
    (0x0C00, 0x0C7F, 'te'),  # Telugu
    (0x0C80, 0x0CFF, 'kn'),  # Kannada
    (0x0D00, 0x0D7F, 'ml'),  # Malayalam
    (0x0D80, 0x0DFF, 'si'),  # Sinhala
    (0x0E00, 0x0E7F, 'th'),  # Thai
    (0x0E80, 0x0EFF, 'lo'),  # Lao
    (0x0F00, 0x0FFF, 'bo'),  # Tibetan
    (0x10A0, 0x10FF, 'ka'),  # Georgian
    (0x1100, 0x11FF, 'ko'),  # Hangul Jamo
    (0x3130, 0x318F, 'ko'),  # Hangul Compatibility Jamo
    (0xAC00, 0xD7AF, 'ko'),  # Hangul Syllables
    (0x3040, 0x30FF, 'ja'),  # Hiragana / Katakana
    (0x31F0, 0x31FF, 'ja'),
    (0xFF66, 0xFF9D, 'ja'),  # Halfwidth Katakana
    (0x3400, 0x4DBF, 'zh'),  # CJK Extension A
    (0x4E00, 0x9FFF, 'zh'),  # CJK Unified Ideographs
)

# Scripts shared by many languages: script alone is ambiguous, so detection
# falls through to a stop-word vote among the languages listed below.
_CYRILLIC_RANGE = (0x0400, 0x04FF)
_CYRILLIC_LANGUAGES = ('ru', 'uk', 'bg')
_LATIN_LANGUAGES = ('de', 'en', 'es', 'fr', 'id', 'it', 'nl', 'pl', 'pt', 'sw', 'tr', 'vi')

# Distinctive stop words keyed by ISO 639-1 code. Matched as whole,
# case-folded tokens.
_STOPWORDS: dict[str, set[str]] = {
    'en': {'the', 'and', 'is', 'of', 'to', 'in', 'that', 'it', 'for', 'with'},
    'de': {'der', 'die', 'und', 'das', 'ist', 'nicht', 'mit', 'für', 'ein', 'eine'},
    'es': {'el', 'la', 'que', 'de', 'los', 'las', 'por', 'para', 'una', 'con'},
    'fr': {'le', 'la', 'les', 'de', 'et', 'est', 'que', 'pour', 'dans', 'une'},
    'it': {'il', 'che', 'di', 'per', 'con', 'una', 'non', 'sono', 'gli', 'della'},
    'pt': {'que', 'não', 'para', 'uma', 'com', 'dos', 'como', 'nós', 'você', 'esse'},
    'nl': {'het', 'een', 'van', 'en', 'dat', 'niet', 'met', 'voor', 'zijn', 'maar'},
    'pl': {'nie', 'jest', 'się', 'że', 'jak', 'ale', 'przez', 'przy', 'tylko', 'oraz'},
    'tr': {'bir', 've', 'bu', 'için', 'ile', 'olarak', 'değil', 'çok', 'daha', 'gibi'},
    'id': {'dan', 'yang', 'untuk', 'dengan', 'tidak', 'adalah', 'ini', 'dari', 'akan', 'itu'},
    'vi': {'của', 'và', 'là', 'các', 'được', 'không', 'cho', 'trong', 'một', 'người'},
    'sw': {'na', 'ya', 'wa', 'kwa', 'ni', 'katika', 'hii', 'kuwa', 'maana', 'sana'},
    'ru': {'и', 'в', 'не', 'на', 'что', 'с', 'по', 'для', 'это', 'как'},
    'uk': {'і', 'в', 'не', 'на', 'що', 'з', 'по', 'для', 'це', 'як'},
    'bg': {'и', 'в', 'не', 'на', 'че', 'с', 'за', 'това', 'кат', 'който'},
}

_TOKEN_RE = re.compile(r'[^\W\d_]+', re.UNICODE)

# Minimum fraction of letters that must fall in the winning script, and the
# minimum number of scored letters, before a detection is trusted.
_MIN_SCRIPT_FRACTION = 0.3
_MIN_LETTERS = 4


def _script_name(char: str) -> str | None:
    """Return the language or script bucket a letter belongs to."""
    code = ord(char)
    for start, end, lang in _SCRIPT_RANGES:
        if start <= code <= end:
            return lang
    if _CYRILLIC_RANGE[0] <= code <= _CYRILLIC_RANGE[1]:
        return 'cyrillic'
    if char.isalpha():
        return 'latin'
    return None


def _stopword_vote(text: str, languages: tuple[str, ...]) -> str | None:
    """Pick the language with the most distinctive stop words."""
    tokens = {token.casefold() for token in _TOKEN_RE.findall(text)}
    if not tokens:
        return None
    best_lang = None
    best_hits = 0
    for lang in languages:
        hits = len(tokens & _STOPWORDS[lang])
        if hits > best_hits:
            best_lang = lang
            best_hits = hits
    return best_lang


def detect_language(text: str) -> str | None:
    """Detect the dominant language of ``text``.

    Returns an ISO 639-1 code, or None when the text is too short or too
    ambiguous to call. Detection is script-first: a non-Latin, non-Cyrillic
    script with a clear majority wins outright. Han resolves to Japanese when
    any kana is present (Japanese prose is Han-heavy, so Han alone would
    misroute it to Chinese) and to Chinese otherwise. Latin and Cyrillic fall
    through to a stop-word vote.
    """
    if not text:
        return None

    normalized = unicodedata.normalize('NFKC', text)
    votes: dict[str, int] = {}
    letters = 0
    for char in normalized:
        script = _script_name(char)
        if script is None:
            continue
        letters += 1
        votes[script] = votes.get(script, 0) + 1
    if letters < _MIN_LETTERS:
        return None

    script, count = max(votes.items(), key=lambda item: (item[1], item[0]))
    if count / letters < _MIN_SCRIPT_FRACTION:
        return None

    if script == 'zh':
        return 'ja' if votes.get('ja') else 'zh'
    if script == 'latin':
        return _stopword_vote(normalized, _LATIN_LANGUAGES)
    if script == 'cyrillic':
        return _stopword_vote(normalized, _CYRILLIC_LANGUAGES)
    return script


_ENV_PREFIX = 'LMDEPLOY_LANG_ADAPTER_'
_MAX_PROMPT_CHARS = 4096
# Adapter names routable by language: exactly two ASCII lowercase letters,
# i.e. an ISO 639-1 code.
_LANG_CODE_RE = re.compile(r'^[a-z]{2}$')


class LangAdapterRouter:
    """Select a per-language LoRA adapter name for a request prompt.

    Args:
        adapters: the engine's ``adapters`` config mapping adapter name to
            adapter path. Only entries named after an ISO 639-1 language code
            (``fr``, ``ja``, ...) are routable; anything else keeps its
            explicit-selection-only behaviour.
    """

    def __init__(self, adapters: dict[str, str] | None):
        adapters = adapters or {}
        self._routes = {name: name for name in adapters if _LANG_CODE_RE.match(name)}
        self.enabled = bool(self._routes)

    def route(self, prompt: str | None) -> str | None:
        """Return the adapter name for ``prompt``, or None for the base model.

        Never called for a request that already names an adapter: the caller's
        explicit ``adapter_name`` wins over detection, which keeps the
        OpenAI-style "model = adapter name" selection working unchanged.

        Detection reads the *head* of the prompt, not the tail. Prompt history
        only grows, so a session resolves to the same adapter on every turn --
        which matters because the prefix cache namespaces blocks by adapter
        name, and a router that flipped languages mid-session would forfeit
        cache reuse for that session.
        """
        if not self.enabled or not prompt:
            return None
        lang = detect_language(prompt[:_MAX_PROMPT_CHARS])
        if lang is None:
            return None
        return self._routes.get(lang)

    def routed_adapters(self) -> dict[str, str]:
        """Return the routable subset of the adapters config."""
        return dict(self._routes)


def build_lang_adapter_router(adapters: dict[str, str] | None) -> LangAdapterRouter:
    """Build a router, or a disabled one when routing is turned off.

    Routing is opt-out via ``LMDEPLOY_LANG_ADAPTER_DISABLE=1`` so an existing
    multilingual deployment that pins a single adapter is unaffected.
    """
    if os.getenv(_ENV_PREFIX + 'DISABLE', '0') == '1':
        return LangAdapterRouter(None)
    return LangAdapterRouter(adapters)
