from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import pronouncing

from medterm.models import Pronunciation
from medterm.normalization import normalize_term

ARPABET_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


def arpabet_to_ipa(value: str) -> str:
    phones = []
    for token in value.split():
        base = re.sub(r"\d", "", token)
        phones.append(ARPABET_TO_IPA.get(base, base.casefold()))
    return "".join(phones)


class PronunciationEngine:
    """Tiered pronunciation: curated, English CMUdict, Epitran, deterministic fallback."""

    def __init__(self, curated: dict[str, list[str]] | None = None) -> None:
        self.curated = {normalize_term(key): value for key, value in (curated or {}).items()}
        self._epitran: dict[str, Any] = {}

    def generate(self, text: str, language: str = "en") -> list[Pronunciation]:
        normalized = normalize_term(text, language)
        if not normalized:
            return []
        curated = self.curated.get(normalized)
        if curated:
            return [
                Pronunciation(value=value, alphabet="ipa", source="curated") for value in curated
            ]

        if language == "en":
            variants = self._cmudict_phrase(normalized)
            if variants:
                return [
                    Pronunciation(value=value, alphabet="ipa", source="cmudict")
                    for value in variants[:3]
                ]

        epitran_value = self._epitran_transliterate(normalized, language)
        if epitran_value:
            return [Pronunciation(value=epitran_value, alphabet="ipa", source="epitran")]

        fallback = self._fallback(normalized)
        return (
            [Pronunciation(value=fallback, alphabet="fallback", source="fallback")]
            if fallback
            else []
        )

    @staticmethod
    @lru_cache(maxsize=4096)
    def _cmudict_phrase(text: str) -> tuple[str, ...]:
        word_variants: list[list[str]] = []
        for word in text.split():
            pronunciations = pronouncing.phones_for_word(word)
            if not pronunciations:
                return ()
            word_variants.append([arpabet_to_ipa(value) for value in pronunciations[:2]])
        results = [""]
        for variants in word_variants:
            results = [prefix + variant for prefix in results for variant in variants]
        return tuple(dict.fromkeys(results))

    def _epitran_transliterate(self, text: str, language: str) -> str | None:
        codes = {
            "es": "spa-Latn",
            "de": "deu-Latn",
            "fr": "fra-Latn",
            "it": "ita-Latn",
            "pt": "por-Latn",
            "ru": "rus-Cyrl",
            "ar": "ara-Arab",
            "hi": "hin-Deva",
        }
        code = codes.get(language)
        if code is None:
            return None
        try:
            import epitran  # type: ignore[import-not-found]

            engine = self._epitran.setdefault(code, epitran.Epitran(code))
            return engine.transliterate(text) or None
        except (ImportError, KeyError, ValueError, OSError):
            return None

    @staticmethod
    def _fallback(text: str) -> str:
        substitutions = (
            (r"ph", "f"),
            (r"tion", "ʃən"),
            (r"ch", "tʃ"),
            (r"sh", "ʃ"),
            (r"th", "θ"),
            (r"qu", "kw"),
            (r"c(?=[eiy])", "s"),
            (r"c", "k"),
            (r"g(?=[eiy])", "dʒ"),
            (r"x", "ks"),
            (r"y", "i"),
        )
        value = text.replace(" ", "")
        for pattern, replacement in substitutions:
            value = re.sub(pattern, replacement, value)
        return re.sub(r"[^a-zɑ-ʒθðŋɹ]+", "", value)


class PhoneticSimilarity:
    def __init__(self) -> None:
        self._distance: Any | None = None
        try:
            from unittest.mock import patch

            import pandas as pd
            import panphon.distance  # type: ignore[import-not-found]

            # PanPhon 0.22 does not declare UTF-8 while reading its packaged IPA table.
            # On Windows that falls back to the active code page and fails before scoring.
            original_read_csv = pd.read_csv

            def read_csv_utf8(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("encoding", "utf-8")
                return original_read_csv(*args, **kwargs)

            with patch.object(pd, "read_csv", read_csv_utf8):
                self._distance = panphon.distance.Distance()
        except (ImportError, AttributeError, OSError, UnicodeError):
            self._distance = None

    def feature_similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if self._distance is not None:
            try:
                distance = float(self._distance.weighted_feature_edit_distance(left, right))
                return max(0.0, 1.0 - distance / max(len(left), len(right), 1))
            except (KeyError, TypeError, ValueError, AttributeError):
                pass
        return normalized_edit_similarity(left, right)


def normalized_edit_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for row, char_left in enumerate(left, start=1):
        current = [row]
        for column, char_right in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (char_left != char_right),
                )
            )
        previous = current
    return max(0.0, 1.0 - previous[-1] / max(len(left), len(right)))


def ngrams(value: str, size: int = 3) -> set[str]:
    padded = f"^{value}$"
    if len(padded) <= size:
        return {padded}
    return {padded[index : index + size] for index in range(len(padded) - size + 1)}


def ngram_similarity(left: str, right: str, size: int = 3) -> float:
    left_grams, right_grams = ngrams(left, size), ngrams(right, size)
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union) if union else 0.0
