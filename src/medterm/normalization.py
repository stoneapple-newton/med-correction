from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from medterm.models import StructuredMedication

TOKEN_RE = re.compile(r"\b[\w][\w'’-]*\b", re.UNICODE)
STRENGTH_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mcg|μg|ug|mg|g|kg|ml|mL|l|L|%|units?)\b", re.I)
ROUTES = {
    "oral",
    "po",
    "iv",
    "intravenous",
    "im",
    "intramuscular",
    "subcutaneous",
    "sc",
    "topical",
    "inhaled",
}
DOSAGE_FORMS = {
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "solution",
    "suspension",
    "injection",
    "cream",
    "ointment",
    "patch",
    "inhaler",
}


@dataclass(frozen=True)
class TextToken:
    text: str
    start: int
    end: int


def normalize_text(text: str, language: str = "en") -> str:
    value = unicodedata.normalize("NFKC", text)
    value = value.replace("’", "'").replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value).strip()
    if language in {"en", "es", "fr", "de", "it", "pt"}:
        value = value.casefold()
    return value


def normalize_term(text: str, language: str = "en") -> str:
    value = normalize_text(text, language)
    value = re.sub(r"[^\w\s'-]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def tokenize_with_offsets(text: str) -> list[TextToken]:
    return [
        TextToken(match.group(), match.start(), match.end()) for match in TOKEN_RE.finditer(text)
    ]


def extract_medication_fields(text: str) -> StructuredMedication:
    normalized = normalize_text(text)
    strengths = [re.sub(r"\s+", "", match.group()).lower() for match in STRENGTH_RE.finditer(text)]
    words = set(re.findall(r"\b[\w]+\b", normalized))
    return StructuredMedication(
        strengths=list(dict.fromkeys(strengths)),
        routes=sorted(words & ROUTES),
        dosage_forms=sorted(words & DOSAGE_FORMS),
    )


def strip_structured_tokens(text: str) -> str:
    value = STRENGTH_RE.sub(" ", text)
    words = [
        word.strip("-'")
        for word in normalize_term(value).split()
        if word.strip("-'") and word.strip("-'") not in ROUTES | DOSAGE_FORMS
    ]
    return " ".join(words)


def detect_scripts(text: str) -> list[str]:
    scripts: set[str] = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            scripts.add("Latn")
        elif "CYRILLIC" in name:
            scripts.add("Cyrl")
        elif "ARABIC" in name:
            scripts.add("Arab")
        elif "CJK" in name or "IDEOGRAPH" in name:
            scripts.add("Hani")
        elif "HIRAGANA" in name:
            scripts.add("Hira")
        elif "KATAKANA" in name:
            scripts.add("Kana")
        elif "HANGUL" in name:
            scripts.add("Hang")
        elif "HEBREW" in name:
            scripts.add("Hebr")
        elif "DEVANAGARI" in name:
            scripts.add("Deva")
        else:
            scripts.add("Zyyy")
    return sorted(scripts) or ["Zyyy"]


def primary_script(text: str) -> str:
    scripts = detect_scripts(text)
    return scripts[0] if len(scripts) == 1 else "+".join(scripts)


def has_suspicious_mixed_scripts(text: str) -> bool:
    scripts = set(detect_scripts(text))
    if len(scripts) <= 1:
        return False
    # Japanese normally combines kanji, hiragana, and katakana in one sentence or term.
    return not scripts.issubset({"Hani", "Hira", "Kana"})
