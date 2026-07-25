from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from medterm.normalization import normalize_term
from medterm.pronunciation import PronunciationEngine, ngrams


class SourceTerm(BaseModel):
    concept_id: str
    term: str
    aliases: list[str] = Field(default_factory=list)
    language: str = "en"
    risk_tier: str = "standard"
    prior: float = Field(default=0.5, ge=0, le=1)
    provenance: str = "local"
    pronunciations: list[str] = Field(default_factory=list)


class DictionaryEntry(SourceTerm):
    normalized_term: str
    normalized_aliases: list[str]
    pronunciation_forms: list[str]


class DictionaryArtifact(BaseModel):
    artifact_version: str
    created_at: datetime
    source_sha256: str
    entries: list[DictionaryEntry]


def parse_aliases(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def load_source_terms(path: Path) -> list[SourceTerm]:
    if path.suffix.casefold() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [SourceTerm.model_validate(item) for item in data]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = []
        for row in csv.DictReader(handle):
            records.append(
                SourceTerm(
                    concept_id=row["concept_id"],
                    term=row["term"],
                    aliases=parse_aliases(row.get("aliases", "")),
                    language=row.get("language") or "en",
                    risk_tier=row.get("risk_tier") or "standard",
                    prior=float(row.get("prior") or 0.5),
                    provenance=row.get("provenance") or "local",
                    pronunciations=parse_aliases(row.get("pronunciations", "")),
                )
            )
        return records


def build_artifact(
    source_path: Path, output_path: Path, version: str | None = None
) -> DictionaryArtifact:
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_terms = load_source_terms(source_path)
    curated = {
        normalize_term(value, term.language): term.pronunciations
        for term in source_terms
        for value in [term.term, *term.aliases]
        if term.pronunciations
    }
    engine = PronunciationEngine(curated)
    entries = []
    for term in source_terms:
        normalized_aliases = list(
            dict.fromkeys(
                normalize_term(value, term.language) for value in [term.term, *term.aliases]
            )
        )
        pronunciation_forms: list[str] = []
        for value in [term.term, *term.aliases]:
            pronunciation_forms.extend(item.value for item in engine.generate(value, term.language))
        entries.append(
            DictionaryEntry(
                **term.model_dump(),
                normalized_term=normalize_term(term.term, term.language),
                normalized_aliases=normalized_aliases,
                pronunciation_forms=list(dict.fromkeys(pronunciation_forms)),
            )
        )
    artifact = DictionaryArtifact(
        artifact_version=version
        or f"dict_{datetime.now(UTC).date().isoformat()}_{source_hash[:8]}",
        created_at=datetime.now(UTC),
        source_sha256=source_hash,
        entries=entries,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return artifact


def load_artifact(path: Path) -> DictionaryArtifact:
    return DictionaryArtifact.model_validate_json(path.read_text(encoding="utf-8"))


class DictionaryIndex:
    def __init__(self, artifact: DictionaryArtifact) -> None:
        self.artifact = artifact
        self.exact: dict[str, set[int]] = defaultdict(set)
        self.alias_values: list[str] = []
        self.alias_entry_ids: list[int] = []
        self.ipa_index: dict[str, set[int]] = defaultdict(set)
        self.curated: dict[str, list[str]] = {}
        for entry_id, entry in enumerate(artifact.entries):
            for alias in entry.normalized_aliases:
                self.exact[alias].add(entry_id)
                self.alias_values.append(alias)
                self.alias_entry_ids.append(entry_id)
                if entry.pronunciations:
                    self.curated[alias] = entry.pronunciations
            for pronunciation in entry.pronunciation_forms:
                for gram in ngrams(pronunciation):
                    self.ipa_index[gram].add(entry_id)

    def is_exact(self, normalized: str) -> bool:
        return normalized in self.exact

    def retrieve(
        self, normalized: str, pronunciations: list[str], limit_per_channel: int = 20
    ) -> dict[int, dict[str, Any]]:
        candidates: dict[int, dict[str, Any]] = {}
        for entry_id in self.exact.get(normalized, set()):
            candidates.setdefault(entry_id, {})["exact"] = 1.0

        for match, score, alias_index in process.extract(
            normalized,
            self.alias_values,
            scorer=fuzz.WRatio,
            limit=limit_per_channel,
            score_cutoff=35,
        ):
            entry_id = self.alias_entry_ids[alias_index]
            info = candidates.setdefault(entry_id, {})
            if score / 100 > info.get("orthographic", 0):
                info["orthographic"] = score / 100
                info["matched_alias"] = match

        phonetic_counts: dict[int, int] = defaultdict(int)
        query_gram_count = 0
        for pronunciation in pronunciations:
            query_grams = ngrams(pronunciation)
            query_gram_count = max(query_gram_count, len(query_grams))
            for gram in query_grams:
                for entry_id in self.ipa_index.get(gram, set()):
                    phonetic_counts[entry_id] += 1
        for entry_id, count in sorted(
            phonetic_counts.items(), key=lambda item: item[1], reverse=True
        )[:limit_per_channel]:
            candidates.setdefault(entry_id, {})["phonetic_block"] = count / max(query_gram_count, 1)
        return candidates

    def search(self, query: str, limit: int = 10) -> list[tuple[DictionaryEntry, float]]:
        normalized = normalize_term(query)
        if not normalized:
            return []
        entry_scores: dict[int, float] = {}
        for _, score, alias_index in process.extract(
            normalized,
            self.alias_values,
            scorer=fuzz.WRatio,
            limit=max(limit * 3, 20),
            score_cutoff=25,
        ):
            entry_id = self.alias_entry_ids[alias_index]
            entry_scores[entry_id] = max(entry_scores.get(entry_id, 0.0), score / 100)
        return [
            (self.artifact.entries[entry_id], score)
            for entry_id, score in sorted(
                entry_scores.items(), key=lambda item: item[1], reverse=True
            )[:limit]
        ]
