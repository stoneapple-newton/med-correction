from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from medterm.normalization import normalize_term
from medterm.pronunciation import PronunciationEngine, ngrams

ASIAN_LANGUAGES = frozenset({"zh", "ja", "ko", "hi"})


@dataclass(frozen=True)
class ApproximateAliasSpan:
    start: int
    end: int
    score: float
    entry_id: int


class SourceTerm(BaseModel):
    concept_id: str
    term: str
    aliases: list[str] = Field(default_factory=list)
    language: str = "en"
    risk_tier: str = "unclassified"
    prior: float = Field(default=0.5, ge=0, le=1)
    provenance: str = "local"
    source_vocabulary: str = "unspecified"
    source_release: str = "unspecified"
    licensing_status: str = "unverified"
    review_status: str = "unreviewed"
    concept_type: str = "medical_term"
    source_record_type: str = "unspecified"
    lasa_status: str = "unclassified"
    provenance_detail: dict[str, Any] = Field(default_factory=dict)
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
                    risk_tier=row.get("risk_tier") or "unclassified",
                    prior=float(row.get("prior") or 0.5),
                    provenance=row.get("provenance") or "local",
                    source_vocabulary=row.get("source_vocabulary") or "unspecified",
                    source_release=row.get("source_release") or "unspecified",
                    licensing_status=row.get("licensing_status") or "unverified",
                    review_status=row.get("review_status") or "unreviewed",
                    concept_type=row.get("concept_type") or "medical_term",
                    source_record_type=row.get("source_record_type") or "unspecified",
                    lasa_status=row.get("lasa_status") or "unclassified",
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
        self,
        normalized: str,
        pronunciations: list[str],
        limit_per_channel: int = 20,
        languages: list[str] | None = None,
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

        # A deliberately bounded mutation channel for deletion/substitution-like errors. It is
        # language-filtered and length-bounded so short/common strings cannot fan out broadly.
        if len(normalized) >= 4 and languages:
            mutation_matches: list[tuple[int, float, str]] = []
            requested_languages = set(languages)
            for alias, entry_id in zip(self.alias_values, self.alias_entry_ids, strict=True):
                entry = self.artifact.entries[entry_id]
                max_distance = 1 if max(len(normalized), len(alias)) <= 7 else 2
                if (
                    entry.language not in requested_languages
                    or abs(len(normalized) - len(alias)) > max_distance
                ):
                    continue
                distance = Levenshtein.distance(normalized, alias, score_cutoff=max_distance)
                if distance <= max_distance:
                    score = 1.0 - distance / max(len(normalized), len(alias), 1)
                    mutation_matches.append((entry_id, score, alias))
            for entry_id, score, alias in sorted(
                mutation_matches, key=lambda item: item[1], reverse=True
            )[:limit_per_channel]:
                info = candidates.setdefault(entry_id, {})
                if score > info.get("mutation", 0.0):
                    info["mutation"] = score
                    info.setdefault("matched_alias", alias)
        return candidates

    def find_approximate_asian_spans(
        self,
        text: str,
        languages: list[str],
        *,
        score_cutoff: float = 0.72,
        length_slack: int = 2,
        limit: int = 40,
    ) -> list[ApproximateAliasSpan]:
        """Find near-dictionary substrings without assuming whitespace token boundaries.

        Normalization is applied to each original-text slice, never to the full text, so the
        returned offsets always address the unmodified source string. Exact aliases are omitted;
        they are negative controls unless independent ASR evidence says they need review.
        """
        requested = set(languages) & ASIAN_LANGUAGES
        if not requested or not text:
            return []

        best: dict[tuple[int, int], ApproximateAliasSpan] = {}
        for alias, entry_id in zip(self.alias_values, self.alias_entry_ids, strict=True):
            entry = self.artifact.entries[entry_id]
            if entry.language not in requested or not alias:
                continue
            if alias in normalize_term(text, entry.language):
                continue
            alignment = fuzz.partial_ratio_alignment(alias, text, score_cutoff=50)
            if alignment is None:
                continue
            min_length = max(1, len(alias) - length_slack)
            max_length = min(len(text), len(alias) + length_slack)
            start_floor = max(0, alignment.dest_start - length_slack)
            start_ceiling = min(len(text), alignment.dest_start + length_slack)
            for span_length in range(min_length, max_length + 1):
                for start in range(start_floor, start_ceiling + 1):
                    end = start + span_length
                    if end > len(text):
                        continue
                    observed = normalize_term(text[start:end], entry.language)
                    if not observed or observed == alias:
                        continue
                    score = fuzz.ratio(observed, alias) / 100
                    if score < score_cutoff:
                        continue
                    key = (start, end)
                    candidate = ApproximateAliasSpan(start, end, score, entry_id)
                    previous = best.get(key)
                    if previous is None or score > previous.score:
                        best[key] = candidate
        return sorted(
            best.values(),
            key=lambda item: (item.score, item.end - item.start),
            reverse=True,
        )[:limit]

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
