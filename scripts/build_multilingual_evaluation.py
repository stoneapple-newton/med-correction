"""Build the deterministic synthetic multilingual medical-term correction benchmark."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_VERSION = "term-trace-multilingual-correction-v1"
DEFAULT_OUTPUT = Path("data/evaluation_multilingual_v1.jsonl")
DEFAULT_MANIFEST = Path("data/evaluation_multilingual_v1.manifest.json")


@dataclass(frozen=True)
class Term:
    key: str
    text: str
    category: str
    risk_tier: str = "standard"


@dataclass(frozen=True)
class Language:
    locale: str
    region: str
    script: str
    replacement: str
    foreign_replacement: str
    terms: tuple[Term, ...]
    templates: tuple[str, str, str, str, str, str]

    @property
    def code(self) -> str:
        return self.locale.split("-")[0]


def _terms(*values: tuple[str, str, str, str | None]) -> tuple[Term, ...]:
    return tuple(
        Term(key, text, category, risk_tier or "standard")
        for key, text, category, risk_tier in values
    )


LANGUAGES: tuple[Language, ...] = (
    Language(
        "en-US",
        "global",
        "Latn",
        "a",
        "а",
        _terms(
            ("metformin", "metformin", "medication", None),
            ("amoxicillin", "amoxicillin", "medication", None),
            ("ibuprofen", "ibuprofen", "medication", None),
            ("acetaminophen", "acetaminophen", "medication", None),
            ("warfarin", "warfarin", "medication", "critical"),
            ("insulin", "insulin", "medication", "high"),
            ("hypertension", "hypertension", "condition", None),
            ("pneumonia", "pneumonia", "condition", None),
            ("diabetes", "diabetes mellitus", "condition", None),
            ("myocardial-infarction", "myocardial infarction", "condition", "high"),
        ),
        (
            "The medication list contains {term}.",
            "Administer {term} 5 mg tonight.",
            "The transcript recorded {term}.",
            "Review the entry {term} before discharge.",
            "The ASR confidence for {term} was low.",
            "Continue {term} as documented.",
        ),
    ),
    Language(
        "es-ES",
        "europe",
        "Latn",
        "a",
        "а",
        _terms(
            ("metformin", "metformina", "medication", None),
            ("amoxicillin", "amoxicilina", "medication", None),
            ("ibuprofen", "ibuprofeno", "medication", None),
            ("acetaminophen", "paracetamol", "medication", None),
            ("warfarin", "warfarina", "medication", "critical"),
            ("insulin", "insulina", "medication", "high"),
            ("hypertension", "hipertensión", "condition", None),
            ("pneumonia", "neumonía", "condition", None),
            ("diabetes", "diabetes mellitus", "condition", None),
            ("myocardial-infarction", "infarto de miocardio", "condition", "high"),
        ),
        (
            "La lista médica contiene {term}.",
            "Administrar {term} 5 mg esta noche.",
            "La transcripción registró {term}.",
            "Revise la entrada {term} antes del alta.",
            "La confianza del ASR para {term} fue baja.",
            "Continuar {term} según lo documentado.",
        ),
    ),
    Language(
        "fr-FR",
        "europe",
        "Latn",
        "a",
        "а",
        _terms(
            ("metformin", "metformine", "medication", None),
            ("amoxicillin", "amoxicilline", "medication", None),
            ("ibuprofen", "ibuprofène", "medication", None),
            ("acetaminophen", "paracétamol", "medication", None),
            ("warfarin", "warfarine", "medication", "critical"),
            ("insulin", "insuline", "medication", "high"),
            ("hypertension", "hypertension", "condition", None),
            ("pneumonia", "pneumonie", "condition", None),
            ("diabetes", "diabète sucré", "condition", None),
            ("myocardial-infarction", "infarctus du myocarde", "condition", "high"),
        ),
        (
            "La liste médicale contient {term}.",
            "Administrer {term} 5 mg ce soir.",
            "La transcription indique {term}.",
            "Vérifier l’entrée {term} avant la sortie.",
            "La confiance ASR pour {term} était faible.",
            "Continuer {term} comme indiqué.",
        ),
    ),
    Language(
        "de-DE",
        "europe",
        "Latn",
        "a",
        "а",
        _terms(
            ("metformin", "Metformin", "medication", None),
            ("amoxicillin", "Amoxicillin", "medication", None),
            ("ibuprofen", "Ibuprofen", "medication", None),
            ("acetaminophen", "Paracetamol", "medication", None),
            ("warfarin", "Warfarin", "medication", "critical"),
            ("insulin", "Insulin", "medication", "high"),
            ("hypertension", "Hypertonie", "condition", None),
            ("pneumonia", "Pneumonie", "condition", None),
            ("diabetes", "Diabetes mellitus", "condition", None),
            ("myocardial-infarction", "Myokardinfarkt", "condition", "high"),
        ),
        (
            "Die Medikamentenliste enthält {term}.",
            "Heute Abend {term} 5 mg verabreichen.",
            "Das Transkript enthält {term}.",
            "Den Eintrag {term} vor der Entlassung prüfen.",
            "Die ASR-Konfidenz für {term} war niedrig.",
            "{term} wie dokumentiert fortsetzen.",
        ),
    ),
    Language(
        "it-IT",
        "europe",
        "Latn",
        "a",
        "а",
        _terms(
            ("metformin", "metformina", "medication", None),
            ("amoxicillin", "amoxicillina", "medication", None),
            ("ibuprofen", "ibuprofene", "medication", None),
            ("acetaminophen", "paracetamolo", "medication", None),
            ("warfarin", "warfarin", "medication", "critical"),
            ("insulin", "insulina", "medication", "high"),
            ("hypertension", "ipertensione", "condition", None),
            ("pneumonia", "polmonite", "condition", None),
            ("diabetes", "diabete mellito", "condition", None),
            ("myocardial-infarction", "infarto miocardico", "condition", "high"),
        ),
        (
            "L’elenco medico contiene {term}.",
            "Somministrare {term} 5 mg stasera.",
            "La trascrizione riporta {term}.",
            "Controllare la voce {term} prima della dimissione.",
            "La confidenza ASR per {term} era bassa.",
            "Continuare {term} come documentato.",
        ),
    ),
    Language(
        "pt-PT",
        "europe",
        "Latn",
        "a",
        "а",
        _terms(
            ("metformin", "metformina", "medication", None),
            ("amoxicillin", "amoxicilina", "medication", None),
            ("ibuprofen", "ibuprofeno", "medication", None),
            ("acetaminophen", "paracetamol", "medication", None),
            ("warfarin", "varfarina", "medication", "critical"),
            ("insulin", "insulina", "medication", "high"),
            ("hypertension", "hipertensão", "condition", None),
            ("pneumonia", "pneumonia", "condition", None),
            ("diabetes", "diabetes mellitus", "condition", None),
            ("myocardial-infarction", "enfarte do miocárdio", "condition", "high"),
        ),
        (
            "A lista médica contém {term}.",
            "Administrar {term} 5 mg esta noite.",
            "A transcrição registou {term}.",
            "Rever a entrada {term} antes da alta.",
            "A confiança do ASR para {term} foi baixa.",
            "Continuar {term} conforme documentado.",
        ),
    ),
    Language(
        "ru-RU",
        "europe",
        "Cyrl",
        "а",
        "a",
        _terms(
            ("metformin", "метформин", "medication", None),
            ("amoxicillin", "амоксициллин", "medication", None),
            ("ibuprofen", "ибупрофен", "medication", None),
            ("acetaminophen", "парацетамол", "medication", None),
            ("warfarin", "варфарин", "medication", "critical"),
            ("insulin", "инсулин", "medication", "high"),
            ("hypertension", "гипертония", "condition", None),
            ("pneumonia", "пневмония", "condition", None),
            ("diabetes", "сахарный диабет", "condition", None),
            ("myocardial-infarction", "инфаркт миокарда", "condition", "high"),
        ),
        (
            "В списке указан термин {term}.",
            "Назначить {term} 5 мг на ночь.",
            "В расшифровке записано {term}.",
            "Проверить запись {term} перед выпиской.",
            "Уверенность ASR для {term} была низкой.",
            "Продолжить {term} согласно записи.",
        ),
    ),
    Language(
        "zh-CN",
        "asia",
        "Hans",
        "医",
        "x",
        _terms(
            ("metformin", "二甲双胍", "medication", None),
            ("amoxicillin", "阿莫西林", "medication", None),
            ("ibuprofen", "布洛芬", "medication", None),
            ("acetaminophen", "对乙酰氨基酚", "medication", None),
            ("warfarin", "华法林", "medication", "critical"),
            ("insulin", "胰岛素", "medication", "high"),
            ("hypertension", "高血压", "condition", None),
            ("pneumonia", "肺炎", "condition", None),
            ("diabetes", "糖尿病", "condition", None),
            ("myocardial-infarction", "心肌梗死", "condition", "high"),
        ),
        (
            "用药清单中记录了{term}。",
            "今晚给予{term}5毫克。",
            "转写文本记录为{term}。",
            "出院前请核对{term}这一项。",
            "{term}的语音识别置信度较低。",
            "按记录继续使用{term}。",
        ),
    ),
    Language(
        "ja-JP",
        "asia",
        "Jpan",
        "ン",
        "x",
        _terms(
            ("metformin", "メトホルミン", "medication", None),
            ("amoxicillin", "アモキシシリン", "medication", None),
            ("ibuprofen", "イブプロフェン", "medication", None),
            ("acetaminophen", "アセトアミノフェン", "medication", None),
            ("warfarin", "ワルファリン", "medication", "critical"),
            ("insulin", "インスリン", "medication", "high"),
            ("hypertension", "高血圧", "condition", None),
            ("pneumonia", "肺炎", "condition", None),
            ("diabetes", "糖尿病", "condition", None),
            ("myocardial-infarction", "心筋梗塞", "condition", "high"),
        ),
        (
            "薬剤一覧に{term}と記載されています。",
            "今夜{term}を5 mg投与します。",
            "文字起こしには{term}と記録されました。",
            "退院前に{term}の項目を確認してください。",
            "{term}の音声認識信頼度は低値でした。",
            "記録どおり{term}を継続します。",
        ),
    ),
    Language(
        "ko-KR",
        "asia",
        "Kore",
        "가",
        "x",
        _terms(
            ("metformin", "메트포르민", "medication", None),
            ("amoxicillin", "아목시실린", "medication", None),
            ("ibuprofen", "이부프로펜", "medication", None),
            ("acetaminophen", "아세트아미노펜", "medication", None),
            ("warfarin", "와파린", "medication", "critical"),
            ("insulin", "인슐린", "medication", "high"),
            ("hypertension", "고혈압", "condition", None),
            ("pneumonia", "폐렴", "condition", None),
            ("diabetes", "당뇨병", "condition", None),
            ("myocardial-infarction", "심근경색", "condition", "high"),
        ),
        (
            "약물 목록에 {term}이 기록되어 있습니다.",
            "오늘 밤 {term} 5 mg을 투여합니다.",
            "전사 기록에는 {term}으로 적혀 있습니다.",
            "퇴원 전에 {term} 항목을 확인하십시오.",
            "{term}의 음성 인식 신뢰도가 낮았습니다.",
            "기록대로 {term}을 계속합니다.",
        ),
    ),
    Language(
        "hi-IN",
        "asia",
        "Deva",
        "क",
        "x",
        _terms(
            ("metformin", "मेटफॉर्मिन", "medication", None),
            ("amoxicillin", "अमोक्सिसिलिन", "medication", None),
            ("ibuprofen", "इबुप्रोफेन", "medication", None),
            ("acetaminophen", "पैरासिटामोल", "medication", None),
            ("warfarin", "वारफारिन", "medication", "critical"),
            ("insulin", "इंसुलिन", "medication", "high"),
            ("hypertension", "उच्च रक्तचाप", "condition", None),
            ("pneumonia", "निमोनिया", "condition", None),
            ("diabetes", "मधुमेह", "condition", None),
            ("myocardial-infarction", "हृदयाघात", "condition", "high"),
        ),
        (
            "दवा सूची में {term} दर्ज है।",
            "आज रात {term} 5 mg दें।",
            "प्रतिलेख में {term} दर्ज हुआ।",
            "छुट्टी से पहले {term} प्रविष्टि जाँचें।",
            "{term} के लिए ASR विश्वास कम था।",
            "रिकॉर्ड के अनुसार {term} जारी रखें।",
        ),
    ),
)


def _units(text: str) -> list[str]:
    units: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if units and (category.startswith("M") or character in {"\u200c", "\u200d"}):
            units[-1] += character
        elif units and units[-1].endswith(("\u200c", "\u200d")):
            units[-1] += character
        else:
            units.append(character)
    return units


def _editable_indices(units: list[str]) -> list[int]:
    return [index for index, unit in enumerate(units) if not unit.isspace()]


def _mutations(term: str, replacement: str, foreign_replacement: str) -> dict[str, str]:
    units = _units(term)
    editable = _editable_indices(units)
    middle = editable[len(editable) // 2]

    deleted = units[:middle] + units[middle + 1 :]

    pair_position = max(0, len(editable) // 2 - 1)
    left, right = editable[pair_position], editable[min(pair_position + 1, len(editable) - 1)]
    transposed = units.copy()
    transposed[left], transposed[right] = transposed[right], transposed[left]

    spaced = units.copy()
    spaced.insert(middle, " ")

    substituted = units.copy()
    substitute = replacement
    if units[middle].casefold() == replacement.casefold():
        substitute = replacement + replacement
    substituted[middle] = substitute

    script_confused = units.copy()
    script_confused[middle] = foreign_replacement

    values = {
        "character_deletion": "".join(deleted),
        "character_transposition": "".join(transposed),
        "word_boundary": "".join(spaced),
        "character_substitution": "".join(substituted),
        "script_confusion": "".join(script_confused),
    }
    if any(value == term for value in values.values()):
        raise ValueError(f"mutation failed to change term: {term!r}")
    if len(set(values.values())) != len(values):
        raise ValueError(f"mutations are not unique for term: {term!r}")
    return values


def _render(template: str, term: str) -> tuple[str, int, int]:
    text = template.format(term=term)
    start = text.index(term)
    return text, start, start + len(term)


def _record(
    language: Language,
    term: Term,
    condition: str,
    source: str,
    template_index: int,
    *,
    negative: bool = False,
) -> dict[str, Any]:
    text, start, end = _render(language.templates[template_index], source)
    record: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "case_id": f"{language.code}-{term.key}-{condition}",
        "text": text,
        "locale": language.locale,
        "tags": [
            "negative" if negative else "positive",
            f"language:{language.code}",
            f"region:{language.region}",
            f"category:{term.category}",
            condition,
        ],
        "split": "test",
        "source_kind": "synthetic",
        "provenance": "term-trace-synthetic-templates-v1",
        "gold_spans": [],
    }
    if negative:
        return record

    record["gold_spans"] = [
        {
            "char_start": start,
            "char_end": end,
            "span_text": source,
            "gold_term": term.text,
            "concept_id": f"SYNTH:{language.code}:{term.key}",
            "language": language.code,
            "script": language.script,
            "code_switch": condition == "script_confusion",
            "dose_adjacent": template_index == 1,
            "risk_tier": term.risk_tier,
            "error_type": condition,
            "term_category": term.category,
        }
    ]
    record["reference_transcript"] = language.templates[template_index].format(term=term.text)
    record["asr_hypothesis"] = text
    if condition == "word_boundary":
        record["n_best"] = [
            {"text": record["reference_transcript"], "confidence": 0.58},
            {"text": text, "confidence": 0.42},
        ]
    if condition == "low_confidence":
        record["tokens"] = [
            {
                "text": source,
                "char_start": start,
                "char_end": end,
                "confidence": 0.35,
            }
        ]
    return record


def build_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for language in LANGUAGES:
        for term in language.terms:
            mutations = _mutations(
                term.text, language.replacement, language.foreign_replacement
            )
            records.extend(
                [
                    _record(
                        language,
                        term,
                        "character_deletion",
                        mutations["character_deletion"],
                        0,
                    ),
                    _record(
                        language,
                        term,
                        "character_transposition",
                        mutations["character_transposition"],
                        1,
                    ),
                    _record(
                        language,
                        term,
                        "word_boundary",
                        mutations["word_boundary"],
                        2,
                    ),
                    _record(
                        language,
                        term,
                        "character_substitution",
                        mutations["character_substitution"],
                        3,
                    ),
                    _record(language, term, "low_confidence", term.text, 4),
                    _record(
                        language,
                        term,
                        "script_confusion",
                        mutations["script_confusion"],
                        3,
                    ),
                    _record(
                        language,
                        term,
                        "correct_term_control",
                        term.text,
                        5,
                        negative=True,
                    ),
                ]
            )
    return records


def serialize(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )


def build_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    language_counts = Counter(record["locale"].split("-")[0] for record in records)
    condition_counts = Counter(record["tags"][-1] for record in records)
    positive_count = sum(bool(record["gold_spans"]) for record in records)
    return {
        "dataset_version": DATASET_VERSION,
        "description": "Synthetic multilingual medical-term detection and correction benchmark",
        "generated_by": "scripts/build_multilingual_evaluation.py",
        "record_count": len(records),
        "positive_record_count": positive_count,
        "negative_record_count": len(records) - positive_count,
        "language_count": len(language_counts),
        "records_per_language": dict(sorted(language_counts.items())),
        "conditions": dict(sorted(condition_counts.items())),
        "languages": [
            {
                "locale": language.locale,
                "region": language.region,
                "script": language.script,
                "term_count": len(language.terms),
                "review_status": "requires_native_speaker_and_clinical_review",
            }
            for language in LANGUAGES
        ],
        "source_kind": "synthetic",
        "contains_real_patient_data": False,
        "contains_third_party_dataset_rows": False,
        "redistribution_notice": (
            "Project-generated synthetic records. The repository owner must declare a dataset "
            "license before external redistribution."
        ),
        "limitations": [
            "Term translations and templates require native-speaker and clinical review.",
            "Character mutations model controlled stress cases, not observed prevalence.",
            "The corpus must not be used to claim clinical performance.",
            "Synthetic concept IDs are not terminology-authority identifiers.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true", help="verify the committed file")
    args = parser.parse_args()
    records = build_records()
    content = serialize(records)
    manifest_content = json.dumps(
        build_manifest(records), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if args.check:
        dataset_matches = (
            args.output.exists() and args.output.read_text(encoding="utf-8") == content
        )
        manifest_matches = (
            args.manifest.exists()
            and args.manifest.read_text(encoding="utf-8") == manifest_content
        )
        if not dataset_matches or not manifest_matches:
            raise SystemExit(f"generated dataset is stale: {args.output}")
        print(
            f"Validated {args.output} and {args.manifest} "
            f"({content.count(chr(10))} records)"
        )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(manifest_content, encoding="utf-8", newline="\n")
    print(
        f"Wrote {args.output} and {args.manifest} "
        f"({content.count(chr(10))} records)"
    )


if __name__ == "__main__":
    main()
