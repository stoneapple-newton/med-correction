# Multilingual medical-term dataset research

Research checked on 2026-07-25. This is a technical inventory, not legal advice. Confirm the
current license and data-use terms with the resource owner before downloading, deriving, or
redistributing any third-party material.

## Decision

The repository benchmark remains fully synthetic and contains no copied third-party rows, audio,
patient text, UMLS content, SNOMED CT content, or ICD-11 content. This avoids mixing incompatible
licenses and controlled clinical data into a distributable test fixture. Existing corpora should
be evaluated through local adapters after the user obtains them under their own applicable terms.

The first language matrix contains English plus six widely used European languages (Spanish,
French, German, Italian, Portuguese, Russian) and four widely used Asian languages (Mandarin
Chinese, Japanese, Korean, Hindi). This is a practical initial scope, not a claim that these are
the only “major” languages or that they represent all speakers, regions, scripts, or clinical
settings.

## Existing resources worth using

| Resource | Languages | Relevant data | Verified access/license signal | Recommended use |
|---|---|---|---|---|
| [MCSCSet](https://github.com/yzhihao/MCSCSet) | Chinese | 196,496 wrong/correct medical-query pairs plus a medical confusion set | Dataset CC BY-NC 2.0; code Apache-2.0 | Strong direct spelling-correction benchmark for non-commercial evaluation. Keep it external unless the repository adopts compatible non-commercial distribution terms. |
| [Eka Medical ASR Evaluation Dataset](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset) | English, Hindi | 3,939 audio/transcript rows with medical-entity types and offsets | Dataset card declares MIT; 3,619 English and 320 Hindi rows | Best immediately reusable source for English/Hindi medical ASR and keyword error analysis. Download locally; preserve attribution and inspect consent/privacy metadata before repackaging audio. |
| [MultiMed](https://huggingface.co/datasets/leduckhai/MultiMed) | English, French, German, Mandarin Chinese, Vietnamese | 48,369 medical audio/transcript rows; about 150 hours | Dataset card declares MIT; paper says source material was collected from YouTube and portions were anonymized | Useful for realistic ASR hypotheses in four benchmark languages. Because underlying recordings came from YouTube, perform a source-rights review before redistributing audio or derived clips. |
| [E3C](https://e3c.fbk.eu/data) | English, French, Italian, Spanish, Basque | Clinical-case text with entity spans, temporal information, factuality, and some concept links | Freely downloadable from the European Language Grid; catalog reports no personal or sensitive data, but the corpus aggregates source documents with licenses | Good span-detection and terminology-context source. Preserve per-document license/provenance; do not assume one blanket redistribution license. |
| [QUAERO French Medical Corpus](https://huggingface.co/datasets/bigbio/quaero/blob/main/README.md) | French | Medical named-entity spans linked using UMLS concepts | BigBio card reports GFDL 1.3; UMLS-linked content introduces separate terminology constraints | Useful French span benchmark. Keep external and require a local build step after confirming both corpus and UMLS terms. |
| [Mantra GSC paper/record](https://repub.eur.nl/pub/82618) | English, French, German, Spanish, Dutch | Biomedical concept spans in parallel titles, drug labels, and patent claims | Access record describes UMLS-based annotations; redistribution terms were not clear from the primary record checked | Valuable multilingual NER comparison, but treat as license-review-required and do not vendor it. |
| [CADEC paper](https://pubmed.ncbi.nlm.nih.gov/25817970) | English | Consumer posts with drug, adverse-event, disease, and symptom spans normalized to controlled terminologies | Corpus is publicly referenced, but confirm the current CSIRO package license and downstream MedDRA/SNOMED terms before reuse | Useful observed misspellings and normalization cases after license review; not suitable for silent vendoring. |
| [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) | English | De-identified real clinical records and notes | Credentialed access. PhysioNet explicitly says derived datasets/models must be treated as sensitive and shared under the same agreement | Do not redistribute. Provide only local evaluation adapters for credentialed users. |
| [SpellingCorpus](https://github.com/AnneDirkson/SpellingCorpus) | English | Manually corrected medical spelling mistakes from a patient forum | Public repository; no clear dataset license was confirmed in this review | Useful as a small external comparison only after author/license confirmation. |
| [KEBAP](https://openreview.net/forum?id=i17SCD0YDI) | Korean | Explainable Korean ASR-error benchmark, not specifically medical | Public paper; dataset redistribution license was not confirmed | Use its error taxonomy to improve Korean synthetic conditions, not as medical ground truth. |

No comparably direct, clearly licensed medical spelling-correction corpus was confirmed for
Japanese, Korean, Russian, German, Italian, Spanish, or Portuguese. E3C and MultiMed cover parts of
the European/Chinese span and ASR problem; language-specific medical correction still requires
clinically reviewed annotation or carefully documented synthetic generation.

## Terminology sources

- [Wikidata structured data is CC0](https://www.wikidata.org/wiki/Wikidata:Licensing) and is the
  safest starting point for multilingual labels, but individual labels still require clinical
  validation and stable retrieval-date/QID provenance.
- [UMLS requires a license](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement.html)
  and includes source vocabularies with additional restrictions. Do not commit a UMLS-derived term
  subset without a source-by-source redistribution review.
- [SNOMED CT use is licensed by territory](https://www.snomed.org/licensing). Registration is
  generally required even where use is free in member countries; redistribution is not equivalent
  to public-domain release.
- [ICD-11 is CC BY-ND 3.0 IGO](https://www.who.int/standards/classifications/classification-of-diseases).
  “NoDerivatives” makes transformed/noisy term datasets legally sensitive; use the API locally and
  obtain permission before distributing derived corrections.

## Safe integration plan

1. Keep the committed benchmark synthetic and provenance-labelled.
2. Add optional download/build adapters for MCSCSet, Eka, MultiMed, and E3C; never download them in
   normal tests or commit their content.
3. Store source URL, version, retrieval date, license identifier, source row ID, transformation
   log, and whether audio or patient-derived content is present in every locally built record.
4. Use separate evaluation splits by source and language. Never mix synthetic results with
   observed-corpus results into one claimed performance number.
5. Require native-speaker and clinical review of canonical terms, contexts, and realistic error
   distributions before using this benchmark for threshold selection or product claims.

