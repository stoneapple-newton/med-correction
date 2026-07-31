# med-rag Terminology TTS and Audio-Vector Job

## Purpose

This document is the operational handoff for continuing the local, release-scoped TTS and
speech-vector build sourced from the existing `med-rag` terminology database.

This is **not an RxNorm run**. The source database currently contains MeSH, Orphadata, French BDPM,
and a small Wikidata benchmark collection. It contains no registered RxNorm release and no
`RxCUI:` concepts.

The build is a development aid for reviewer-facing retrieval. It is not clinically validated, must
not be published, and must never automatically commit a terminology candidate.

## Safety and licensing boundary

- Build classification: `development_local_only`.
- Recorded license state: `source_catalog_open_local_processing_unreviewed`.
- Artifact distribution approval: `false`.
- Synthetic audio is labeled `synthetic_tts`.
- Imported concepts use `risk_tier=unclassified` and `lasa_status=unclassified`.
- Search decisions remain `review` or `abstain`.
- `auto_commit_enabled` remains `false`.
- The raw 2048-dimensional XLS-R statistics-pooled vectors are an untrained baseline, not evidence
  of clinical retrieval quality.

Do not publish terminology, WAVs, vector shards, or indexes until every source and generated
dependency has separately reviewed distribution approval.

## Source snapshot

The normalized SQLite database is the source of truth. Chroma is only a derived retrieval index.

| Item | Value |
|---|---|
| SQLite database | `C:\Users\Newto\Documents\project\med-rag\data\medical_terms.sqlite` |
| SQLite SHA-256 | `db88ef38e04b650efb7a4b81cdbe3a21ca6c335facd05c271e8e99f92537d642` |
| Source catalog | `C:\Users\Newto\Documents\project\med-rag\data\catalog.json` |
| Chroma collection | `medical_terms_official_local_v1` |
| Chroma documents | 357,793 |
| Chroma manifest | `C:\Users\Newto\Documents\project\med-rag\chroma_db\index-3e6df26967f3f40d.manifest.json` |
| Build root | `C:\Users\Newto\Documents\project\test\data\builds\med-rag-official-local-v1` |

Before continuing, recompute the SQLite checksum. If it differs, stop and register a new release
identity instead of mixing new selections with this build:

```powershell
Get-FileHash `
  C:\Users\Newto\Documents\project\med-rag\data\medical_terms.sqlite `
  -Algorithm SHA256
```

## Eligible preferred-term inventory

Only active preferred terms on active concepts are selected. Ordering is deterministic:

```text
concept_id, normalized_term, term_id
```

| Language | Initial eligible terms | Completed | Remaining |
|---|---:|---:|---:|
| English (`en`) | 40,866 | 5,500 | 35,366 |
| Simplified Chinese (`zh-Hans` → `zh`) | 8,681 | 500 | 8,181 |
| Japanese (`ja`) | 0 | 0 | 0 |

French and the small German/Spanish/Italian/Portuguese/Russian inventories are not part of this
run. The current release policy enables only English and Mandarin Chinese.

## Completed batches

### Batch `000001`

| Language | Source range | TTS | Vectors | Active index SHA-256 |
|---|---:|---:|---:|---|
| English | 1–500 | 500 | 500 × 2048 | `1a3077a3b845ce974c74e1d3c4778bd2a9d017849a7e49f79a4de93b76f6c584` |
| Chinese | 1–500 | 500 | 500 × 2048 | `a02e58f460b67697b12500a6aa7294b55dc3be8e4b7b6208a72a4fef37a1af33` |

The `v2` indexes are authoritative. They preserve source entity types such as
`biomedical_concept`, `clinical_concept`, and `substance`. Earlier non-`v2` indexes remain generated
history and must not be selected for publication or evaluation.

### Batch `000002-5000`

- Language: English.
- Source offset: 500.
- Source range: 501–5,500.
- Planned/generated/indexed: 5,000 / 5,000 / 5,000.
- TTS failures: zero.
- Missing audio: zero.
- WAV checksum mismatches: zero.
- Duplicate reference IDs: zero.
- Vector failures: zero.
- Unexplained reconciliation difference: zero.
- Dimension: 2,048.
- TTS runtime: 436 seconds.
- Index SHA-256:
  `bdea09a9bde37c7c5821c55b87d4fcdc1fbe1b479a0e8ecf1c427382c7aaa361`.
- Vector shard SHA-256:
  `c836ecbf4b59b463bcf1c4274457ebe700afc4f23e7e39bbc419a4fe3d78c198`.
- A no-op resume left both the shard checksum and timestamp unchanged.
- Same-audio smoke test returned `review` and ranked the exact Struvite reference first.

Authoritative files:

```text
data/builds/med-rag-official-local-v1/
  run_manifest.json
  terminology/en/batch-000002-5000.selection.json
  terminology/en/batch-000002-5000.source.json
  terminology/en/batch-000002-5000.dictionary.json
  tts/en/batch-000002-5000.active_references.jsonl
  tts/en/batch-000002-5000.checkpoint.json
  tts/en/batch-000002-5000.failures.jsonl
  tts/en/batch-000002-5000.audio/*.wav
  vectors/en/batch-000002-5000/checkpoint.json
  vectors/en/batch-000002-5000/shards/batch-000001.npz
  indexes/kokoro_vectors_en_batch-000002-5000.npz
  indexes/kokoro_vectors_en_batch-000002-5000.summary.json
```

## Next deterministic selections

Do not reuse an existing batch ID with different offsets, limits, model settings, or source bytes.

| Proposed batch | Language | SQL offset | Limit | Human-readable range |
|---|---|---:|---:|---:|
| `batch-000003-5000` | English | 5,500 | 5,000 | 5,501–10,500 |
| `batch-000002-5000` | Chinese | 500 | 5,000 | 501–5,500 |

After those batches, the next English offset is `10500` and the next Chinese offset is `5500`.
The final batch in a language may contain fewer than 5,000 terms.

## Deterministic selection query

Run this against SQLite in read-only mode. Bind the source language, limit, and offset. Convert
`zh-Hans` to the TTS language `zh` only in the generated artifact; retain `zh-Hans` in provenance.

```sql
SELECT
  t.term_id,
  t.concept_id,
  t.term,
  t.language_tag,
  t.source_release_id,
  c.entity_type,
  c.review_status,
  s.source_id,
  r.version
FROM term AS t
JOIN concept AS c ON c.concept_id = t.concept_id
JOIN source_release AS r ON r.release_id = t.source_release_id
JOIN source AS s ON s.source_id = r.source_id
WHERE t.preferred = 1
  AND t.status = 'active'
  AND c.status = 'active'
  AND t.language_tag = ?
ORDER BY t.concept_id, t.normalized_term, t.term_id
LIMIT ? OFFSET ?;
```

For every selected concept, preserve active aliases in the same source language:

```sql
SELECT term
FROM term
WHERE concept_id = ?
  AND language_tag = ?
  AND term <> ?
  AND status = 'active'
ORDER BY preferred DESC, normalized_term, term_id;
```

Each generated `SourceTerm` must preserve `concept_id`, preferred term, aliases, source vocabulary,
source release, review status, and entity type. Use these conservative fields until separately
reviewed classification and licensing approvals exist:

```json
{
  "risk_tier": "unclassified",
  "lasa_status": "unclassified",
  "licensing_status": "source_catalog_open_local_processing_unreviewed",
  "source_record_type": "preferred"
}
```

## Build the dictionary artifact

After writing and hashing a deterministic source JSON file:

```powershell
$root = "data/builds/med-rag-official-local-v1"
$language = "en"
$batch = "batch-000003-5000"

uv run medterm-build `
  --source "$root/terminology/$language/$batch.source.json" `
  --output "$root/terminology/$language/$batch.dictionary.json" `
  --version "med-rag-official-local-v1-$language-$batch"
```

Write a sibling selection manifest containing the SQLite checksum, source export checksum, source
language, offset, limit, ordering fields, creation time, license state, and
`auto_commit_enabled=false`.

## TTS settings and resume behavior

| Language | Kokoro code | Voice | Device | Sample rate |
|---|---|---|---|---:|
| English | `a` | `af_heart` | `cuda` | 24,000 Hz |
| Chinese | `z` | `zf_xiaobei` | `cuda` | 24,000 Hz |

The TTS worker must:

1. Load the Kokoro pipeline once per language process.
2. Process preferred terms individually so one terminal term failure does not stop the batch.
3. Reuse an existing WAV only when its logical identity and byte checksum match.
4. Write `active_references.jsonl`, `failures.jsonl`, and a checkpoint every 500 terms.
5. Keep full SHA-256 logical `reference_id` and WAV-byte `rendition_id` values.
6. Preserve `entry.concept_type` in every reference.
7. Finish as `complete` only with zero failures, otherwise `complete_with_failures`.

The operational implementation used these library components:

```python
from medterm.terminology import load_artifact
from medterm.tts import (
    KokoroConfig,
    KokoroSynthesizer,
    build_synthetic_references,
    write_reference_manifest,
)

artifact = load_artifact(dictionary_path)
synthesizer = KokoroSynthesizer(
    KokoroConfig(language_code="a", voice="af_heart", device="cuda")
)

for entry in artifact.entries:
    one_entry = artifact.model_copy(update={"entries": [entry]})
    try:
        references.extend(
            build_synthetic_references(
                one_entry,
                synthesizer,
                audio_directory,
                language="en",
            )
        )
    except Exception as exc:
        # Persist concept_id, term, normalized exception class, and bounded reason.
        failures.append(...)
```

Manifests and checkpoints must be written atomically through a same-directory temporary file and
`os.replace`. On restart, validate existing manifest audio checksums before adding their concept IDs
to the completed set. Existing valid WAVs without a committed manifest record may be revalidated and
promoted; do not silently trust filenames alone.

## Vector ingest and finalization

Once a TTS active-reference manifest is complete, run:

```powershell
$root = "data/builds/med-rag-official-local-v1"
$language = "en"
$batch = "batch-000003-5000"

uv run medterm-vector-batch `
  --manifest "$root/tts/$language/$batch.active_references.jsonl" `
  --output "$root/vectors/$language/$batch" `
  --micro-batch-size 16 `
  --shard-size 5000 `
  --device cuda `
  --resume

uv run medterm-finalize-index `
  --vectors "$root/vectors/$language/$batch" `
  --output "$root/indexes/kokoro_vectors_${language}_${batch}.npz"
```

Do not change `micro-batch-size`, `shard-size`, model, pooling, projection, or dtype settings while
resuming an existing checkpoint. A changed stage hash requires a new output directory or explicit,
audited invalidation.

## Monitoring commands

TTS progress:

```powershell
$root = "data/builds/med-rag-official-local-v1"
$language = "en"
$batch = "batch-000003-5000"

(Get-ChildItem "$root/tts/$language/$batch.audio" -Filter "*.wav").Count
Get-Content -Raw "$root/tts/$language/$batch.checkpoint.json"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
```

Vector progress:

```powershell
Get-Content -Raw "$root/vectors/$language/$batch/checkpoint.json"
Get-ChildItem "$root/vectors/$language/$batch/shards"
```

## Required reconciliation

Before marking a batch verified, confirm:

```text
planned references = active references + terminal TTS failures
active references = indexed rows + explicit vector failures
unexplained inventory difference = 0
```

For every active reference:

- resolve the manifest-relative WAV path;
- confirm the file exists;
- recompute SHA-256;
- require it to equal both `audio_sha256` and `rendition_id`;
- require unique logical `reference_id` values.

For every finalized index:

- require the expected row count and dimension;
- reject zero and non-finite vectors;
- require unique reference IDs;
- verify the shard and index checksums in the summary;
- confirm `decision=review` and `auto_commit_enabled=false`.

Repeat `medterm-vector-batch` with the same settings and `--resume`. The completed shard checksum and
timestamp must remain unchanged.

## Smoke test

Use the first active reference as a same-audio execution check. Pass its actual stored concept type;
the CLI defaults to `medication`, which is not appropriate for most MeSH records.

```powershell
uv run medterm-audio-search `
  "$root/indexes/kokoro_vectors_${language}_${batch}.npz" `
  <absolute-path-to-first-active-wav> `
  --language $language `
  --concept-type <stored-concept-type> `
  --top-k 1 `
  --device cuda
```

The exact same recording should rank first. This checks pipeline execution only. It does not validate
retrieval on human audio or establish discrimination between similar medical terms.

## Known gaps before a complete release

- There is no single persisted command yet that exports the next SQLite slice and runs all TTS,
  checkpoint, vector, reconciliation, and manifest-update stages. The first batches used the library
  APIs above with operational scripts.
- Per-batch indexes have not yet been merged into a single language index. Do not concatenate NPZ
  files manually; implement checksum-validated finalization across batch shards.
- No approved human-audio held-out evaluation has been run.
- The raw pooled encoder has no validated projection checkpoint.
- No pronunciation review has been completed for high-risk/LASA terms.
- All imported risk/LASA classifications remain unclassified.
- Distribution licensing approval remains absent.
- Generated artifact garbage collection is not implemented. Do not delete old/v1 artifacts by hand.

Before launching many more batches, the recommended engineering step is to persist the operational
SQLite-export/TTS loop as a release-level runner with the same stage hashes, atomic checkpoint rules,
and reconciliation behavior documented here.

## Verification after code changes

If continuation requires code changes, run from
`C:\Users\Newto\Documents\project\test`:

```powershell
uv run ruff check .
uv run pytest
uv run pytest --cov=medterm --cov-report=term-missing
uv run medterm-evaluate data/evaluation_sample.jsonl
```

Do not weaken review-only or auto-commit safety assertions to make a batch pass.
