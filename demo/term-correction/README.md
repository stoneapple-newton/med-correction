# Audio transcript correction demo

This demo pairs two short source-audio clips with deliberately altered transcripts. The audio is
unchanged. Only the demo transcript contains the synthetic mistranscription, and the original
excerpt is retained beside it.

| Case | Audio says | Demo transcript says | Expected controlled term |
|---|---|---|---|
| English | `amlodipine` | `am low dipine` | `amlodipine` (`RxCUI:596`) |
| Chinese | `高血压` | `高血鸭` | `高血压` (`LOCAL:hypertension`) |

`高血鸭` is an exact Mandarin homophone of `高血压` (`gāo xuè yā`). The current deterministic
Chinese fallback may highlight the strongest substring, `高血`, while still ranking `高血压` as
the first candidate. The reviewer must confirm the correction; the tool never rewrites the source
transcript automatically.

## Run locally

Build the development dictionary and run the verification script:

```powershell
uv run medterm-build
uv run python scripts/run_term_correction_demo.py
```

The command exits nonzero unless both expected terms rank first, both decisions are `review`, and
`auto_commit_enabled` remains `false`.

## Exercise the reviewer UI

Start the API and reviewer console in separate terminals:

```powershell
uv run medterm-api --host 127.0.0.1 --port 8000
uv run streamlit run reviewer_app.py --server.port 8501
```

For each case:

1. Upload the corresponding clip from `data/audio/term-correction-demo`.
2. Paste the matching `*.mistranscribed.txt` content into the transcript field.
3. Use locale `en-US` for English or `zh-CN` for Chinese.
4. Select **Find terms for review** and inspect the top candidate and score evidence.

The audio clips are generated local demo artifacts and are ignored by Git. Their source files,
time ranges, transcripts, deliberate replacement, and expected result are recorded in
`manifest.json`.
