# Strategy A: Fixed-Length Speech-Content Vectors

> **Implementation status (2026-07-26):** The repository now includes an optional XLS-R-compatible
> encoder, mean/statistics pooling, validated projection-checkpoint loading, multiple reference
> vectors per concept, metadata filtering, exact FAISS/NumPy cosine search, and review-only CLI
> commands. The initial smoke test is a baseline test, not medical validation. Medical-term
> contrastive training, held-out evaluation, phoneme reranking, and ASR-evidence fusion remain
> required before production use.

## Local CUDA Smoke-Test Result

On 2026-07-26, the baseline ran successfully on an NVIDIA GeForce RTX 4070 Laptop GPU using
PyTorch 2.13.0 with CUDA 13.0. The model revision was
`facebook/wav2vec2-xls-r-300m@1a640f32ac3e39899438a2931f9924c02f080a54`; statistics pooling
produced a 2048-dimensional normalized vector and FAISS `IndexFlatIP` performed retrieval.

The test used only locally synthesized, isolated terms. A slower synthetic “metformin” query had
cosine similarity 0.985787 to a synthetic “metformin” reference, but 0.995959 to a synthetic
“metoprolol” reference. The wrong term therefore ranked first. This is useful negative evidence:
the GPU pipeline and review-only search work, but raw generic XLS-R statistics vectors do **not**
yet provide adequate word discrimination. No accuracy or clinical-performance conclusion can be
drawn from three synthetic recordings. The next required experiment is contrastive training of
the 256-dimensional projection head with same-term variants and hard medical-name negatives,
followed by held-out multilingual/accent/noise evaluation and phoneme/ASR reranking.

The projection-training command was also smoke-tested on four synthetic, in-sample recordings:
two “metformin” and two “metoprolol” variants. Thirty CUDA epochs reduced the supervised
contrastive training loss from 1.170815 to 1.083556 and produced a 256-dimensional checkpoint.
For the training-set “metformin” query, the projected search ranked “metformin” at 0.999513 above
“metoprolol” at 0.995150. This merely proves that checkpoint training, loading, and retrieval are
wired correctly. The example is extremely small, in-sample, synthetic, and still poorly
separated; it is not evidence of generalization or clinical suitability.

For Strategy A, you need to convert every query recording and every stored reference pronunciation into a fixed-length speech-content vector.

The best starting choices are XLS-R, Wav2Vec2-BERT, or MMS. Avoid ordinary speaker-embedding models such as ECAPA-TDNN: those are designed to recognize who is speaking, not what word was spoken.

## Best Practical Options

### 1. XLS-R

Recommended first baseline:

`facebook/wav2vec2-xls-r-300m`

XLS-R was pretrained on speech from 128 languages and produces frame-level multilingual speech representations.

Use it like this:

```text
audio waveform
    ↓
XLS-R encoder
    ↓
sequence of vectors: [T, 1024]
    ↓
pooling
    ↓
one vector: [1024]
```

Common pooling methods:

- Mean pooling
- Mean plus standard deviation
- Attention pooling
- Temporal convolution followed by pooling

For an initial experiment, use mean pooling.

```python
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def create_audio_embedding(audio_path: str) -> np.ndarray:
    waveform, sample_rate = librosa.load(
        audio_path,
        sr=16_000,
        mono=True,
    )
    inputs = feature_extractor(
        waveform,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    )
    with torch.inference_mode():
        output = model(**inputs)
        frame_embeddings = output.last_hidden_state

    # Shape: [batch, time, hidden_size]
    embedding = frame_embeddings.mean(dim=1)

    # Normalize for cosine-similarity search.
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return embedding[0].cpu().numpy()
```

Then compare two pronunciations:

```python
from numpy.linalg import norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b)))
```

#### Limitation

Raw XLS-R embeddings still contain:

- Speaker identity
- Accent
- Recording environment
- Speaking rate
- Microphone characteristics

Therefore, raw mean pooling is a useful baseline, but usually not accurate enough for production medical-term matching.

---

### 2. Wav2Vec2-BERT

A stronger multilingual encoder candidate is:

`facebook/w2v-bert-2.0`

Wav2Vec2-BERT was pretrained on about 4.5 million hours across more than 143 languages. It is intended as a pretrained representation model that is fine-tuned for downstream tasks.

Its usage is similar:

```python
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

MODEL_NAME = "facebook/w2v-bert-2.0"

processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2BertModel.from_pretrained(MODEL_NAME)
model.eval()


def create_embedding(audio_path: str) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=16_000, mono=True)
    inputs = processor(
        audio,
        sampling_rate=16_000,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state

    vector = hidden.mean(dim=1)
    vector = torch.nn.functional.normalize(vector, dim=-1)
    return vector[0].cpu().numpy()
```

I would test this alongside XLS-R rather than assume one will always be better.

---

### 3. MMS

MMS is useful when your dictionary includes many lower-resource languages. Its pretrained speech models cover more than 1,400 languages, while its ASR models cover more than 1,100.

MMS is particularly attractive when your European and Asian language list extends beyond common languages such as:

- English
- Spanish
- French
- German
- Mandarin
- Japanese
- Korean

For example, it may help with regional languages and languages with substantially less training data.

However, like XLS-R, its generic hidden states are not automatically optimized for word-level cosine matching. Fine-tuning is still desirable.

---

## The Most Suitable Vector: Acoustic Word Embedding

The ideal model for your task is not merely a generic speech encoder. It is an **acoustic word embedding (AWE)** model.

An AWE maps a variable-duration spoken word or phrase into one vector:

```text
"metformin" spoken by person A ─┐
"metformin" spoken by person B ─┼── nearby vectors
"metformin" spoken with accent ─┘
"metronidazole" ─────────────────── farther away
```

Multilingual acoustic word embeddings have been developed specifically for speech search, spoken-term detection, and word discrimination. Research also shows that jointly trained acoustic and written-word embeddings can place spoken segments and written words into compatible phonetic spaces.

For your project, the strongest design would be:

```text
XLS-R or Wav2Vec2-BERT encoder
            ↓
small projection network
            ↓
256-dimensional normalized vector
```

For example:

```python
import torch
from torch import nn


class MedicalTermEmbeddingHead(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        output_size: int = 256,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_size),
        )

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled = frame_embeddings.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(frame_embeddings.dtype)
            pooled = (frame_embeddings * mask).sum(dim=1)
            pooled /= mask.sum(dim=1).clamp(min=1)

        embedding = self.projection(pooled)
        return nn.functional.normalize(embedding, dim=-1)
```

Train it so that:

- Recordings of the same term are close
- Recordings of different terms are far apart
- Highly confusable medical terms are explicitly used as negatives

---

## How to Train the Vector Model

For each medical term, create several audio variants:

```text
metformin
  ├── English speaker 1
  ├── English speaker 2
  ├── Indian English pronunciation
  ├── Mandarin-accented English
  ├── synthetic female voice
  ├── synthetic male voice
  ├── slower version
  └── noisy telephone version
```

A training batch could contain:

```text
Anchor:    metformin, speaker A
Positive:  metformin, speaker B
Negative:  metronidazole
Negative:  metoprolol
Negative:  methotrexate
```

Use contrastive loss, triplet loss, or supervised contrastive loss.

```python
loss = triplet_loss(
    anchor_embedding,
    positive_embedding,
    negative_embedding,
)
```

The negative examples should heavily emphasize terms that:

- Begin similarly
- End similarly
- Differ by one phoneme
- Are commonly confused clinically
- Occur in the same medical specialty

Few-shot multilingual keyword-spotting research has shown that embedding models can adapt to new keywords with only a few examples, although performance will depend strongly on language and recording conditions.

---

## How Many Vectors per Term?

Do not store only one vector per medical term.

A better structure is:

```text
concept: metformin
English pronunciation vectors:
  vector 1: US English
  vector 2: British English
  vector 3: Indian English
  vector 4: synthetic reference
  vector 5: noisy reference
Spanish pronunciation vectors:
  vector 6: metformina, Spain
  vector 7: metformina, Mexico
Mandarin pronunciation vectors:
  vector 8: 二甲双胍
```

At lookup time, either:

1. Compare against all pronunciation vectors.
2. Calculate a centroid per language/accent cluster.

```python
centroid = np.mean(reference_vectors, axis=0)
centroid /= np.linalg.norm(centroid)
```

Keeping the individual vectors often performs better than using one centroid because pronunciations can be multimodal.

---

## Vector Database Options

After generating normalized vectors, store them in:

- FAISS for a local prototype
- pgvector if your metadata is already in PostgreSQL
- Qdrant for filtering by language and terminology metadata
- Milvus for larger-scale deployments

Each vector should include metadata:

```json
{
  "concept_id": "RXNORM:6809",
  "term": "metformin",
  "language": "en",
  "country": "US",
  "pronunciation_type": "human",
  "speaker_accent": "en-US",
  "source": "validated_recording",
  "embedding_model": "medical-awe-v1"
}
```

Search should filter before or during retrieval:

```text
language = English
concept_type = medication
country in [US, UK]
```

That prevents irrelevant matches from unrelated languages or concept categories.

---

## Do Not Use These as Your Main Vector

### Speaker Embeddings

Avoid:

- ECAPA-TDNN
- x-vector
- Speaker-verification embeddings

These are deliberately trained to make all words from the same speaker similar. SpeechBrain's ECAPA model, for example, is explicitly a speaker-verification model.

That is nearly the opposite of your goal.

### Generic CLAP Embeddings

CLAP-style embeddings are useful for broad semantic audio retrieval, but generally do not preserve fine phoneme distinctions well enough for medication-name matching.

### Whisper Encoder Mean Pooling

You can experiment with Whisper encoder states, but Whisper is trained mainly for transcription rather than same-word acoustic retrieval. Its ASR output is usually more useful than its raw pooled vector.

### MFCC Averages

MFCCs can support DTW, but averaging MFCCs into a single vector loses too much temporal information. They are useful as a baseline, not the preferred production representation.

---

## Recommended Starting Configuration

| Component | Recommendation |
|---|---|
| Encoder | `facebook/wav2vec2-xls-r-300m` |
| Input | 16 kHz mono audio |
| Segment | One isolated word or short phrase |
| Pooling | Masked mean plus standard deviation |
| Projection | 2048 → 512 → 256 |
| Output | 256-dimensional L2-normalized vector |
| Training | Supervised contrastive loss |
| Index | FAISS initially; Qdrant or pgvector later |

Mean-plus-standard-deviation pooling:

```python
def statistics_pooling(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    mean = hidden_states.mean(dim=1)
    std = hidden_states.std(dim=1, unbiased=False)
    return torch.cat([mean, std], dim=-1)
```

This normally retains more information than plain mean pooling.

## Most Important Caveat

A generic XLS-R vector may initially cluster recordings by accent, speaker, or audio quality rather than by exact medical word. That does not mean Strategy A is invalid—it means the encoder needs a word-discrimination projection head and medical-term contrastive fine-tuning.

A sensible progression is:

1. Raw XLS-R plus mean-pooling baseline
2. XLS-R plus statistics pooling
3. Fine-tuned projection head
4. Add hard medical-name negatives
5. Add phoneme and ASR reranking

For a production system, use the vector search to generate approximately 20 candidates, then rerank them using phoneme distance and ASR evidence rather than accepting the nearest vector as the final answer. Automatic candidate commitment should remain disabled, with final outcomes limited to human review or abstention.
