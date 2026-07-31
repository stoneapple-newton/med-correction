from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field


class AudioEmbeddingUnavailableError(RuntimeError):
    """Raised when optional speech-embedding dependencies or hardware are unavailable."""


class InvalidAudioError(ValueError):
    """Raised when an audio file cannot safely be embedded."""


class ReferencePronunciation(BaseModel):
    reference_id: str = Field(min_length=1)
    audio_path: Path
    concept_id: str = Field(min_length=1)
    term: str = Field(min_length=1)
    language: str = Field(min_length=1)
    country: str | None = None
    pronunciation_type: str = "human"
    speaker_accent: str | None = None
    source: str = "validated_recording"
    concept_type: str = "medication"
    synthetic: bool = False
    tts_model_id: str | None = None
    tts_voice: str | None = None
    tts_language_code: str | None = None
    tts_phonemes: str | None = None
    tts_sample_rate: int | None = None
    tts_speed: float | None = None
    terminology_version: str | None = None
    terminology_source_sha256: str | None = None
    identity_schema_version: int | None = None
    rendition_id: str | None = None
    audio_sha256: str | None = None


class AudioCandidate(BaseModel):
    reference_id: str
    concept_id: str
    term: str
    language: str
    score: float
    metadata: dict[str, Any]


class AudioSearchResponse(BaseModel):
    decision: Literal["review", "abstain"]
    decision_reasons: list[str]
    auto_commit_enabled: bool = False
    embedding_provenance: dict[str, Any]
    candidates: list[AudioCandidate]


class AudioEncoder(Protocol):
    @property
    def provenance(self) -> dict[str, Any]: ...

    def encode(self, audio_path: Path) -> np.ndarray: ...

    def encode_many(self, audio_paths: list[Path]) -> np.ndarray: ...


@dataclass(frozen=True)
class SpeechEncoderConfig:
    model_name: str = "facebook/wav2vec2-xls-r-300m"
    device: str = "auto"
    pooling: Literal["mean", "statistics"] = "statistics"
    projection_checkpoint: Path | None = None
    sample_rate: int = 16_000
    max_seconds: float = 15.0
    use_half_on_cuda: bool = True


class SpeechContentEncoder:
    """Lazy, optional XLS-R/Wav2Vec2-BERT/MMS content encoder.

    Without a projection checkpoint, this deliberately returns a raw pooled baseline. A random
    256-dimensional head would look production-like while having no word-discrimination training,
    so this class never creates one implicitly.
    """

    def __init__(self, config: SpeechEncoderConfig | None = None) -> None:
        self.config = config or SpeechEncoderConfig()
        self._torch: Any | None = None
        self._feature_extractor: Any | None = None
        self._model: Any | None = None
        self._projection: Any | None = None
        self._device: str | None = None
        self._model_revision: str | None = None
        self._output_dimension: int | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "embedding_model": self.config.model_name,
            "model_revision": self._model_revision or "unresolved_until_model_load",
            "pooling": self.config.pooling,
            "sample_rate": self.config.sample_rate,
            "projection_checkpoint": (
                self.config.projection_checkpoint.name
                if self.config.projection_checkpoint is not None
                else None
            ),
            "projection_trained": self.config.projection_checkpoint is not None,
            "output_dimension": self._output_dimension,
            "device": self._device or self.config.device,
        }

    def encode(self, audio_path: Path) -> np.ndarray:
        pooled = self._encode_pooled_tensor(audio_path)
        torch = self._torch
        with torch.inference_mode():
            if self._projection is not None:
                pooled = self._projection(pooled)
            vector = torch.nn.functional.normalize(pooled.float(), dim=-1)
        result = vector[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise InvalidAudioError(f"Embedding contains non-finite values: {audio_path}")
        self._output_dimension = int(result.shape[0])
        return result

    def encode_many(self, audio_paths: list[Path]) -> np.ndarray:
        """Encode a duration-bucketed batch while excluding padded feature frames from pooling."""
        if not audio_paths:
            raise ValueError("audio_paths must not be empty.")
        self._load_model()
        waveforms = [self._load_waveform(path) for path in audio_paths]
        ordered = sorted(range(len(waveforms)), key=lambda index: waveforms[index].size)
        sorted_waveforms = [waveforms[index] for index in ordered]
        inputs = self._feature_extractor(
            sorted_waveforms,
            sampling_rate=self.config.sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_dtype = next(self._model.parameters()).dtype
        prepared = {}
        for key, value in inputs.items():
            value = value.to(self._device)
            if value.is_floating_point() and key != "attention_mask":
                value = value.to(model_dtype)
            prepared[key] = value
        torch = self._torch
        with torch.inference_mode():
            hidden = self._model(**prepared).last_hidden_state
            feature_mask = self._feature_attention_mask(
                hidden.shape[1], prepared.get("attention_mask")
            )
            pooled = self._pool(hidden, feature_mask)
            if self._projection is not None:
                pooled = self._projection(pooled)
            vectors = torch.nn.functional.normalize(pooled.float(), dim=-1)
        matrix = vectors.detach().cpu().numpy().astype(np.float32, copy=False)
        inverse = np.empty(len(ordered), dtype=np.int64)
        inverse[np.asarray(ordered)] = np.arange(len(ordered))
        matrix = matrix[inverse]
        if not np.all(np.isfinite(matrix)):
            raise InvalidAudioError("A batched embedding contains non-finite values.")
        self._output_dimension = int(matrix.shape[1])
        return matrix

    def extract_pooled(self, audio_path: Path) -> np.ndarray:
        """Return unprojected pooled features for projection-head training."""
        pooled = self._encode_pooled_tensor(audio_path)
        result = pooled[0].float().detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise InvalidAudioError(f"Pooled features contain non-finite values: {audio_path}")
        return result

    def _encode_pooled_tensor(self, audio_path: Path) -> Any:
        self._load_model()
        waveform = self._load_waveform(audio_path)
        torch = self._torch
        inputs = self._feature_extractor(
            waveform,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
        )
        model_dtype = next(self._model.parameters()).dtype
        prepared = {}
        for key, value in inputs.items():
            value = value.to(self._device)
            if value.is_floating_point():
                value = value.to(model_dtype)
            prepared[key] = value
        with torch.inference_mode():
            hidden = self._model(**prepared).last_hidden_state
            return self._pool(hidden)

    def _feature_attention_mask(self, feature_length: int, sample_mask: Any | None) -> Any | None:
        if sample_mask is None:
            return None
        if hasattr(self._model, "_get_feature_vector_attention_mask"):
            return self._model._get_feature_vector_attention_mask(feature_length, sample_mask)
        lengths = sample_mask.sum(dim=-1)
        scaled = self._torch.ceil(lengths * feature_length / sample_mask.shape[1]).long()
        positions = self._torch.arange(feature_length, device=sample_mask.device)
        return positions.unsqueeze(0) < scaled.unsqueeze(1)

    def _pool(self, hidden: Any, attention_mask: Any | None = None) -> Any:
        if attention_mask is None:
            mean = hidden.mean(dim=1)
            variance = hidden.float().var(dim=1, unbiased=False)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            count = mask.sum(dim=1).clamp_min(1)
            mean = (hidden * mask).sum(dim=1) / count
            variance = (((hidden - mean.unsqueeze(1)).float() ** 2) * mask.float()).sum(
                dim=1
            ) / count.float()
        if self.config.pooling == "mean":
            return mean
        std = variance.sqrt().to(hidden.dtype)
        return self._torch.cat([mean, std], dim=-1)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import soundfile  # noqa: F401
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise AudioEmbeddingUnavailableError(
                "Install the 'speech-embeddings' optional dependency to encode audio."
            ) from exc

        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise AudioEmbeddingUnavailableError("CUDA was requested but PyTorch cannot access it.")

        feature_extractor = AutoFeatureExtractor.from_pretrained(self.config.model_name)
        model = AutoModel.from_pretrained(self.config.model_name)
        model.eval().to(device)
        if device.startswith("cuda") and self.config.use_half_on_cuda:
            model.half()

        self._torch = torch
        self._feature_extractor = feature_extractor
        self._model = model
        self._device = device
        self._model_revision = getattr(model.config, "_commit_hash", None)
        if self.config.projection_checkpoint is not None:
            self._projection = self._load_projection(self.config.projection_checkpoint)
        else:
            hidden_size = int(model.config.hidden_size)
            self._output_dimension = (
                hidden_size if self.config.pooling == "mean" else hidden_size * 2
            )

    def _load_projection(self, path: Path) -> Any:
        if not path.is_file():
            raise AudioEmbeddingUnavailableError(f"Projection checkpoint does not exist: {path}")
        checkpoint = self._torch.load(path, map_location=self._device, weights_only=True)
        required = {"input_size", "output_size", "state_dict", "trained"}
        missing = required - checkpoint.keys()
        if missing:
            raise AudioEmbeddingUnavailableError(
                f"Projection checkpoint is missing fields: {', '.join(sorted(missing))}"
            )
        if checkpoint["trained"] is not True:
            raise AudioEmbeddingUnavailableError("Refusing an untrained projection checkpoint.")
        module = _make_projection_module(
            self._torch,
            int(checkpoint["input_size"]),
            int(checkpoint["output_size"]),
        )
        module.load_state_dict(checkpoint["state_dict"])
        module.eval().to(self._device)
        if self._device.startswith("cuda") and self.config.use_half_on_cuda:
            module.half()
        self._output_dimension = int(checkpoint["output_size"])
        return module

    def _load_waveform(self, audio_path: Path) -> np.ndarray:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise AudioEmbeddingUnavailableError(
                "Install the 'speech-embeddings' optional dependency to read audio."
            ) from exc
        if not audio_path.is_file():
            raise InvalidAudioError(f"Audio file does not exist: {audio_path}")
        try:
            audio, source_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        except (RuntimeError, OSError) as exc:
            raise InvalidAudioError(f"Unable to decode audio: {audio_path}") from exc
        waveform = audio.mean(axis=1)
        if waveform.size == 0:
            raise InvalidAudioError(f"Audio file is empty: {audio_path}")
        if waveform.size / source_rate > self.config.max_seconds:
            raise InvalidAudioError(
                f"Audio exceeds the {self.config.max_seconds:g}-second isolated-term limit."
            )
        if source_rate != self.config.sample_rate:
            target_size = round(waveform.size * self.config.sample_rate / source_rate)
            source_positions = np.arange(waveform.size, dtype=np.float64)
            target_positions = np.linspace(0, waveform.size - 1, target_size, dtype=np.float64)
            waveform = np.interp(target_positions, source_positions, waveform).astype(np.float32)
        peak = float(np.max(np.abs(waveform)))
        if peak == 0.0:
            raise InvalidAudioError(f"Audio contains only silence: {audio_path}")
        return waveform


def _make_projection_module(torch: Any, input_size: int, output_size: int) -> Any:
    class ProjectionModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(input_size, 512),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, output_size),
            )

        def forward(self, pooled: Any) -> Any:
            return self.projection(pooled)

    return ProjectionModule()


def train_projection_head(
    features: np.ndarray,
    labels: list[str],
    output_path: Path,
    *,
    output_size: int = 256,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    temperature: float = 0.1,
    device: str = "auto",
    seed: int = 7,
    training_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the projection head with full-batch supervised contrastive loss.

    This is an experiment utility, not a clinical model-training pipeline. Callers remain
    responsible for held-out, versioned multilingual/accent/noise evaluation.
    """
    try:
        import torch
    except ImportError as exc:
        raise AudioEmbeddingUnavailableError(
            "Install the 'speech-embeddings' optional dependency to train a projection."
        ) from exc
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels) or matrix.shape[0] < 3:
        raise ValueError("Training features must align with at least three labels.")
    if len(set(labels)) < 2:
        raise ValueError("Supervised contrastive training requires at least two concepts.")
    counts = {label: labels.count(label) for label in set(labels)}
    if min(counts.values()) < 2:
        raise ValueError("Every concept must have at least two positive recordings.")
    if epochs < 1 or output_size < 1 or temperature <= 0:
        raise ValueError("Epochs, output size, and temperature must be positive.")
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise AudioEmbeddingUnavailableError("CUDA was requested but PyTorch cannot access it.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    label_ids = {label: index for index, label in enumerate(sorted(set(labels)))}
    inputs = torch.from_numpy(matrix).to(resolved_device)
    targets = torch.tensor([label_ids[label] for label in labels], device=resolved_device)
    module = _make_projection_module(torch, matrix.shape[1], output_size).to(resolved_device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate)
    losses = []
    module.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        embeddings = torch.nn.functional.normalize(module(inputs), dim=-1)
        loss = _supervised_contrastive_loss(torch, embeddings, targets, temperature)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "input_size": int(matrix.shape[1]),
        "output_size": output_size,
        "state_dict": module.state_dict(),
        "trained": True,
        "training_provenance": training_provenance or {},
        "training_summary": {
            "example_count": matrix.shape[0],
            "concept_count": len(label_ids),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "seed": seed,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "device": resolved_device,
        },
    }
    torch.save(checkpoint, output_path)
    return checkpoint["training_summary"]


def _supervised_contrastive_loss(
    torch: Any, embeddings: Any, targets: Any, temperature: float
) -> Any:
    similarity = embeddings @ embeddings.T / temperature
    identity = torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
    positives = targets[:, None].eq(targets[None, :]) & ~identity
    valid = positives.sum(dim=1) > 0
    if not bool(valid.any()):
        raise ValueError("No positive pairs are available in the training batch.")
    logits = similarity.masked_fill(identity, float("-inf"))
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probability = torch.where(
        positives, log_probabilities, torch.zeros_like(log_probabilities)
    ).sum(dim=1)
    positive_log_probability /= positives.sum(dim=1).clamp(min=1)
    return -positive_log_probability[valid].mean()


class AudioEmbeddingIndex:
    """Normalized exact cosine index with an optional FAISS acceleration path."""

    def __init__(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
        provenance: dict[str, Any],
    ) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(metadata):
            raise ValueError("Vectors must be a 2-D matrix aligned with metadata.")
        if matrix.shape[0] == 0:
            raise ValueError("At least one reference pronunciation is required.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.all(np.isfinite(matrix)):
            raise ValueError("Reference vectors must be finite and non-zero.")
        self.vectors = matrix / norms
        self.metadata = metadata
        self.provenance = provenance
        self._last_search_backend = "not_searched"

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 20,
        filters: dict[str, str | list[str]] | None = None,
    ) -> list[AudioCandidate]:
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.vectors.shape[1] or not np.all(np.isfinite(vector)):
            raise ValueError("Query vector is invalid or has the wrong dimension.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("Query vector must be non-zero.")
        allowed = [index for index, item in enumerate(self.metadata) if _matches(item, filters)]
        if not allowed:
            return []
        query = (vector / norm).reshape(1, -1)
        subset = np.ascontiguousarray(self.vectors[allowed], dtype=np.float32)
        scores, order = _cosine_search(subset, query, min(top_k, len(allowed)))
        self._last_search_backend = "faiss_index_flat_ip" if _faiss_available() else "numpy"
        results = []
        for score, position in zip(scores, order, strict=True):
            item = self.metadata[allowed[position]]
            results.append(
                AudioCandidate(
                    reference_id=str(item["reference_id"]),
                    concept_id=str(item["concept_id"]),
                    term=str(item["term"]),
                    language=str(item["language"]),
                    score=round(float(score), 6),
                    metadata=item,
                )
            )
        return results

    def search_for_review(
        self,
        query: np.ndarray,
        *,
        top_k: int = 20,
        filters: dict[str, str | list[str]] | None = None,
    ) -> AudioSearchResponse:
        candidates = self.search(query, top_k=top_k, filters=filters)
        return AudioSearchResponse(
            decision="review" if candidates else "abstain",
            decision_reasons=(
                ["audio_candidates_require_phoneme_asr_and_human_review"]
                if candidates
                else ["no_audio_candidates_after_metadata_filtering"]
            ),
            embedding_provenance={
                **self.provenance,
                "search_backend": self._last_search_backend,
            },
            candidates=candidates,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                vectors=self.vectors,
                metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
                provenance=np.asarray(json.dumps(self.provenance, ensure_ascii=False)),
                schema_version=np.asarray(1, dtype=np.int64),
            )

    @classmethod
    def load(cls, path: Path) -> AudioEmbeddingIndex:
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("Unsupported audio embedding index schema.")
            return cls(
                data["vectors"],
                json.loads(str(data["metadata"])),
                json.loads(str(data["provenance"])),
            )


def build_reference_index(
    references: list[ReferencePronunciation], encoder: AudioEncoder
) -> AudioEmbeddingIndex:
    if not references:
        raise ValueError("Reference manifest is empty.")
    encode_many = getattr(encoder, "encode_many", None)
    if callable(encode_many):
        matrix = np.asarray(encode_many([reference.audio_path for reference in references]))
        vectors = [row for row in matrix]
    else:
        vectors = [encoder.encode(reference.audio_path) for reference in references]
    dimensions = {vector.shape for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("All reference embeddings must have the same dimension.")
    metadata = [
        reference.model_dump(mode="json", exclude={"audio_path"}) for reference in references
    ]
    return AudioEmbeddingIndex(np.stack(vectors), metadata, encoder.provenance)


def validate_query_encoder(index: AudioEmbeddingIndex, encoder: AudioEncoder) -> None:
    """Reject query/reference vectors made with materially different configurations."""
    expected = index.provenance
    actual = encoder.provenance
    fields = ("embedding_model", "pooling", "sample_rate", "projection_checkpoint")
    mismatches = [field for field in fields if expected.get(field) != actual.get(field)]
    if mismatches:
        raise ValueError(
            "Query encoder does not match the reference index: " + ", ".join(mismatches)
        )


def load_reference_manifest(path: Path) -> list[ReferencePronunciation]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            audio_path = Path(payload["audio_path"])
            if not audio_path.is_absolute():
                audio_path = path.parent / audio_path
            payload["audio_path"] = audio_path
            records.append(ReferencePronunciation.model_validate(payload))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid reference manifest record on line {line_number}.") from exc
    return records


def _matches(item: dict[str, Any], filters: dict[str, str | list[str]] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        values = expected if isinstance(expected, list) else [expected]
        if item.get(key) not in values:
            return False
    return True


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def _cosine_search(
    vectors: np.ndarray, query: np.ndarray, top_k: int
) -> tuple[list[float], list[int]]:
    try:
        import faiss
    except ImportError:
        scores = vectors @ query[0]
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [float(scores[index]) for index in order], [int(index) for index in order]
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    scores, positions = index.search(np.ascontiguousarray(query, dtype=np.float32), top_k)
    return scores[0].astype(float).tolist(), positions[0].astype(int).tolist()
