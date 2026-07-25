from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from medterm.normalization import normalize_term
from medterm.terminology import DictionaryIndex

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


class ChatCompletions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class CandidateCatalogItem(BaseModel):
    concept_id: str
    term: str
    language: str
    risk_tier: str
    provenance: str


class ProposedSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    concept_id: str
    candidate_term: str


class ModelProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: list[ProposedSpan] = Field(default_factory=list)


class ExperimentState(TypedDict, total=False):
    text: str
    locale: str | None
    catalog: list[dict[str, str]]
    raw_proposal: dict[str, Any]
    predictions: list[dict[str, Any]]
    rejected_predictions: list[dict[str, Any]]


def build_deepseek_client() -> ChatCompletions:
    """Build a traced OpenAI-compatible client without copying the key to another variable."""
    api_key = os.getenv("deep-seek-api")
    if not api_key:
        raise RuntimeError("environment variable 'deep-seek-api' is required")
    from langsmith import wrappers

    client = OpenAI(api_key=api_key, base_url=DEFAULT_DEEPSEEK_BASE_URL)
    return wrappers.wrap_openai(client).chat.completions


def build_correction_graph(
    index: DictionaryIndex,
    *,
    completions: ChatCompletions | None = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
) -> Any:
    """Compile the synthetic-only LLM correction experiment.

    The graph can select only controlled-dictionary entries and preserves model suggestions
    separately from validated output. It is deliberately not wired into the API service.
    """
    completion_client = completions or build_deepseek_client()

    def prepare_catalog(state: ExperimentState) -> dict[str, Any]:
        language = (state.get("locale") or "und").split("-")[0].split("_")[0].casefold()
        entries = [entry for entry in index.artifact.entries if entry.language == language]
        if not entries:
            entries = [entry for entry in index.artifact.entries if entry.language == "en"]
        catalog = [
            CandidateCatalogItem(
                concept_id=entry.concept_id,
                term=entry.term,
                language=entry.language,
                risk_tier=entry.risk_tier,
                provenance=entry.provenance,
            ).model_dump()
            for entry in entries
        ]
        return {"catalog": catalog}

    def propose(state: ExperimentState) -> dict[str, Any]:
        system_prompt = (
            "You assist a human reviewer with synthetic medical-term transcription tests. "
            "Never diagnose, prescribe, or rewrite source text. Identify only a likely corrupted "
            "medical-term span and select only from the supplied controlled catalog. Exact correct "
            "terms without corruption must produce no predictions. Return JSON only, shaped as "
            '{"predictions":[{"char_start":0,"char_end":3,"concept_id":"id",'
            '"candidate_term":"term"}]}. Character offsets are Python-style half-open offsets '
            "into the exact source string. If uncertain, return {\"predictions\":[]}."
        )
        user_payload = {
            "text": state["text"],
            "locale": state.get("locale"),
            "controlled_catalog": state["catalog"],
        }
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = completion_client.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": "Analyze this JSON input and return JSON output:\n"
                            + json.dumps(user_payload, ensure_ascii=False),
                        },
                    ],
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                    temperature=0,
                    max_tokens=800,
                )
                content = response.choices[0].message.content or ""
                return {"raw_proposal": ModelProposal.model_validate_json(content).model_dump()}
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
        raise ValueError(
            "DeepSeek returned no valid correction JSON after two attempts"
        ) from last_error

    def validate(state: ExperimentState) -> dict[str, Any]:
        text = state["text"]
        catalog = {
            (item["concept_id"], normalize_term(item["term"], item["language"])): item
            for item in state["catalog"]
        }
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        occupied: list[tuple[int, int]] = []
        proposal = ModelProposal.model_validate(state.get("raw_proposal", {}))
        for prediction in proposal.predictions:
            normalized_candidate = normalize_term(prediction.candidate_term)
            catalog_item = catalog.get((prediction.concept_id, normalized_candidate))
            valid_offsets = 0 <= prediction.char_start < prediction.char_end <= len(text)
            overlaps = any(
                prediction.char_start < end and prediction.char_end > start
                for start, end in occupied
            )
            observed = text[prediction.char_start : prediction.char_end] if valid_offsets else ""
            is_unchanged = bool(catalog_item) and normalize_term(
                observed, catalog_item["language"]
            ) == normalize_term(catalog_item["term"], catalog_item["language"])
            if catalog_item is None or not valid_offsets or overlaps or is_unchanged:
                rejected.append(prediction.model_dump())
                continue
            accepted.append(
                {
                    **prediction.model_dump(),
                    "span_text": observed,
                    "language": catalog_item["language"],
                    "risk_tier": catalog_item["risk_tier"],
                    "provenance": catalog_item["provenance"],
                    "decision": "review",
                    "decision_reason": [
                        "experimental_llm_dictionary_selection",
                        "human_review_required",
                    ],
                }
            )
            occupied.append((prediction.char_start, prediction.char_end))
        return {"predictions": accepted, "rejected_predictions": rejected}

    builder = StateGraph(ExperimentState)
    builder.add_node("prepare_catalog", prepare_catalog)
    builder.add_node("deepseek_propose", propose)
    builder.add_node("validate_controlled_output", validate)
    builder.add_edge(START, "prepare_catalog")
    builder.add_edge("prepare_catalog", "deepseek_propose")
    builder.add_edge("deepseek_propose", "validate_controlled_output")
    builder.add_edge("validate_controlled_output", END)
    return builder.compile()


def graph_target(graph: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        result = graph.invoke({"text": inputs["text"], "locale": inputs.get("locale")})
        return {
            "predictions": result.get("predictions", []),
            "rejected_predictions": result.get("rejected_predictions", []),
            "auto_commit_enabled": False,
        }

    return target
