import json
from types import SimpleNamespace

from medterm.llm_experiment import build_correction_graph, graph_target


class FakeCompletions:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def create(self, **kwargs):
        content = json.dumps(self.payload, ensure_ascii=False)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_graph_accepts_only_controlled_candidate_and_preserves_source(index) -> None:
    text = "记录二甲双"
    graph = build_correction_graph(
        index,
        completions=FakeCompletions(
            {
                "predictions": [
                    {
                        "char_start": 2,
                        "char_end": 5,
                        "concept_id": "RxCUI:6809",
                        "candidate_term": "二甲双胍",
                    }
                ]
            }
        ),
    )

    output = graph_target(graph)({"text": text, "locale": "zh-CN"})

    assert output["predictions"][0]["span_text"] == "二甲双"
    assert text == "记录二甲双"
    assert output["predictions"][0]["decision"] == "review"
    assert output["auto_commit_enabled"] is False


def test_graph_rejects_out_of_catalog_and_unchanged_terms(index) -> None:
    graph = build_correction_graph(
        index,
        completions=FakeCompletions(
            {
                "predictions": [
                    {
                        "char_start": 0,
                        "char_end": 4,
                        "concept_id": "invented",
                        "candidate_term": "invented",
                    },
                    {
                        "char_start": 0,
                        "char_end": 4,
                        "concept_id": "RxCUI:6809",
                        "candidate_term": "二甲双胍",
                    },
                ]
            }
        ),
    )

    output = graph_target(graph)({"text": "二甲双胍", "locale": "zh-CN"})

    assert output["predictions"] == []
    assert len(output["rejected_predictions"]) == 2
