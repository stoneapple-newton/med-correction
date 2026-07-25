from medterm.matcher import MedicalTermMatcher
from medterm.models import MatchRequest, NBestHypothesis, TokenMetadata


def test_detects_asr_like_medication_confusion_and_forces_review(
    matcher: MedicalTermMatcher,
) -> None:
    text = "The patient takes met for men 500 mg with breakfast."
    response = matcher.match(MatchRequest(text=text, locale="en-US", top_k=5))
    assert response.auto_commit_enabled is False
    assert response.spans
    span = next(item for item in response.spans if item.candidates[0].concept_id == "RxCUI:6809")
    assert "met for men" in span.span_text
    assert span.candidates[0].term == "metformin"
    assert span.decision == "review"
    assert "dose_adjacent" in span.decision_reason
    assert span.char_start == text.index("met")
    assert span.artifact_version == "test_v1"
    assert span.candidates[0].score_breakdown.phonetic > 0


def test_does_not_flag_already_correct_term_without_asr_evidence(
    matcher: MedicalTermMatcher,
) -> None:
    response = matcher.match(MatchRequest(text="Continue metformin daily.", locale="en-US"))
    assert response.spans == []


def test_uses_nbest_disagreement_as_detection_evidence(matcher: MedicalTermMatcher) -> None:
    response = matcher.match(
        MatchRequest(
            text="Start clone azepam tonight",
            locale="en-US",
            n_best=[
                NBestHypothesis(text="Start clonazepam tonight"),
                NBestHypothesis(text="Start clone as a pam tonight"),
            ],
        )
    )
    assert response.spans
    assert any(span.evidence.nbest_disagreement > 0 for span in response.spans)


def test_carries_asr_offsets_into_audit_output(matcher: MedicalTermMatcher) -> None:
    text = "met for men"
    response = matcher.match(
        MatchRequest(
            text=text,
            locale="en-US",
            tokens=[
                TokenMetadata(
                    text="met", char_start=0, char_end=3, confidence=0.7, start_time=1, end_time=1.2
                ),
                TokenMetadata(
                    text="for",
                    char_start=4,
                    char_end=7,
                    confidence=0.6,
                    start_time=1.2,
                    end_time=1.4,
                ),
                TokenMetadata(
                    text="men",
                    char_start=8,
                    char_end=11,
                    confidence=0.7,
                    start_time=1.4,
                    end_time=1.8,
                ),
            ],
        )
    )
    assert response.spans[0].audio_start == 1.0
    assert response.spans[0].audio_end == 1.8
