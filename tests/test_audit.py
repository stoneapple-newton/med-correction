from medterm.audit import AuditStore
from medterm.matcher import MedicalTermMatcher
from medterm.models import MatchRequest, ReviewRequest


def test_round_trips_match_and_review(audit: AuditStore, matcher: MedicalTermMatcher) -> None:
    response = matcher.match(MatchRequest(text="met for men", locale="en-US"))
    audit.save_match(response)
    loaded = audit.get_match(response.request_id)
    assert loaded is not None
    assert loaded.model_dump() == response.model_dump()
    span = response.spans[0]
    review = audit.save_review(
        response.request_id,
        span.span_id,
        ReviewRequest(
            reviewer_id="clinician-7",
            action="accept_candidate",
            concept_id=span.candidates[0].concept_id,
        ),
        response.artifact_version,
    )
    assert review.reviewer_id == "clinician-7"
    assert audit.list_reviews(response.request_id)[0].review_id == review.review_id
