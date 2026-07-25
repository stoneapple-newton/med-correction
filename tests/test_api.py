from __future__ import annotations

from fastapi.testclient import TestClient

from medterm.api import AppServices, create_app, get_services
from medterm.config import Settings


def client_for(settings: Settings) -> TestClient:
    app = create_app()
    services = AppServices(settings)
    app.dependency_overrides[get_services] = lambda: services
    return TestClient(app)


def test_health_match_fetch_and_review(settings: Settings) -> None:
    with client_for(settings) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["auto_commit_enabled"] is False

        search = client.get("/v1/dictionary/search", params={"q": "Coumadin"})
        assert search.status_code == 200
        assert search.json()[0]["concept_id"] == "RxCUI:11289"

        match = client.post(
            "/v1/match", json={"text": "She takes met for men 500 mg", "locale": "en-US"}
        )
        assert match.status_code == 200
        payload = match.json()
        assert payload["spans"]

        fetched = client.get(f"/v1/matches/{payload['request_id']}")
        assert fetched.json() == payload

        span = payload["spans"][0]
        review = client.post(
            f"/v1/matches/{payload['request_id']}/spans/{span['span_id']}/review",
            json={"reviewer_id": "qa-1", "action": "keep_original"},
        )
        assert review.status_code == 201
        assert review.json()["action"] == "keep_original"


def test_audio_endpoint_accepts_supplied_transcript(settings: Settings) -> None:
    with client_for(settings) as client:
        response = client.post(
            "/v1/match/audio",
            data={"transcript": "met for men", "locale": "en-US"},
            files={"audio": ("clip.wav", b"RIFF-demo", "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json()["spans"]


def test_audio_without_transcript_explains_unconfigured_asr(settings: Settings) -> None:
    with client_for(settings) as client:
        response = client.post(
            "/v1/match/audio", files={"audio": ("clip.wav", b"RIFF-demo", "audio/wav")}
        )
        assert response.status_code == 503
        assert "MEDTERM_WHISPER_MODEL" in response.json()["detail"]
