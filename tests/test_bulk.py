from __future__ import annotations

import json
from pathlib import Path

import pytest

from medterm.bulk import finalize_dictionary, import_rxnorm_release, register_source
from medterm.terminology import load_artifact


def _rrf_row(
    concept_id: str,
    term: str,
    *,
    preferred: bool = False,
    tty: str = "IN",
    suppress: str = "N",
) -> str:
    values = [
        concept_id,
        "ENG",
        "P",
        f"L{concept_id}",
        "PF",
        f"S{concept_id}",
        "Y" if preferred else "N",
        f"A{concept_id}-{term}",
        "",
        "",
        "",
        "RXNORM",
        tty,
        concept_id,
        term,
        "0",
        suppress,
        "",
        "",
    ]
    return "|".join(values)


def _register(tmp_path: Path, source: Path) -> Path:
    register_source(
        source,
        tmp_path / "builds",
        release_id="rxnorm-test",
        source_release="2026-07-test",
        license_status="approved_for_local_processing",
        license_approval_id="synthetic-test-approval",
        license_owner="test-owner",
        license_evidence="synthetic fixture only",
        license_approval_date="2026-07-26",
    )
    return tmp_path / "builds" / "rxnorm-test" / "source" / "source_manifest.json"


def test_streaming_import_preserves_aliases_across_checkpoints_and_is_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "RXNCONSO.RRF"
    source.write_text(
        "\n".join(
            [
                _rrf_row("2", "Glucophage"),
                _rrf_row("1", "Lopressor"),
                _rrf_row("2", "metformin", preferred=True),
                _rrf_row("1", "metoprolol", preferred=True),
                _rrf_row("3", "suppressed", preferred=True, suppress="Y"),
                "malformed|row",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = _register(tmp_path, source)
    output = tmp_path / "builds" / "rxnorm-test" / "import"

    summary = import_rxnorm_release(
        manifest,
        output,
        concept_batch_size=1,
        parse_checkpoint_rows=2,
    )

    assert summary.partial is False
    assert summary.counts["emitted_records"] == 2
    assert summary.counts["suppressed"] == 1
    assert summary.counts["rejected"] == 1
    first = json.loads((output / "batches" / "batch-000001.jsonl").read_text())
    second = json.loads((output / "batches" / "batch-000002.jsonl").read_text())
    assert [first["concept_id"], second["concept_id"]] == ["RxCUI:1", "RxCUI:2"]
    assert second["term"] == "metformin"
    assert second["aliases"] == ["Glucophage"]
    assert second["risk_tier"] == "unclassified"
    resumed = import_rxnorm_release(
        manifest,
        output,
        concept_batch_size=1,
        parse_checkpoint_rows=2,
        resume=True,
    )
    assert resumed.shard_checksums == summary.shard_checksums


def test_partial_import_scans_later_aliases_and_cannot_be_reused_as_full(tmp_path: Path) -> None:
    source = tmp_path / "RXNCONSO.RRF"
    source.write_text(
        "\n".join(
            [
                _rrf_row("1", "metformin", preferred=True),
                _rrf_row("2", "metoprolol", preferred=True),
                _rrf_row("1", "Glucophage"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = _register(tmp_path, source)
    output = tmp_path / "builds" / "rxnorm-test" / "import"
    summary = import_rxnorm_release(manifest, output, max_concepts=1)

    record = json.loads((output / "batches" / "batch-000001.jsonl").read_text())
    assert summary.partial is True
    assert record["aliases"] == ["Glucophage"]
    with pytest.raises(ValueError, match="stage hash"):
        import_rxnorm_release(manifest, output, resume=True)


def test_finalization_validates_checksums_and_reconciles_inventory(tmp_path: Path) -> None:
    source = tmp_path / "RXNCONSO.RRF"
    source.write_text(_rrf_row("1", "metformin", preferred=True) + "\n", encoding="utf-8")
    manifest = _register(tmp_path, source)
    import_dir = tmp_path / "builds" / "rxnorm-test" / "import"
    import_rxnorm_release(manifest, import_dir)

    result = finalize_dictionary(import_dir, tmp_path / "terminology", version="rxnorm-test-v1")
    artifact = load_artifact(tmp_path / "terminology" / "dictionary.json")

    assert result["record_count"] == 1
    assert result["inventory"]["risk_tiers"] == {"unclassified": 1}
    assert result["auto_commit_enabled"] is False
    assert artifact.entries[0].source_record_type == "IN"
    assert artifact.entries[0].licensing_status == "approved_for_local_processing"


def test_registration_rejects_unapproved_license(tmp_path: Path) -> None:
    source = tmp_path / "RXNCONSO.RRF"
    source.write_text("synthetic", encoding="utf-8")
    with pytest.raises(ValueError, match="license"):
        register_source(
            source,
            tmp_path / "builds",
            release_id="rxnorm-test",
            source_release="test",
            license_status="pending",
            license_approval_id="pending",
            license_owner="test",
            license_evidence="none",
            license_approval_date="2026-07-26",
        )
