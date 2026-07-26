from pathlib import Path

from medterm.terminology import DictionaryIndex, build_artifact, load_artifact


def test_builds_reproducible_versioned_artifact(tmp_path: Path, source_path: Path) -> None:
    output = tmp_path / "dictionary.json"
    artifact = build_artifact(source_path, output, version="dict_test")
    loaded = load_artifact(output)
    assert loaded.artifact_version == "dict_test"
    assert loaded.source_sha256 == artifact.source_sha256
    assert len(loaded.entries) >= 25
    index = DictionaryIndex(loaded)
    assert index.is_exact("metformin")
    assert index.is_exact("glucophage")
    assert not index.is_exact("met for men")


def test_bounded_mutation_channel_is_language_filtered(index: DictionaryIndex) -> None:
    english = index.retrieve("metformn", [], languages=["en"])
    spanish = index.retrieve("metformn", [], languages=["es"])

    metformin_entry = next(
        entry_id
        for entry_id, entry in enumerate(index.artifact.entries)
        if entry.concept_id == "RxCUI:6809" and entry.language == "en"
    )
    assert english[metformin_entry]["mutation"] > 0.8
    assert "mutation" not in spanish.get(metformin_entry, {})
