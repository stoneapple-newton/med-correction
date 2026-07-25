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
