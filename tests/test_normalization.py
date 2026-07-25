from medterm.normalization import (
    detect_scripts,
    extract_medication_fields,
    normalize_text,
    strip_structured_tokens,
    tokenize_with_offsets,
)


def test_normalization_is_loss_minimizing_and_extracts_medication_fields() -> None:
    source = "  Metformin—500 mg  PO Tablet  "
    assert normalize_text(source) == "metformin-500 mg po tablet"
    fields = extract_medication_fields(source)
    assert fields.strengths == ["500mg"]
    assert fields.routes == ["po"]
    assert fields.dosage_forms == ["tablet"]
    assert strip_structured_tokens(source) == "metformin"


def test_offsets_reference_original_text() -> None:
    source = "Take met for men today."
    tokens = tokenize_with_offsets(source)
    assert source[tokens[1].start : tokens[3].end] == "met for men"


def test_script_detection_preserves_mixed_script_evidence() -> None:
    assert detect_scripts("metformin") == ["Latn"]
    assert detect_scripts("metформин") == ["Cyrl", "Latn"]
