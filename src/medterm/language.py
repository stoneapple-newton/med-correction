from __future__ import annotations

from pathlib import Path
from typing import Any

from medterm.normalization import detect_scripts


class LanguageIdentifier:
    """fastText lid.176 adapter with an explicit, deterministic script fallback."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model: Any | None = None
        if model_path and model_path.exists():
            try:
                import fasttext  # type: ignore[import-not-found]

                self.model = fasttext.load_model(str(model_path))
            except (ImportError, OSError, ValueError):
                self.model = None

    def detect(self, text: str, locale: str | None = None) -> tuple[str, float, list[str]]:
        if locale:
            language = locale.split("-")[0].split("_")[0].casefold()
            return language, 1.0, []
        if self.model is not None:
            labels, probabilities = self.model.predict(text.replace("\n", " "), k=2)
            languages = [label.removeprefix("__label__") for label in labels]
            return languages[0], float(probabilities[0]), languages[1:]
        scripts = detect_scripts(text)
        script_defaults = {
            "Cyrl": "ru",
            "Arab": "ar",
            "Deva": "hi",
            "Hani": "zh",
            "Hira": "ja",
            "Kana": "ja",
            "Hang": "ko",
            "Hebr": "he",
            "Latn": "en",
        }
        languages = list(dict.fromkeys(script_defaults.get(script, "und") for script in scripts))
        return languages[0], 0.55 if len(languages) == 1 else 0.35, languages[1:]
