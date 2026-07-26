"""Run the local synthetic language-context retrieval comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medterm.context_rag_experiment import load_context_rag_dataset, run_context_rag_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/context_rag_dummy_v1.json")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_context_rag_experiment(load_context_rag_dataset(args.dataset))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
