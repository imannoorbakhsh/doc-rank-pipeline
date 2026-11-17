from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import PipelineConfig
from .evaluation import evaluate
from .models import Query
from .pipeline import DocumentRankPipeline

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data" / "sample_documents.json"
DEFAULT_GOLDEN = ROOT / "evals" / "golden_placeholder.json"
DEFAULT_CONFIG = ROOT / "configs" / "default.json"


def load_queries(path: str | Path) -> list[Query]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        Query(
            query_id=item["query_id"],
            text=item["text"],
            category=item["category"],
            relevant_chunk_ids=frozenset(item["relevant_chunk_ids"]),
        )
        for item in payload["queries"]
    ]


def evaluate_config(config_path: str | Path, corpus: Path = DEFAULT_CORPUS, golden: Path = DEFAULT_GOLDEN) -> dict[str, object]:
    config = PipelineConfig.from_path(config_path)
    result = evaluate(DocumentRankPipeline.from_corpus(config, corpus), load_queries(golden))
    return {"config": config.name, **result}


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a lightweight document-ranking pipeline.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subcommands.add_parser("evaluate")
    evaluate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    evaluate_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    evaluate_parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    evaluate_parser.add_argument("--output", type=Path)
    compare_parser = subcommands.add_parser("compare")
    compare_parser.add_argument("--output", type=Path)
    regression_parser = subcommands.add_parser("check-regression")
    regression_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    regression_parser.add_argument("--baseline", type=Path, default=ROOT / "evals" / "baseline.json")
    regression_parser.add_argument("--maximum-drop", type=float, default=0.01)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.command == "evaluate":
        result = evaluate_config(args.config, args.corpus, args.golden)
    elif args.command == "compare":
        result = {"comparisons": [evaluate_config(ROOT / "configs" / name) for name in ("default.json", "dense.json", "small_chunks.json")]}
    else:
        result = evaluate_config(args.config)
        with args.baseline.open(encoding="utf-8") as handle:
            baseline = json.load(handle)
        current = float(result["overall"]["recall@10"])
        expected = float(baseline["overall"]["recall@10"])
        drop = expected - current
        result["regression_check"] = {"baseline_recall@10": expected, "current_recall@10": current, "drop": drop, "maximum_drop": args.maximum_drop}
        if drop > args.maximum_drop:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
