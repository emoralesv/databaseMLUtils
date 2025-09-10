from __future__ import annotations

from databaseMLUtils.cli import build_parser


if __name__ == "__main__":
    # Example usage via CLI parser
    parser = build_parser()
    print("Example: dbutils convert --task det2cls --ann instances.json --images imgs --out results/crops")

