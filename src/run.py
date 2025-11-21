# run_experiment.py
"""CLI entry point for benchmarking TabICL/TabPFN on FB15k-237."""

from experiment import parse_args, run_experiment


def main() -> None:
    """Parse CLI args, train the requested model, and print metrics."""
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
