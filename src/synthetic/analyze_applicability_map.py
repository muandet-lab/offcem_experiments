"""Train applicability rules on declared benchmark outputs and test held-out data."""
import argparse

from applicability.modeling import fit_applicability_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit held-out OffCEM applicability models"
    )
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline", default="DR")
    parser.add_argument("--minimum-relative-effect", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = fit_applicability_models(
        train_csvs=args.train,
        test_csvs=args.test,
        output_dir=args.output_dir,
        baseline=args.baseline,
        minimum_relative_effect=args.minimum_relative_effect,
    )
    print(
        f"held-out {args.baseline}: "
        f"logistic Brier={metrics['logistic']['brier_score']:.4f}, "
        f"tree Brier={metrics['tree']['brier_score']:.4f}"
    )


if __name__ == "__main__":
    main()

