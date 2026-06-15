"""Shared CLI arguments for imprecise-clustering experiments."""
from pathlib import Path


DEFAULT_METHODS = (
    "matched,original,feature_bucket,kmeans,random,fcm,pcm,rough"
)


def add_common_arguments(parser, default_output):
    parser.add_argument(
        "--methods",
        default=DEFAULT_METHODS,
        help=(
            "Comma-separated controls and clustering methods. "
            "ECM is optional and requires a validated evclust installation."
        ),
    )
    parser.add_argument(
        "--gen-methods",
        default="original,feature_bucket",
        help="Comma-separated reward-generating partitions",
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=3000)
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--n-actions", type=int, default=1000)
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--dim-context", type=int, default=10)
    parser.add_argument("--n-cat-dim", type=int, default=10)
    parser.add_argument("--n-cat-per-dim", type=int, default=5)
    parser.add_argument("--n-unobserved-cat-dim", type=int, default=2)
    parser.add_argument("--reward-alpha", default="0.4,0.3,0.2,0.1")
    parser.add_argument("--feature-nonlinearity", type=float, default=0.25)
    parser.add_argument("--reward-scale", type=float, default=3.0)
    parser.add_argument("--reward-std", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=-0.1)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--fuzzifier", type=float, default=2.0)
    parser.add_argument("--cluster-max-iter", type=int, default=150)
    parser.add_argument("--cluster-tolerance", type=float, default=1e-5)
    parser.add_argument("--rough-ratio", type=float, default=1.25)
    parser.add_argument("--environment-seed", type=int, default=12345)
    parser.add_argument("--sample-seed", type=int, default=50000)
    parser.add_argument("--model-seed", type=int, default=24680)
    parser.add_argument("--cluster-seed", type=int, default=13579)
    parser.add_argument("--out-dir", default=default_output)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--quick", action="store_true")


def apply_quick_mode(args, suffix):
    if not args.quick:
        return
    args.n_seeds = 2
    args.n_rounds = 600
    args.n_users = 20
    args.n_actions = 40
    args.n_clusters = 4
    args.dim_context = 6
    args.n_cat_dim = 5
    args.n_cat_per_dim = 3
    args.n_unobserved_cat_dim = 1
    args.methods = "matched,kmeans,fcm,pcm,rough"
    args.gen_methods = "feature_bucket"
    args.reward_std = 1.0
    args.reward_scale = 2.0
    args.out_dir = str(Path(args.out_dir) / suffix)


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]

