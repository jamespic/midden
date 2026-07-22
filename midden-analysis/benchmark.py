#!/usr/bin/env python
import resource
import tempfile
import time
from argparse import ArgumentParser

from midden_analysis import EstimatorPrecision, HeapDumpExplorer

if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark Midden's query performance")
    parser.add_argument("dump_path", help="Path to the heap dump to load")
    parser.add_argument(
        "--estimator-precision",
        choices=["no_estimates", "low", "medium", "high", "exact"],
        default="medium",
        help="Precision level for size estimates (default: medium)",
    )
    args = parser.parse_args()

    precision_map = {
        "no_estimates": EstimatorPrecision.NoEstimates,
        "low": EstimatorPrecision.Low,
        "medium": EstimatorPrecision.Medium,
        "high": EstimatorPrecision.High,
        "exact": EstimatorPrecision.Exact,
    }
    estimator_precision = precision_map[args.estimator_precision]

    with tempfile.TemporaryDirectory(suffix=".lmdb") as temp_dir:
        print(
            f"Loading heap dump from {args.dump_path} with estimator precision {args.estimator_precision}..."
        )
        start_time = time.time()

        explorer = HeapDumpExplorer(temp_dir)
        with open(args.dump_path, "rb") as f:
            explorer.import_lines(f, estimator_precision)

        elapsed_time = time.time() - start_time
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(
            f"Loading {args.dump_path} at precision {args.estimator_precision} took {elapsed_time:.2f} seconds and used {mem_usage:.2f} MB of memory"
        )
