#!/usr/bin/env python
import argparse
import random
import tempfile
import time

from midden_analysis import EstimatorPrecision, HeapDumpExplorer

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Use real heap dumps to benchmark path finding"
    )
    arg_parser.add_argument("dump_path", help="Path to the heap dump to load")
    arg_parser.add_argument(
        "--no-use-sketches",
        action="store_false",
        help="Don't use set membership sketches during path finding",
        dest="use_sketches",
    )
    args = arg_parser.parse_args()

    with tempfile.TemporaryDirectory(suffix=".lmdb") as temp_dir:
        print(f"Loading heap dump from {args.dump_path}...")
        explorer = HeapDumpExplorer(temp_dir)
        with open(args.dump_path, "rb") as f:
            explorer.import_lines(
                f,
                estimator_precision=EstimatorPrecision.Exact
                if args.use_sketches
                else EstimatorPrecision.NoEstimates,
            )

        all_ids = [obj.id for obj in explorer.get_objects_by_type("All Types", None)]

        print("Finding random paths...")
        start_time = time.monotonic()
        successes = 0
        for i in range(1000):
            if i % 100 == 0:
                print(f"Finding path {i}...")
            start = random.choice(all_ids)
            end = random.choice(all_ids)
            successes += bool(explorer.find_path_between_objects(start, end, set()))
        elapsed = time.monotonic() - start_time
        print(
            f"Finding 1000 random paths took {elapsed:.2f} seconds ({'with' if args.use_sketches else 'without'} sketches). There were {successes} successful finds."
        )
