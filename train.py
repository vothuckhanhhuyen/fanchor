import os
import sys
import time

from datasets import get_dataset
from evaluate import EVALUATE
from models import get_method
from utils import append_result, matching_results, parse, set_seed


def main():
    """Train one (dataset, method, train_seed) run, evaluate it and append the record to results_path."""
    dataset, method_name, defaults, args = parse(sys.argv[1:])

    # skip if a matching config already has a result for this seed
    if any(r['config'].get('train_seed') == args.train_seed for r in matching_results(args.results_path, args, defaults)):
        print(f'already done, skipping: train_seed={args.train_seed}', flush=True)
        return
    set_seed(args.train_seed)
    t0 = time.time()

    # build data and model, train, then run the dataset's evaluation
    Method = get_method(method_name)
    data = get_dataset(dataset)(args, views=Method.uses_views)
    method = Method(args, data)
    record = method.fit(data)
    record['train_sec'] = time.time() - t0
    record.update(EVALUATE[dataset](method, data))

    # save config + metrics as one json record
    append_result(args.results_path, {'config': vars(args),
                                      'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                                      'elapsed_sec': time.time() - t0,
                                      **record})
    print(f'done in {time.time() - t0:.1f}s -> {args.results_path}', flush=True)


if __name__ == '__main__':
    main()
