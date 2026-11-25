import sys

import numpy as np

from evaluate import SUMMARY_GROUPS
from utils import MATCH_EXCLUDE, append_result, matching_results, parse


def main():
    """Aggregate mean/std over summary_seeds for one config and append the summary record to summary_path."""
    dataset, method, defaults, args = parse(sys.argv[1:])
    summary_path = args.summary_path or f'{args.results_path}_summary'

    # skip if this config is already summarized over the same seeds
    if any(r.get('seeds') == args.summary_seeds for r in matching_results(summary_path, args, defaults)):
        sys.exit('already summarized')

    # one run per requested seed; abort if any is missing
    records = {}
    for r in matching_results(args.results_path, args, defaults):
        if r['config']['train_seed'] in args.summary_seeds:
            records[r['config']['train_seed']] = r

    missing = sorted(set(args.summary_seeds) - set(records))
    if missing:
        sys.exit(f'missing seeds {missing} in {args.results_path}')
    runs = [records[s] for s in args.summary_seeds]

    config = {k: v for k, v in vars(args).items() if k not in MATCH_EXCLUDE}
    summary = {'config': config, 'seeds': args.summary_seeds, 'n_seeds': len(args.summary_seeds)}

    # mean/std of every metric in each group, skipping groups absent from the records
    for name, path in SUMMARY_GROUPS[dataset].items():
        if not all(path.split('.')[0] in r for r in runs):
            continue
        values = []
        for r in runs:
            for key in path.split('.'):
                r = r[key]
            values.append(r)
        summary[f'{name}_mean'] = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
        summary[f'{name}_std'] = {k: float(np.std([v[k] for v in values])) for k in values[0]}

    for key, value in summary.items():
        if key.endswith('_mean'):
            print(key, value)

    append_result(summary_path, summary)
    print(f'-> {summary_path}')


if __name__ == '__main__':
    main()
