"""Recompute the paper's results from the supplied experimental evidence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def verify_files():
    checked = set()
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or name in checked or not path.is_file():
            raise ValueError('Missing, duplicate or invalid evidence path: ' + name)
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('Changed evidence file: ' + name)
        checked.add(name)
    return len(checked)


def main():
    if not __debug__:
        raise RuntimeError('Run without -O: the scientific checks use assertions.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Include SciFact vector and ANN score replay (requires NumPy).')
    parser.add_argument('--output', type=Path, help='Save the recomputed results to a new JSON file.')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Choose a new output filename; existing results are not overwritten.')
    count = verify_files()
    names = ['upstream_update', 'migration', 'reader_repair', 'equal_access', 'attribution', 'precision', 'geometry', 'backend_updates', 'policy_refresh', 'access_sql', 'access_qdrant', 'score_retention', 'extensions', 'paper_results']
    if args.all:
        names += ['scifact', 'ann']
    results = {}
    for name in names:
        path = ROOT / 'scripts' / (name + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        results[name] = module.check(ROOT)
        if not results[name]['passed']:
            raise ValueError('Experiment verification failed: ' + name)
        print('PASS ' + name, flush=True)
    report = {'passed': True, 'verified_files': count, 'experiments': results,
              'scope': 'Recomputation from supplied observations and stored vectors; no new model or service collection.'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Passed {len(results)} experiment checks; {count} files verified.')


if __name__ == '__main__':
    main()
