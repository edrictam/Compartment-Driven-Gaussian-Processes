"""Reproduce all paper analyses from source data in fresh notebook processes.

Run from any directory: python /path/to/code/reproduce.py --output /tmp/pkgp-fresh
The output directory must be new or empty. Stored notebook outputs are cleared
before execution, and no earlier computational results are read.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

CODE_DIR = Path(__file__).resolve().parent
NOTEBOOKS = (
    'simulations.ipynb',
    'warfarin_experiment.ipynb',
    'theophylline_experiment.ipynb',
    'chemo_experiment.ipynb',
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_notebook(source, target):
    """Execute ordinary Python cells in order, preserving displayed results."""
    import nbformat
    from nbformat.v4 import new_output

    nb = nbformat.read(source, as_version=4)
    nb.metadata['language_info'] = {
        'name': 'python', 'version': platform.python_version(),
        'file_extension': '.py', 'mimetype': 'text/x-python',
    }
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            cell.metadata = {}
    namespace = {'__name__': '__main__'}
    try:
        count = 0
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            count += 1
            cell.execution_count = count
            stdout, stderr = io.StringIO(), io.StringIO()
            filename = f'{source.name}:cell-{index + 1}'
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    tree = ast.parse(cell.source, filename=filename)
                    expression = tree.body.pop().value if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                    exec(compile(tree, filename, 'exec'), namespace)
                    value = eval(compile(ast.Expression(expression), filename, 'eval'), namespace) if expression is not None else None
                    if value is not None:
                        cell.outputs.append(new_output(
                            'execute_result', execution_count=count,
                            data={'text/plain': repr(value)}, metadata={}))
            except BaseException as error:
                cell.outputs.append(new_output(
                    'error', ename=type(error).__name__, evalue=str(error),
                    traceback=traceback.format_exc().splitlines()))
                raise
            finally:
                streams = []
                for name, stream in [('stdout', stdout), ('stderr', stderr)]:
                    text = stream.getvalue()
                    if text:
                        streams.append(new_output('stream', name=name, text=text))
                        print(text, end='', file=sys.stdout if name == 'stdout' else sys.stderr, flush=True)
                cell.outputs = streams + cell.outputs
    finally:
        nbformat.write(nb, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New or empty output directory for the complete run.')
    parser.add_argument('--jobs', type=int, default=3, help='Independent notebook processes (default: 3).')
    parser.add_argument('--list', action='store_true', help='List notebooks in the full workflow without running them.')
    parser.add_argument('--worker', nargs=2, metavar=('SOURCE', 'TARGET'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        execute_notebook(Path(args.worker[0]), Path(args.worker[1]))
        return
    if args.list:
        print('\n'.join(NOTEBOOKS))
        return
    if args.output is None:
        parser.error('--output is required; choose a new or empty directory.')
    if args.jobs < 1:
        parser.error('--jobs must be positive.')
    output = args.output.expanduser().resolve()
    if output == CODE_DIR or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        parser.error('Output must be a new or empty directory; existing results are never overwritten.')
    missing = [name for name in NOTEBOOKS if not (CODE_DIR / name).is_file()]
    if missing:
        parser.error(f'Missing notebooks: {missing}')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('results', 'figures', 'notebooks', 'logs'):
        (output / name).mkdir()
    env = os.environ.copy()
    env.update(PKGP_CODE_DIR=str(CODE_DIR), PKGP_OUTPUT_DIR=str(output), MPLBACKEND='Agg')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = '1'
    manifest = {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
        'python': sys.version, 'platform': platform.platform(), 'jobs': args.jobs,
        'packages': {name: version(name) for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'pyreadr', 'nbformat')},
        'source_sha256': {p.name: sha256(p) for p in sorted(CODE_DIR.iterdir())
                          if p.is_file() and p.suffix in ('.py', '.ipynb', '.txt', '.json')},
        'data_sha256': {str(p.relative_to(CODE_DIR)): sha256(p)
                        for p in sorted((CODE_DIR / 'data').rglob('*')) if p.is_file()},
        'cache_reuse': False, 'blas_threads_per_process': 1,
        'notebooks': list(NOTEBOOKS), 'runs': [],
    }
    manifest_path = output / 'run.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    def run_one(name):
        started = time.monotonic()
        print(f'Running {name}', flush=True)
        with (output / 'logs' / f'{name}.log').open('w') as log:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), '--worker',
                 str(CODE_DIR / name), str(output / 'notebooks' / name)],
                cwd=CODE_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {'notebook': name, 'exit_code': result.returncode,
               'seconds': time.monotonic() - started, 'source_sha256': sha256(CODE_DIR / name)}
        print(f'{name}: {"passed" if result.returncode == 0 else "FAILED"} ({row["seconds"]:.1f}s)', flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, name) for name in NOTEBOOKS]
        for future in as_completed(futures):
            manifest['runs'].append(future.result())
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    failed = any(row['exit_code'] != 0 for row in manifest['runs'])
    manifest.update(status='failed' if failed else 'complete', finished_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{manifest["status"].capitalize()}: {output}', flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
