"""Dispatch raw-data operations to the declared benchmark protocol."""
from pathlib import Path
from cam.io import read
from cam.benchmarks.registry import data_protocol


def extract(raw_data, output, benchmark='robomme', count=25, tasks=None, depth=True,
            videos=True, workers=2, benchmark_source=None, resume=False):
    return data_protocol(benchmark).extract(raw_data, output, benchmark, count, tasks,
        depth, videos, workers, benchmark_source, resume)


def validate_dataset(root, verify_hashes=True):
    manifest = read(Path(root) / 'manifest.json')
    if manifest.get('schema') != 'cam-dataset/1':
        raise ValueError('Expected a fresh CaM raw-data extraction')
    return data_protocol(manifest['benchmark']).validate_dataset(root, verify_hashes)
