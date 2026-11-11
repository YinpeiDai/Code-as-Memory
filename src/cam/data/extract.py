"""Dispatch raw-data operations to the declared benchmark protocol."""
from pathlib import Path
from cam.io import read
from cam.interface.registry import data_protocol


def raw_data_reference(manifest):
    """Use recorded acquisition provenance, or the immutable public demo release.

    Published pools omit private acquisition paths. Their pinned release is a
    reference to the distributed demonstrations, not to the historical raw data.
    """
    original = manifest.get('raw_data') or manifest.get('source')
    if original:
        return original
    release = manifest.get('download', {})
    if release.get('repo_id') and release.get('revision'):
        return f"hf://datasets/{release['repo_id']}@{release['revision']}"
    raise ValueError('Dataset must record its acquisition source or pinned public release')


def extract(raw_data, output, benchmark='robomme', count=25, tasks=None, depth=True,
            videos=True, workers=2, benchmark_source=None, resume=False, reuse=None,
            max_successful=None, replay_cache=None, single_pass=False, validation_episodes=0):
    options = {key: value for key, value in {'reuse': reuse, 'max_successful': max_successful,
                                            'replay_cache': replay_cache}.items() if value is not None}
    if single_pass or validation_episodes:
        options.update(single_pass=single_pass, validation_episodes=validation_episodes)
    if options and benchmark != 'robomemarena':
        raise ValueError('Replay reuse/caps are supported only for RoboMemArena')
    return data_protocol(benchmark).extract(raw_data, output, benchmark, count, tasks,
        depth, videos, workers, benchmark_source, resume, **options)


def validate_dataset(root, verify_hashes=True):
    manifest = read(Path(root) / 'manifest.json')
    if manifest.get('schema') not in ('cam-dataset/1', 'cap-regular-extraction/2'):
        raise ValueError('Expected a fresh CaM raw-data extraction')
    return data_protocol(manifest['benchmark']).validate_dataset(root, verify_hashes)


def load_dataset_manifest(root, adapter=None):
    """Read a previously validated dataset without opening or hashing its H5s."""
    root = Path(root).resolve()
    manifest = read(root / 'manifest.json')
    if manifest.get('schema') == 'cap-regular-extraction/2' and manifest.get('benchmark') == 'robomme':
        from cam.interface.robomme.data import published_manifest
        manifest = published_manifest(root, manifest, adapter)
    if manifest.get('schema') != 'cam-dataset/1' or not manifest.get('tasks'):
        raise ValueError('Expected a prepared CaM dataset manifest')
    for task, item in manifest['tasks'].items():
        path = root / item['file']
        if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
            raise ValueError(f'{task}: demonstration file must exist inside the dataset')
        digest = item.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'{task}: missing published demonstration checksum')
        records = item['episodes']
        identities = [int(row['episode']) for row in records]
        if not identities or min(identities) < 0 or len(set(identities)) != len(identities):
            raise ValueError(f'{task}: expected distinct assigned demonstration identities')
    return manifest
