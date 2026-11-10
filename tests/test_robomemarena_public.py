"""Archive moves and verified copies preserve canonical data and refuse collisions."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def archive():
    spec = importlib.util.spec_from_file_location(
        'archive', Path(__file__).resolve().parents[1] / 'scripts/organize_robomemarena_public.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_move_preserves_hardlinks_and_canonical_access(archive, tmp_path):
    source = tmp_path / 'old/run'; source.mkdir(parents=True)
    original = source / 'demo.h5'; original.write_bytes(b'original data')
    assigned = tmp_path / 'assigned.h5'; assigned.hardlink_to(original)
    inode = original.stat().st_ino
    target = tmp_path / 'public/task'; job = tmp_path / 'job'; job.mkdir()
    archive.move(source, target, job)
    assert source.is_symlink() and source.resolve() == target
    assert (source / 'demo.h5').read_bytes() == assigned.read_bytes() == b'original data'
    assert (target / 'demo.h5').stat().st_ino == assigned.stat().st_ino == inode
    archive.move(source, target, job)  # Idempotent resume.


def test_copy_checks_source_and_rejects_destination_collision(archive, tmp_path):
    source = tmp_path / 'source'; source.write_bytes(b'recorded score')
    target = tmp_path / 'target'; info = source.stat()
    expected = {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns}
    digest = archive.copy_file(source, target, expected)
    assert digest == archive.sha256(source) == archive.sha256(target)
    assert archive.copy_file(source, target, expected) == digest
    target.write_bytes(b'other scored data')
    with pytest.raises(ValueError, match='differs'):
        archive.copy_file(source, target, expected)
    assert target.read_bytes() == b'other scored data'
    source.write_bytes(b'changed source')
    with pytest.raises(AssertionError):
        archive.copy_file(source, tmp_path / 'new', expected)


def test_inventory_does_not_follow_external_artifacts(archive, tmp_path):
    external = tmp_path / 'assets'; external.mkdir(); (external / 'large.bin').write_bytes(b'asset')
    root = tmp_path / 'run'; root.mkdir(); (root / 'dependency').symlink_to(external)
    (root / 'result.json').write_text('{}')
    files, links = archive.inventory(root)
    assert [x['path'] for x in files] == ['result.json']
    assert links == [{'path': 'dependency', 'target': str(external)}]
