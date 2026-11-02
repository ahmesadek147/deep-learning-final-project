"""P4-only continuation into a NEW directory using the existing P4 runner.

The source run is read-only. Only the epoch cap changes. No teammate code is edited.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import torch

# Populated from the user's reviewed overlay when packaging.
REVIEWED_P4_HASHES = {'src/models/p4_variants.py': 'bf2e54311b0b7d2ccf7f693f3d15cc3074664b60856f1ca1f3e5f039a9231edc', 'src/training/p4_experiments.py': '490e1d61fe6d190e19f6ff1f3ffb6c2aed0fab6ab5c8c3511864c36ac389277b'}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding='utf-8')


def check_code(repo):
    for name, expected in REVIEWED_P4_HASHES.items():
        path = repo / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'{name} differs from the reviewed P4 overlay. Stop and review it first.')
    # Bind future resumes to the currently installed dependencies without modifying them.
    names = list(REVIEWED_P4_HASHES) + [
        'src/training/trainer.py', 'src/dataset.py', 'src/data/splits.py',
        'src/data/augmentation.py', 'src/models/simple_cnn.py', 'src/training/p4_continue.py']
    return {name: file_hash(repo / name) for name in names}


def prepare(source, destination, epochs, repo, code_hashes):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate, non-nested run directories.')
    if destination.exists():
        raise FileExistsError(f'{destination} already exists. Use --resume for the new run.')
    names = ['config.json', 'last.pth', 'best_model.pth']
    original_hashes = {name: file_hash(source / name) for name in names}
    config = json.loads((source / 'config.json').read_text())
    last = torch.load(source / 'last.pth', map_location='cpu', weights_only=True)
    best = torch.load(source / 'best_model.pth', map_location='cpu', weights_only=True)
    required = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                'scaler_state_dict', 'rng', 'history', 'best_epoch', 'best_key', 'wait', 'config_hash']
    missing = [key for key in required if key not in last]
    if missing:
        raise ValueError(f'Incomplete training state: {missing}')
    if last['config_hash'] != digest(config):
        raise ValueError('Source config.json does not match last.pth.')
    if config['model']['name'] != 'residual_cnn' or config['model'].get('pretrained') is not False:
        raise ValueError('This continuation is scoped to the reviewed scratch Residual CNN.')
    if config['data']['dataset_name'] != 'Multimodal-Fatima/CUB_train':
        raise ValueError('Only the training partition is permitted.')
    if config['data'].get('max_train_samples') or config['data'].get('max_val_samples'):
        raise ValueError('Cannot promote a sample-limited smoke run.')
    if config['data']['split_strategy'] != 'stratified_group':
        raise ValueError('Expected the existing grouped split.')
    if config['training']['monitor'] != 'val_macro_f1':
        raise ValueError('Expected validation macro-F1 selection.')
    if type(epochs) is not int or epochs <= last['epoch'] or epochs <= config['training']['epochs']:
        raise ValueError('Continuation must extend the original epoch cap.')
    rows = last['history']
    if [row['epoch'] for row in rows] != list(range(1, last['epoch'] + 1)):
        raise ValueError('Training history is incomplete.')
    selected = max(rows, key=lambda r: (r['val_macro_f1'], r['val_accuracy'], -r['val_loss']))
    key = (selected['val_macro_f1'], selected['val_accuracy'], -selected['val_loss'])
    if tuple(last['best_key']) != key or selected['epoch'] != last['best_epoch']:
        raise ValueError('Best-selection state does not match history.')
    if best['epoch'] != last['best_epoch'] or best['metrics'] != selected:
        raise ValueError('Source best checkpoint does not match the selected epoch.')
    if best['metadata']['model_config'] != config['model']:
        raise ValueError('Best checkpoint architecture differs from config.')
    if best['metadata']['image_size'] != config['data']['image_size']:
        raise ValueError('Image size mismatch.')
    if config['training'].get('early_stopping', False) and last['wait'] >= config['training'].get('patience', 20):
        raise ValueError('Source run already met early stopping; do not silently restart it.')
    # Checkpoint tensor layouts must agree even before model construction.
    if set(last['model_state_dict']) != set(best['model_state_dict']) or any(
        last['model_state_dict'][k].shape != best['model_state_dict'][k].shape
        for k in last['model_state_dict']):
        raise ValueError('Best/latest checkpoint tensor layouts differ.')
    manifest = Path(config['data']['split_manifest'])
    if not manifest.is_absolute():
        manifest = repo / manifest
    manifest_hash = file_hash(manifest)
    new_config = deepcopy(config)
    new_config['training']['epochs'] = epochs
    last['config_hash'] = digest(new_config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + '.preparing-', dir=destination.parent))
    try:
        write_json(staging / 'config.json', new_config)
        torch.save(last, staging / 'last.pth')
        shutil.copy2(source / 'best_model.pth', staging / 'best_model.pth')
        write_json(staging / 'history.json', rows)
        provenance = {'source_run': str(source), 'source_hashes': original_hashes,
            'parent_epoch': rows[-1]['epoch'], 'original_epoch_cap': config['training']['epochs'],
            'new_epoch_cap': epochs, 'config_hash': digest(new_config), 'code_hashes': code_hashes,
            'manifest_path': str(manifest.resolve()), 'manifest_sha256': manifest_hash,
            'torch_version_at_fork': str(torch.__version__),
            'changed_training_settings': ['epochs'],
            'historical_code_and_manifest_hashes_available': False,
            'note': 'Source code/manifest hashes were not saved by the original runner; these bind continuation from now onward.'}
        write_json(staging / 'continuation.json', provenance)
        for name, expected in original_hashes.items():
            if file_hash(source / name) != expected:
                raise RuntimeError('Source changed while preparing continuation.')
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return new_config, provenance


def verify_resume(destination, source, epochs, code_hashes):
    provenance = json.loads((destination / 'continuation.json').read_text())
    config = json.loads((destination / 'config.json').read_text())
    if provenance['source_run'] != str(source.resolve()) or provenance['new_epoch_cap'] != epochs:
        raise ValueError('Continuation source or epoch cap differs.')
    if provenance['code_hashes'] != code_hashes or provenance['config_hash'] != digest(config):
        raise ValueError('Code or continuation configuration changed.')
    if file_hash(provenance['manifest_path']) != provenance['manifest_sha256']:
        raise ValueError('Split manifest changed.')
    if str(torch.__version__) != provenance['torch_version_at_fork']:
        raise ValueError('PyTorch version changed; review before resuming.')
    for name, expected in provenance['source_hashes'].items():
        if file_hash(source / name) != expected:
            raise ValueError(f'Original source artifact changed: {name}')
    return config, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run-id', default='p4_residual_continued_120')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {'', '.', '..'}:
        parser.error('Use a simple run ID.')
    repo = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve() != repo:
        parser.error(f'Run from the repository root: {repo}')
    hashes = check_code(repo)
    source = args.source_run.resolve()
    destination = (args.output_root / 'runs' / args.run_id).resolve()
    if args.resume:
        config, provenance = verify_resume(destination, source, args.epochs, hashes)
    else:
        config, provenance = prepare(source, destination, args.epochs, repo, hashes)
    print(f'Original preserved: {source}')
    print(f'Continuation directory: {destination}')
    print(f"Parent epoch: {provenance['parent_epoch']}; maximum total epochs: {args.epochs}")
    if args.prepare_only:
        print('Prepared successfully. Run again with --resume and without --prepare-only to train.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Enable the Colab GPU, then rerun with --resume.')
    state = torch.load(destination / 'last.pth', map_location='cpu', weights_only=True)
    stopped = config['training'].get('early_stopping', False) and state['wait'] >= config['training'].get('patience', 20)
    if state['epoch'] >= args.epochs or stopped:
        print('Continuation already reached its epoch cap or early-stopping condition.')
        return
    del state
    from src.training.p4_experiments import run_one
    try:
        result = run_one(config, args.output_root, args.run_id, resume=True)
        print(json.dumps({k: result[k] for k in ['best_epoch', 'best_val_accuracy', 'best_val_macro_f1', 'checkpoint']}, indent=2))
    finally:
        for name, expected in provenance['source_hashes'].items():
            if file_hash(source / name) != expected:
                raise RuntimeError(f'Original source artifact changed: {name}')


if __name__ == '__main__':
    main()
