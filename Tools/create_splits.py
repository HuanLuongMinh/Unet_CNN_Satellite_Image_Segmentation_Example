"""
Generate all split files for OpenEarthMap.

Workflow (called automatically in sequence):
  1. prepare_splits.py  — validate + copy val images, augment if < val_size,
                          write val_2000_fixed.txt
  2. (this script)      — scan images/train, write train_500/1000/1500_fixed.txt

Usage (Kaggle):
    python tools/create_splits.py \
        --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
        --output-dir /kaggle/working/unetformer-openearthmap

Usage (local):
    python tools/create_splits.py --data-root /path/to/dataset

NOTE: On Kaggle, /kaggle/input/ is read-only.
      Always set --output-dir to /kaggle/working/... when running on Kaggle.
"""

import argparse
import os
import random
import subprocess
import sys


# ── Helpers ───────────────────────────────────────────────────────────────────

def scan_images(directory, suffix='.tif'):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.endswith(suffix)
    )


def write_split(path, names, skip_if_exists=False):
    if skip_if_exists and os.path.exists(path):
        print(f'  Skipped (already exists): {path}')
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(names) + '\n')
    print(f'  Created: {path}  ({len(names)} samples)')


def find_data_base(data_root):
    """Auto-detect actual dataset base by probing common subdirectory layouts."""
    candidates = [
        data_root,
        os.path.join(data_root, 'OpenEarthMap_Mini'),
        os.path.join(data_root, 'OpenEarthMap_flat'),
        os.path.join(data_root, 'OpenEarthMap'),
        os.path.join(data_root, 'openearthmap'),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'images', 'val')) or \
           os.path.isdir(os.path.join(c, 'images', 'train')):
            if c != data_root:
                print(f'  Auto-detected dataset base: {c}')
            return c

    # Diagnostic output if nothing found
    print(f'\nERROR: Could not find images/train or images/val under any of:')
    for c in candidates:
        print(f'  {c}')
    if os.path.isdir(data_root):
        contents = sorted(os.listdir(data_root))
        print(f'\nContents of {data_root}:')
        for item in contents[:30]:
            full = os.path.join(data_root, item)
            kind = '[dir]' if os.path.isdir(full) else '[file]'
            print(f'  {kind}  {item}')
    return data_root  # fall through so downstream raises the real error


def find_label_subdir(base):
    """Detect whether labels live in 'labels/' or 'label/'."""
    for name in ('labels', 'label'):
        if os.path.isdir(os.path.join(base, name)):
            return name
    return 'labels'  # default


# ── Step 1: call prepare_splits.py ───────────────────────────────────────────

def run_prepare_splits(data_root, output_dir, seed, img_suffix):
    """Delegate val split generation to prepare_splits.py."""
    base = find_data_base(data_root)
    label_sub = find_label_subdir(base)

    script = os.path.join(os.path.dirname(__file__), 'prepare_splits.py')
    cmd = [
        sys.executable, script,
        '--val-dir',       os.path.join(base, 'images', 'val'),
        '--label-dir',     os.path.join(base, label_sub, 'val'),
        '--out-dir',       output_dir,
        '--aug-img-dir',   os.path.join(output_dir, 'images', 'val'),
        '--aug-label-dir', os.path.join(output_dir, label_sub, 'val'),
        '--seed',          str(seed),
        '--img-suffix',    img_suffix,
        '--val-size',      '2000',
    ]
    val_path = os.path.join(output_dir, 'val_2000_fixed.txt')
    if os.path.exists(val_path):
        print(f'  val_2000_fixed.txt already exists — skipping prepare_splits.')
        return

    print('\n=== Step 1: prepare_splits.py (val split) ===')
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError('prepare_splits.py failed — see output above.')


# ── Step 2: train splits ──────────────────────────────────────────────────────

def run_train_splits(data_root, output_dir, seed, img_suffix):
    base = find_data_base(data_root)
    train_dir = os.path.join(base, 'images', 'train')
    print(f'\n=== Step 2: train splits ===')
    print(f'Scanning: {train_dir}')
    train_names = scan_images(train_dir, img_suffix)
    print(f'  Found {len(train_names)} training images')

    random.seed(seed)
    random.shuffle(train_names)

    for n in [500, 1000, 1500]:
        if len(train_names) < n:
            print(f'  WARNING: only {len(train_names)} images — '
                  f'train_{n}_fixed.txt will use all of them.')
        subset = train_names[:min(n, len(train_names))]
        write_split(os.path.join(output_dir, f'train_{n}_fixed.txt'), subset)

    # Full training split
    write_split(
        os.path.join(output_dir, f'train_{len(train_names)}_fixed.txt'),
        train_names,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Create all OpenEarthMap split files (val via prepare_splits, then train)')
    parser.add_argument('--data-root', required=True,
                        help='Dataset root (e.g. /kaggle/input/datasets/dyiyacao/openearthmap)')
    parser.add_argument('--output-dir', default=None,
                        help='Where to write .txt files. '
                             'On Kaggle: /kaggle/working/unetformer-openearthmap. '
                             'Default: same as --data-root.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--img-suffix', default='.tif')
    args = parser.parse_args()

    base_dir   = args.output_dir if args.output_dir else args.data_root
    output_dir = os.path.join(base_dir, 'dataset')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Data root  : {args.data_root}')
    print(f'Output dir : {output_dir}  (dataset/)')

    run_prepare_splits(args.data_root, output_dir, args.seed, args.img_suffix)
    run_train_splits(args.data_root, output_dir, args.seed, args.img_suffix)

    print('\n=== All splits ready ===')
    print(f'  Location: {output_dir}')
    print('\nNext step — start training:')
    print('  torchrun --nproc_per_node=2 src/train.py --config configs/luot1_500.yaml')


if __name__ == '__main__':
    main()
