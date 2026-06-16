"""
v1.py — Prepare OpenEarthMap val split (2000 images) for MMSegmentation on Kaggle.

Workflow:
  1. Scan images/val, filter to only those with a matching label (Fix 5)
  2. Copy ALL valid originals to aug_img_dir / aug_label_dir (Fix 1)
  3. If originals >= val_size: select top-N, done
  4. Else: run multi-round geometric augmentation until val_size reached (Fix 3)
  5. Write split file with relative-path entries "images/val/<name>" (Fix 2)
  6. Separate --label-suffix for cases where label extension differs (Fix 4)

Usage (Kaggle, no args needed):
    python v1.py

Override any path:
    python v1.py --val-dir /path/to/images/val --label-dir /path/to/label/val \\
                 --out-dir /kaggle/working --val-size 2000

After running, update openearthmap_val2000.py:
    data_root = '/kaggle/working'   # parent of images/ and label/
"""

import argparse
import os
import shutil
import random
import sys
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Create val split .txt for OpenEarthMap (with augmentation fallback)')
    p.add_argument(
        '--val-dir',
        default='/kaggle/input/datasets/dyiyacao/openearthmap/images/val',
        help='Source directory containing validation images')
    p.add_argument(
        '--label-dir',
        default='/kaggle/input/datasets/dyiyacao/openearthmap/labels/val',
        help='Source directory containing validation labels/masks')
    p.add_argument(
        '--out-dir',
        default='/kaggle/working/unetformer-openearthmap/dataset',
        help='Directory to write val_N_fixed.txt')
    p.add_argument(
        '--aug-img-dir',
        default='/kaggle/working/unetformer-openearthmap/dataset/images/val',
        help='Destination for copies of originals + augmented images')
    p.add_argument(
        '--aug-label-dir',
        default='/kaggle/working/unetformer-openearthmap/dataset/labels/val',
        help='Destination for copies of originals + augmented labels')
    p.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)')
    p.add_argument(
        '--val-size', type=int, default=2000,
        help='Number of validation samples to produce (default: 2000)')
    p.add_argument(
        '--img-suffix', default='.tif',
        help='Image file extension (default: .tif)')
    # Fix 4: separate label suffix
    p.add_argument(
        '--label-suffix', default=None,
        help='Label file extension (default: same as --img-suffix)')
    p.add_argument(
        '--max-aug-rounds', type=int, default=3,
        help='Maximum augmentation rounds; each round multiplies pool by 6 (default: 3)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def scan_images(directory, suffix):
    """Return sorted list of basenames (without extension) for images in directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.endswith(suffix)
    )


def load_tif(path):
    try:
        import tifffile
        return tifffile.imread(path)
    except ImportError:
        from PIL import Image
        return np.array(Image.open(path))


def save_tif(array, path):
    try:
        import tifffile
        tifffile.imwrite(path, np.ascontiguousarray(array))
    except ImportError:
        from PIL import Image
        Image.fromarray(np.ascontiguousarray(array)).save(path)


# ---------------------------------------------------------------------------
# Data copying
# ---------------------------------------------------------------------------

def copy_originals(names, src_img_dir, src_label_dir,
                   dst_img_dir, dst_label_dir, img_suffix, label_suffix):
    """Copy original image-label pairs from source to working directories.

    Called for BOTH Branch A and Branch B so every file referenced in the
    split file always exists under aug_img_dir / aug_label_dir (Fix 1).
    """
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)
    for name in names:
        shutil.copy2(
            os.path.join(src_img_dir,   name + img_suffix),
            os.path.join(dst_img_dir,   name + img_suffix))
        shutil.copy2(
            os.path.join(src_label_dir, name + label_suffix),
            os.path.join(dst_label_dir, name + label_suffix))
    print(f'  Copied {len(names)} original pairs → {dst_img_dir}')


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

# Geometric-only transforms — safe to apply identically to image AND label
_AUG_TRANSFORMS = [
    ('hflip',       lambda a: np.fliplr(a)),
    ('vflip',       lambda a: np.flipud(a)),
    ('rot90',       lambda a: np.rot90(a, k=1)),
    ('rot180',      lambda a: np.rot90(a, k=2)),
    ('rot270',      lambda a: np.rot90(a, k=3)),
    ('hflip_rot90', lambda a: np.rot90(np.fliplr(a), k=1)),
]


def _augment_one(base, img_suffix, label_suffix,
                 src_img_dir, src_label_dir,
                 dst_img_dir, dst_label_dir,
                 aug_fn, new_name):
    """Apply one geometric transform to an image-label pair and save the result."""
    save_tif(
        aug_fn(load_tif(os.path.join(src_img_dir,   base + img_suffix))),
        os.path.join(dst_img_dir, new_name + img_suffix))
    save_tif(
        aug_fn(load_tif(os.path.join(src_label_dir, base + label_suffix))),
        os.path.join(dst_label_dir, new_name + label_suffix))


def generate_augmented_multiround(orig_names, img_suffix, label_suffix,
                                   src_img_dir, src_label_dir,
                                   dst_img_dir, dst_label_dir,
                                   needed, max_rounds):
    """Generate augmented pairs using multi-round augmentation (Fix 3).

    Round 1: apply 6 transforms to each original → up to 6N new samples.
    Round 2: apply 6 transforms to round-1 outputs → up to 36N samples.
    … continues until `needed` samples produced or `max_rounds` exhausted.

    All new files are written directly to dst_img_dir / dst_label_dir
    (originals were already copied there by copy_originals).

    Returns list of new basenames (without extension).
    """
    aug_names = []
    counter = 0
    # Pool starts as originals; each round expands it
    current_pool = list(orig_names)
    current_pool_dirs = (src_img_dir, src_label_dir)

    for rnd in range(1, max_rounds + 1):
        if len(aug_names) >= needed:
            break

        next_pool = []
        print(f'  Augmentation round {rnd}: {len(current_pool)} source images')

        for _, aug_fn in _AUG_TRANSFORMS:
            if len(aug_names) >= needed:
                break
            for base in current_pool:
                if len(aug_names) >= needed:
                    break

                new_name = f'{base}_aug{counter}'
                _augment_one(
                    base, img_suffix, label_suffix,
                    current_pool_dirs[0], current_pool_dirs[1],
                    dst_img_dir, dst_label_dir,
                    aug_fn, new_name)

                aug_names.append(new_name)
                next_pool.append(new_name)
                counter += 1

        # Next round reads from the working output directory
        current_pool = next_pool
        current_pool_dirs = (dst_img_dir, dst_label_dir)

    return aug_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # When running inside Kaggle/Jupyter notebook, override sys.argv
    if 'ipykernel' in sys.modules:
        sys.argv = [
            'prepare_splits.py',
            '--val-dir',       '/kaggle/input/datasets/dyiyacao/openearthmap/images/val',
            '--label-dir',     '/kaggle/input/datasets/dyiyacao/openearthmap/labels/val',
            '--out-dir',       '/kaggle/working/unetformer-openearthmap/dataset',
            '--aug-img-dir',   '/kaggle/working/unetformer-openearthmap/dataset/images/val',
            '--aug-label-dir', '/kaggle/working/unetformer-openearthmap/dataset/labels/val',
            '--val-size',      '2000',
        ]
    args = parse_args()

    # Fix 4: resolve effective label suffix
    label_suffix = args.label_suffix if args.label_suffix else args.img_suffix

    print('=== OpenEarthMap Val Split Generator (v1) ===')
    print(f'  val-dir      : {args.val_dir}')
    print(f'  label-dir    : {args.label_dir}')
    print(f'  aug-img-dir  : {args.aug_img_dir}')
    print(f'  aug-label-dir: {args.aug_label_dir}')
    print(f'  out-dir      : {args.out_dir}')
    print(f'  val-size     : {args.val_size}')
    print(f'  seed         : {args.seed}')
    print(f'  img-suffix   : {args.img_suffix}')
    print(f'  label-suffix : {label_suffix}')
    print(f'  max-aug-rounds: {args.max_aug_rounds}')

    # 1. Scan images
    print(f'\nScanning: {args.val_dir}')
    all_img_names = scan_images(args.val_dir, args.img_suffix)
    print(f'  Found {len(all_img_names)} images')

    # Fix 5: filter to only images that have a matching label
    if not os.path.isdir(args.label_dir):
        raise FileNotFoundError(f'Label directory not found: {args.label_dir}')

    valid_names = [
        n for n in all_img_names
        if os.path.isfile(os.path.join(args.label_dir, n + label_suffix))
    ]
    n_skipped = len(all_img_names) - len(valid_names)
    if n_skipped:
        print(f'  WARNING: {n_skipped} image(s) skipped — no matching label found')
    print(f'  {len(valid_names)} valid image-label pairs available')

    if not valid_names:
        raise RuntimeError('No valid image-label pairs found. Check --val-dir and --label-dir.')

    # 2. Shuffle
    random.seed(args.seed)
    names = list(valid_names)
    random.shuffle(names)

    # Fix 1: always copy originals to the working directory
    print(f'\nCopying originals to working directory ...')
    copy_originals(names, args.val_dir, args.label_dir,
                   args.aug_img_dir, args.aug_label_dir,
                   args.img_suffix, label_suffix)

    n_orig = len(names)
    n_aug = 0

    if n_orig >= args.val_size:
        # Branch A: enough originals
        selected = names[:args.val_size]
        print(f'  Sufficient images — selecting {args.val_size} (no augmentation needed)')

    else:
        # Branch B: need augmentation
        needed = args.val_size - n_orig
        max_possible = n_orig * (len(_AUG_TRANSFORMS) ** args.max_aug_rounds)
        print(f'\n  Need {needed} augmented samples '
              f'(max possible with {args.max_aug_rounds} rounds: {max_possible})')

        aug_names = generate_augmented_multiround(
            orig_names=names,
            img_suffix=args.img_suffix,
            label_suffix=label_suffix,
            src_img_dir=args.val_dir,
            src_label_dir=args.label_dir,
            dst_img_dir=args.aug_img_dir,
            dst_label_dir=args.aug_label_dir,
            needed=needed,
            max_rounds=args.max_aug_rounds,
        )
        n_aug = len(aug_names)

        if n_aug < needed:
            raise RuntimeError(
                f'Could only generate {n_aug} augmented samples (needed {needed}). '
                f'Increase --max-aug-rounds (current: {args.max_aug_rounds}) or '
                f'provide more source images.')

        selected = names + aug_names

    # Fix 2: write relative paths "images/val/<name>" not just "<name>"
    os.makedirs(args.out_dir, exist_ok=True)
    out_filename = f'val_{args.val_size}_fixed.txt'
    out_path = os.path.join(args.out_dir, out_filename)
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(f'images/val/{n}' for n in selected) + '\n')

    # Summary
    data_root_for_config = os.path.dirname(os.path.dirname(
        os.path.abspath(args.aug_img_dir)))
    print(f'\n=== Done ===')
    print(f'  Original pairs   : {n_orig}')
    if n_aug:
        print(f'  Augmented pairs  : {n_aug}')
    print(f'  Total in split   : {len(selected)}')
    print(f'  Output file      : {out_path}')
    print(f'\n  [CONFIG] Update openearthmap_val2000.py:')
    print(f'    data_root = \'{data_root_for_config}\'')


if __name__ == '__main__':
    main()
