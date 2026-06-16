"""
Main training script for UNetFormer on OpenEarthMap.

Usage (Kaggle, 2× T4):
    torchrun --nproc_per_node=2 src/train.py --config configs/luot1_500.yaml
    torchrun --nproc_per_node=2 src/train.py --config configs/luot1_500.yaml --dry-run
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ── Path so `src.*` imports work regardless of cwd ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unetformer import build_model
from src.utils.callbacks import EarlyStopping
from src.utils.losses import CombinedLoss
from src.utils.metrics import SegmentationMetrics
from src.utils.visualizer import save_visualization


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def poly_lr(base_lr: float, cur_iter: int, max_iters: int,
            warmup_iters: int = 500, warmup_ratio: float = 1e-6,
            power: float = 0.9, min_lr: float = 0.0) -> float:
    if cur_iter < warmup_iters:
        k = (1 - warmup_ratio) / warmup_iters
        return base_lr * (warmup_ratio + k * cur_iter)
    progress = (cur_iter - warmup_iters) / max(max_iters - warmup_iters, 1)
    scale = (1 - progress) ** power
    return max(min_lr, base_lr * scale)


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def is_main() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def log(msg: str):
    if is_main():
        print(msg, flush=True)


# ── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, num_classes: int,
             device: torch.device, amp_enabled: bool) -> dict:
    model.eval()
    metrics  = SegmentationMetrics(num_classes=num_classes)
    total_loss, n_batches = 0.0, 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        n_batches  += 1
        metrics.update(logits, masks)

    # Aggregate confusion matrix across all DDP ranks → true global mIoU
    confusion_t = torch.from_numpy(metrics.confusion).to(device)
    dist.all_reduce(confusion_t, op=dist.ReduceOp.SUM)
    metrics.confusion = confusion_t.cpu().numpy()

    loss_t = torch.tensor([total_loss, float(n_batches)], device=device)
    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)

    result = metrics.compute()
    result['val_loss'] = (loss_t[0] / max(loss_t[1].item(), 1)).item()
    return result


# ── Visualise a few samples ──────────────────────────────────────────────────

@torch.no_grad()
def visualise_samples(model, loader, work_dir: str, iteration: int,
                      device: torch.device, amp_enabled: bool,
                      miou: float = 0.0, n: int = 4):
    model.eval()
    count = 0
    title = f'Iter {iteration:,}  |  mIoU = {miou:.4f}'
    for images, masks in loader:
        for i in range(images.size(0)):
            if count >= n:
                return
            img   = images[i].to(device).unsqueeze(0)
            with autocast(enabled=amp_enabled):
                logit = model(img)
            pred   = logit.argmax(dim=1).squeeze(0).cpu().numpy()
            gt     = masks[i].cpu().numpy()
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            save_path = os.path.join(work_dir, 'vis', f'iter{iteration:06d}_s{count}.png')
            save_visualization(img_np, gt, pred, save_path,
                               denormalize_img=True, title=title)
            count += 1


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  required=True, help='Path to YAML config')
    parser.add_argument('--dry-run', action='store_true',
                        help='Quick smoke-test: 5 iters, 2 val steps, 4 samples')
    args = parser.parse_args()

    # ── DDP init ────────────────────────────────────────────────────────────
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    cfg = load_config(args.config)
    ds  = cfg['DATASET']
    mdl = cfg['MODEL']
    tr  = cfg['TRAIN']
    opt = cfg['OPTIMIZER']
    out = cfg['OUTPUT']

    # Dry-run overrides
    if args.dry_run:
        tr['MAX_ITERS']   = 5
        tr['VAL_INTERVAL'] = 2

    os.makedirs(out['WORK_DIR'], exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    # Support both absolute paths and paths relative to ROOT_DIR
    tsf = ds['TRAIN_SPLIT_FILE']
    train_split = tsf if os.path.isabs(tsf) else os.path.join(ds['ROOT_DIR'], tsf)
    val_split   = ds['VAL_SPLIT_FILE']

    train_ds = OpenEarthMapDataset(
        root_dir=ds['ROOT_DIR'], img_dir=ds['TRAIN_IMG_DIR'],
        mask_dir=ds['TRAIN_MASK_DIR'], split_file=train_split,
        transform=get_train_transforms(),
    )
    val_root = ds.get('VAL_ROOT_DIR', ds['ROOT_DIR'])
    val_ds = OpenEarthMapDataset(
        root_dir=val_root, img_dir=ds['VAL_IMG_DIR'],
        mask_dir=ds['VAL_MASK_DIR'], split_file=val_split,
        transform=get_val_transforms(),
    )

    if args.dry_run:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
        val_ds   = Subset(val_ds,   list(range(min(4, len(val_ds)))))

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=tr['BATCH_SIZE_PER_GPU'],
        sampler=train_sampler, num_workers=tr['NUM_WORKERS'],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr['BATCH_SIZE_PER_GPU'],
        sampler=val_sampler, num_workers=tr['NUM_WORKERS'],
        pin_memory=True, drop_last=False,
    )

    # Infinite iterator over training data
    def cycle(loader):
        epoch = 0
        while True:
            loader.sampler.set_epoch(epoch)
            epoch += 1
            yield from loader

    train_iter = cycle(train_loader)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Loss, Optimizer, Scaler ──────────────────────────────────────────────
    criterion = CombinedLoss(num_classes=tr['NUM_CLASSES']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt['BASE_LR'],
        weight_decay=opt['WEIGHT_DECAY'],
    )
    scaler = GradScaler()

    # ── Training state ───────────────────────────────────────────────────────
    early_stopping    = EarlyStopping(patience=tr['EARLY_STOPPING_PATIENCE'])
    best_miou         = 0.0
    best_per_class    = None   # per-class IoU at best val
    best_state_dict   = None   # model weights at best val (kept in CPU memory)
    history           = []     # list of {iter, mIoU, val_loss, iou_<class>…}

    # ── Training loop ────────────────────────────────────────────────────────
    log(f"Training {tr['MAX_ITERS']} iterations | "
        f"encoder={mdl['ENCODER']} | split={ds['TRAIN_SPLIT_FILE']}")

    for iteration in range(1, tr['MAX_ITERS'] + 1):
        model.train()

        lr = poly_lr(
            base_lr=opt['BASE_LR'],
            cur_iter=iteration - 1,
            max_iters=tr['MAX_ITERS'],
            warmup_iters=tr.get('WARMUP_ITERS', 500),
        )
        set_lr(optimizer, lr)

        images, masks = next(train_iter)
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt['GRAD_CLIP'])
        scaler.step(optimizer)
        scaler.update()

        if iteration % 100 == 0:
            log(f"[{iteration:>6}/{tr['MAX_ITERS']}] loss={loss.item():.4f}  lr={lr:.2e}")

        # ── Validation ───────────────────────────────────────────────────────
        if iteration % tr['VAL_INTERVAL'] == 0 or iteration == tr['MAX_ITERS']:
            val_result = validate(
                model, val_loader, criterion, tr['NUM_CLASSES'],
                device, amp_enabled=True,
            )
            miou = val_result['mIoU']
            log(f"  [Val iter {iteration}] mIoU={miou:.4f}  val_loss={val_result['val_loss']:.4f}")

            if is_main():
                _cls  = ['Background','Bareland','Rangeland','Developed','Road',
                         'Tree','Water','Agriculture','Building']
                _abbr = ['Bg','Bare','Range','Dev','Road','Tree','Water','Agri','Bldg']
                per_class = val_result['per_class_iou']

                history.append({
                    'iter':     iteration,
                    'mIoU':     val_result['mIoU'],
                    'val_loss': val_result['val_loss'],
                    **{f'iou_{n}': round(v, 4) for n, v in zip(_cls, per_class)},
                })
                iou_str = '  '.join(f'{a}:{v:.3f}' for a, v in zip(_abbr, per_class))
                log(f"    {iou_str}")

                # Save visualisations
                visualise_samples(model, val_loader, out['WORK_DIR'],
                                  iteration, device, amp_enabled=True, miou=miou)

                # Track best in memory
                if miou > best_miou:
                    best_miou       = miou
                    best_per_class  = list(per_class)
                    best_state_dict = {k: v.cpu().clone()
                                       for k, v in model.module.state_dict().items()}
                    log(f"  ✔ New best mIoU={best_miou:.4f}")

                # Periodic checkpoint at multiples of 12000
                if iteration % 12000 == 0:
                    ckpt_path = os.path.join(out['WORK_DIR'],
                                             f'checkpoint_iter{iteration:06d}.pth')
                    torch.save(model.module.state_dict(), ckpt_path)
                    log(f"  Saved checkpoint iter {iteration} → {ckpt_path}")

            # Early stopping (check on rank 0, broadcast decision)
            should_stop = torch.tensor(
                int(early_stopping.step(miou)), device=device)
            dist.broadcast(should_stop, src=0)
            if should_stop.item():
                log(f"Early stopping at iteration {iteration}.")
                if is_main() and best_state_dict is not None:
                    ckpt_path = os.path.join(out['WORK_DIR'], 'best_model.pth')
                    torch.save(best_state_dict, ckpt_path)
                    log(f"  Saved best model (mIoU={best_miou:.4f}) → {ckpt_path}")
                break

    # ── Export artefacts ─────────────────────────────────────────────────────
    if is_main():
        # Save best_model.pth if training ended normally (no early stop saved it yet)
        ckpt_path = os.path.join(out['WORK_DIR'], 'best_model.pth')
        if not os.path.exists(ckpt_path) and best_state_dict is not None:
            torch.save(best_state_dict, ckpt_path)
            log(f"Saved best model (mIoU={best_miou:.4f}) → {ckpt_path}")

        # benchmark_results.csv
        csv_path = os.path.join(out['WORK_DIR'], 'benchmark_results.csv')
        if history:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
            log(f"Saved benchmark results to {csv_path}")

        try:
            import matplotlib.pyplot as plt

            # learning_curves.png
            iters  = [h['iter']     for h in history]
            mious  = [h['mIoU']     for h in history]
            losses = [h['val_loss'] for h in history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(iters, mious,  marker='o')
            ax1.set_title('Val mIoU'); ax1.set_xlabel('Iteration'); ax1.grid(True)
            ax2.plot(iters, losses, marker='o', color='orange')
            ax2.set_title('Val Loss'); ax2.set_xlabel('Iteration'); ax2.grid(True)
            fig.tight_layout()
            curve_path = os.path.join(out['WORK_DIR'], 'learning_curves.png')
            fig.savefig(curve_path, dpi=120)
            plt.close(fig)
            log(f"Saved learning curves to {curve_path}")

            # per_class_iou_best.png — bar chart at best checkpoint
            if best_per_class is not None:
                _cls = ['Background','Bareland','Rangeland','Developed','Road',
                        'Tree','Water','Agriculture','Building']
                colors = [
                    '#000000','#800000','#008000','#808000','#000080',
                    '#800080','#008080','#808080','#400000',
                ]
                fig2, ax = plt.subplots(figsize=(11, 5))
                bars = ax.bar(_cls, best_per_class, color=colors, edgecolor='white')
                for bar, val in zip(bars, best_per_class):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel('IoU')
                ax.set_title(f'Per-Class IoU at Best Checkpoint  (mIoU = {best_miou:.4f})',
                             fontweight='bold')
                ax.tick_params(axis='x', rotation=30)
                ax.grid(axis='y', alpha=0.3)
                fig2.tight_layout()
                bar_path = os.path.join(out['WORK_DIR'], 'per_class_iou_best.png')
                fig2.savefig(bar_path, dpi=120)
                plt.close(fig2)
                log(f"Saved per-class IoU chart to {bar_path}")

        except Exception as e:
            log(f"Warning: could not save charts — {e}")

    dist.destroy_process_group()
    log("Done.")


if __name__ == '__main__':
    main()
