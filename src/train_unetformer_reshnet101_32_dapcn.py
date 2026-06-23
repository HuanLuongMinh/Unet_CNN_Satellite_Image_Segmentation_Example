"""
Training script for UNetFormer + DAPCN (decoder GLA transformer + Dynamic
Anchor Prototype Grouping + Boundary loss) — encoder
resnext101_32x16d.fb_swsl_ig1b_ft_in1k, BASE_LR = 1e-5.
Loss = CrossEntropy + Loss Boundary + Loss DAPCN (DAPG), KHÔNG contrastive.

File này độc lập với src/train_unet_former_reshnet101_32.py (CE thuần) và
src/train_unet_former_reshnet101_32_combineLoss.py (CE+Dice) — chỉ khác model
import (model riêng unetformer_reshnet101_32_dapcn.py, forward trả thêm aux
loss dict khi train) và cách cộng loss trong training loop. Toàn bộ pipeline
DDP/AMP/checkpoint/resume/early-stopping/visualization giữ nguyên để các thí
nghiệm cũ không bị ảnh hưởng khi chạy lại.

Usage (Kaggle, 2x T4):
    torchrun --nproc_per_node=2 src/train_unetformer_reshnet101_32_dapcn.py --config configs/unetformer_reshnet101_32_dapcn/luot1_500.yaml
    torchrun --nproc_per_node=2 src/train_unetformer_reshnet101_32_dapcn.py --config configs/unetformer_reshnet101_32_dapcn/luot1_500.yaml --dry-run
    torchrun --nproc_per_node=2 src/train_unetformer_reshnet101_32_dapcn.py --config configs/unetformer_reshnet101_32_dapcn/luot1_500.yaml --resume <work_dir>/latest_checkpoint.pth
"""

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ── Path so `src.*` imports work regardless of cwd ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unetformer_reshnet101_32_dapcn import build_model
from src.utils.callbacks import EarlyStopping
from src.utils.metrics import SegmentationMetrics
from src.utils.visualizer import save_visualization


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Đặt seed chung cho random/numpy/torch để 3 lượt 500/1000/1500 ảnh
    đều khởi tạo model (decoder/proj random init) và shuffle dữ liệu theo
    đúng cùng 1 seed, chỉ khác nhau ở lượng dữ liệu train."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_checkpoint_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f'--resume checkpoint not found: {path}')
    return torch.load(path, map_location='cpu', weights_only=False)


def save_checkpoint(path: str, model, optimizer, scaler, iteration: int,
                     early_stopping, best_miou: float, best_per_class):
    ckpt = {
        'iteration':            iteration,
        'model_state_dict':     model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'early_stopping': {
            'prev':      early_stopping.prev,
            'counter':   early_stopping.counter,
            'triggered': early_stopping.triggered,
        },
        'best_miou':      best_miou,
        'best_per_class': best_per_class,
    }
    tmp_path = path + '.tmp'
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


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
        with autocast('cuda', enabled=amp_enabled):
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
            with autocast('cuda', enabled=amp_enabled):
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
    parser.add_argument('--resume', default=None,
                        help='Path to latest_checkpoint.pth to resume training from')
    args = parser.parse_args()

    # ── DDP init ────────────────────────────────────────────────────────────
    dist.init_process_group(backend='nccl')
    train_start    = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    cfg = load_config(args.config)
    ds  = cfg['DATASET']
    mdl = cfg['MODEL']
    tr  = cfg['TRAIN']
    opt = cfg['OPTIMIZER']
    out = cfg['OUTPUT']

    resume_ckpt = load_checkpoint_file(args.resume) if args.resume else None

    seed = tr.get('SEED', 42)
    set_seed(seed)
    log(f"Using seed={seed}")

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

    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=seed)
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
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['model_state_dict'])
        log(f"Resumed model weights from {args.resume} (iteration {resume_ckpt['iteration']})")
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Loss, Optimizer, Scaler ──────────────────────────────────────────────
    # CE + Loss Boundary + Loss DAPCN (DAPG) — 2 aux loss được tính ngay trong
    # model.forward(images, masks) và cộng vào CE bên dưới (xem training loop).
    criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt['BASE_LR'],
        weight_decay=opt['WEIGHT_DECAY'],
    )
    scaler = GradScaler('cuda')
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        scaler.load_state_dict(resume_ckpt['scaler_state_dict'])

    # ── Training state ───────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=tr['EARLY_STOPPING_PATIENCE'])
    if resume_ckpt is not None:
        early_stopping.prev      = resume_ckpt['early_stopping']['prev']
        early_stopping.counter   = resume_ckpt['early_stopping']['counter']
        early_stopping.triggered = resume_ckpt['early_stopping']['triggered']

    best_miou      = resume_ckpt['best_miou']      if resume_ckpt is not None else 0.0
    best_per_class = resume_ckpt['best_per_class'] if resume_ckpt is not None else None
    best_state_dict = None   # model weights at best val (kept in CPU memory)
    history         = []     # list of {iter, mIoU, val_loss, iou_<class>…}

    if resume_ckpt is not None and is_main():
        best_path = os.path.join(out['WORK_DIR'], 'best_model.pth')
        if os.path.exists(best_path):
            best_state_dict = torch.load(best_path, map_location='cpu', weights_only=False)
            log(f"Reloaded best_model.pth (mIoU={best_miou:.4f}) for export continuity")
        elif best_miou > 0.0:
            log(f"WARNING: resumed with best_miou={best_miou:.4f} but {best_path} "
                f"was not found — best checkpoint weights are unavailable until a new best is found.")

        csv_path = os.path.join(out['WORK_DIR'], 'benchmark_results.csv')
        if os.path.exists(csv_path):
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    row['iter']     = int(row['iter'])
                    row['mIoU']     = float(row['mIoU'])
                    row['val_loss'] = float(row['val_loss'])
                    history.append(row)
            log(f"Reconstructed {len(history)} history rows from {csv_path}")

    start_iter = resume_ckpt['iteration'] + 1 if resume_ckpt is not None else 1
    if args.dry_run and resume_ckpt is not None and start_iter > tr['MAX_ITERS']:
        log(f"Warning: --dry-run MAX_ITERS={tr['MAX_ITERS']} <= resumed "
            f"start_iter={start_iter}; loop will not execute.")

    # ── Training loop ────────────────────────────────────────────────────────
    log(f"Training {tr['MAX_ITERS']} iterations | "
        f"encoder={mdl['ENCODER']} | loss=CE+Boundary+DAPG | split={ds['TRAIN_SPLIT_FILE']}")

    for iteration in range(start_iter, tr['MAX_ITERS'] + 1):
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
        with autocast('cuda'):
            logits, aux = model(images, masks)   # DAPCN: forward trả thêm aux losses
            loss = criterion(logits, masks)      # CE
            for _v in aux.values():               # + boundary + DAPG (đã weighted sẵn)
                loss = loss + _v

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

            stop_flag = False
            if is_main():
                _cls  = ['Background','Bareland','Rangeland','Developed','Road',
                         'Tree','Water','Agriculture','Building']
                _abbr = ['Bg','Bare','Range','Dev','Road','Tree','Water','Agri','Bldg']
                per_class = val_result['per_class_iou']

                # Track best in memory + on disk
                is_new_best = miou > best_miou
                if is_new_best:
                    best_miou       = miou
                    best_per_class  = list(per_class)
                    best_state_dict = {k: v.cpu().clone()
                                       for k, v in model.module.state_dict().items()}
                    best_path = os.path.join(out['WORK_DIR'], 'best_model.pth')
                    torch.save(best_state_dict, best_path)
                    log(f"  ✔ New best mIoU={best_miou:.4f} → saved {best_path}")

                row = {
                    'iter':     iteration,
                    'mIoU':     val_result['mIoU'],
                    'val_loss': val_result['val_loss'],
                    **{f'iou_{n}': round(v, 4) for n, v in zip(_cls, per_class)},
                    'is_best':  is_new_best,
                }
                history.append(row)
                iou_str = '  '.join(f'{a}:{v:.3f}' for a, v in zip(_abbr, per_class))
                log(f"    {iou_str}")

                # Append to benchmark_results.csv incrementally (survives a crash)
                csv_path = os.path.join(out['WORK_DIR'], 'benchmark_results.csv')
                write_header = not os.path.exists(csv_path)
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

                # Save visualisations
                visualise_samples(model, val_loader, out['WORK_DIR'],
                                  iteration, device, amp_enabled=True, miou=miou)

                # Early stopping decision (computed on rank 0, broadcast below)
                stop_flag = early_stopping.step(miou)

                # Full resumable checkpoint, overwritten after every validation
                ckpt_path = os.path.join(out['WORK_DIR'], 'latest_checkpoint.pth')
                save_checkpoint(ckpt_path, model, optimizer, scaler, iteration,
                                 early_stopping, best_miou, best_per_class)
                log(f"  Saved checkpoint (iter {iteration}) → {ckpt_path}")

            should_stop = torch.tensor(int(stop_flag), device=device)
            dist.broadcast(should_stop, src=0)
            if should_stop.item():
                log(f"Early stopping at iteration {iteration}.")
                break

    # ── Export artefacts ─────────────────────────────────────────────────────
    if is_main():
        # Save best_model.pth if training ended normally (no early stop saved it yet)
        ckpt_path = os.path.join(out['WORK_DIR'], 'best_model.pth')
        if not os.path.exists(ckpt_path):
            if best_state_dict is not None:
                torch.save(best_state_dict, ckpt_path)
                log(f"Saved best model (mIoU={best_miou:.4f}) → {ckpt_path}")
            elif best_miou > 0.0:
                log(f"WARNING: best_miou={best_miou:.4f} but no best checkpoint weights "
                    f"were available to write to {ckpt_path}.")

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

    if is_main():
        elapsed      = time.time() - train_start
        end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hours        = int(elapsed // 3600)
        minutes      = int((elapsed % 3600) // 60)
        seconds      = int(elapsed % 60)

        num       = os.path.basename(out['WORK_DIR']).split('_')[-1]
        time_path = os.path.join(out['WORK_DIR'], f'time_{num}.txt')
        with open(time_path, 'w', encoding='utf-8') as f:
            f.write(f'Bắt đầu      : {start_datetime}\n')
            f.write(f'Kết thúc     : {end_datetime}\n')
            f.write(f'Tổng thời gian: {hours:02d}h {minutes:02d}m {seconds:02d}s\n')
        print(f'Saved timing → {time_path}', flush=True)

    dist.destroy_process_group()
    log("Done.")


if __name__ == '__main__':
    main()
