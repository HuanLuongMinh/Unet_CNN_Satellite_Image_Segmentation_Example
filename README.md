# UNet CNN — Satellite Image Semantic Segmentation

An ablation study comparing **CNN encoders (ResNet-18, ResNet-101)** vs. **Vision Transformer encoder (MiT-B0)** on the [OpenEarthMap](https://open-earth-map.org/) dataset using a dynamic multi-encoder UNetFormer architecture.

---

## Architecture

```
Input (512×512)
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Encoder (interchangeable)                       │
│  ├─ ResNet-18   → channels [64, 128, 256, 512]   │
│  ├─ ResNet-101  → channels [256, 512, 1024, 2048]│
│  └─ MiT-B0     → channels [32, 64, 160, 256]     │
└──────────────────────────────────────────────────┘
     │  4 multi-scale feature maps (stride 4→32)
     ▼
┌─────────────────────────────────────┐
│  Channel Projection Block           │
1×1 Conv → unified [64,128, 256, 512] │  
└─────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  CNN Decoder (fixed)         │  3× DecoderBlock (upsample + skip + ConvBNReLU)             │
└──────────────────────────────┘
     │
     ▼
Segmentation Map (9 classes)
```

**Key design**: The Channel Projection Block normalises all encoder outputs to the same dimensions before decoding, so the CNN decoder is identical regardless of encoder choice — enabling fair ablation comparison.

---

## Dataset — OpenEarthMap

| Property | Value |
|---|---|
| Task | Multi-class semantic segmentation |
| Classes | 9 |
| Image format | GeoTIFF (`.tif`) |
| Patch size | 1024 × 1024 px |
| Training crop | 512 × 512 px (random crop) |
| Val patches | 2,000 (fixed split) |

**Classes:**

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | Background | 5 | Tree |
| 1 | Bareland | 6 | Water |
| 2 | Rangeland | 7 | Agriculture |
| 3 | Developed | 8 | Building |
| 4 | Road | | |

---

## Ablation Experiments

| Experiment | Train images | Config |
|---|---|---|
| Luot 1 | 500 | `configs/luot1_500.yaml` |
| Luot 2 | 1,000 | `configs/luot2_1000.yaml` |
| Luot 3 | 1,500 | `configs/luot3_1500.yaml` |

All experiments use **ResNet-101** encoder with identical decoder and training settings to isolate the effect of training data size.

---

## Project Structure

```
UnetFormer Satellite Image/
├── configs/
│   ├── luot1_500.yaml       # 500-image ablation config
│   ├── luot2_1000.yaml      # 1000-image ablation config
│   └── luot3_1500.yaml      # 1500-image ablation config
├── src/
│   ├── train.py             # Main training script (DDP + AMP)
│   ├── data/
│   │   ├── dataset.py       # OpenEarthMap PyTorch Dataset
│   │   └── transforms.py    # Albumentations pipelines
│   ├── models/
│   │   └── unetformer.py    # Multi-encoder UNetFormer
│   └── utils/
│       ├── losses.py        # Cross-Entropy + Dice loss
│       ├── metrics.py       # Confusion-matrix mIoU
│       ├── callbacks.py     # Early stopping
│       └── visualizer.py    # RGB prediction visualiser
├── Tools/
│   ├── create_splits.py     # Generate train split .txt files
│   └── prepare_splits.py    # Validate & prepare val split
├── dataset/
│   └── val_2000_fixed.txt   # Pre-generated validation split
├── requirements.txt
└── run_experiment.sh        # One-command training launcher
```

---

## Requirements

- Python ≥ 3.9
- CUDA ≥ 12.8
- 2× NVIDIA GPU (tested on 2× T4 16 GB — Kaggle)

Install dependencies:

```bash
pip install -r requirements.txt
```

**Key packages:**

```
torch==2.10.0+cu128   torchvision==0.25.0+cu128
timm==1.0.26          albumentations>=1.3.1
rasterio>=1.4         pyyaml>=6.0.3
matplotlib>=3.9       einops>=0.7.0
```

---

## Usage

### On Kaggle (recommended)

```bash
# Train with 500 images
bash run_experiment.sh 1

# Train with 1000 images
bash run_experiment.sh 2

# Train with 1500 images
bash run_experiment.sh 3

# Quick smoke-test (5 iterations only)
bash run_experiment.sh 1 --dry-run
```

`run_experiment.sh` handles: pip install → split generation → distributed training automatically.

### Manual launch

```bash
# Step 1 — generate split files
python Tools/create_splits.py \
    --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
    --output-dir /kaggle/working/unetformer-openearthmap

# Step 2 — start training (2 GPUs)
torchrun --nproc_per_node=2 src/train.py --config configs/luot1_500.yaml
```

---

## Training Configuration

Key hyperparameters (identical across all 3 experiments):

| Parameter | Value |
|---|---|
| Encoder | ResNet-101 (ImageNet pretrained) |
| Optimizer | AdamW |
| Base LR | 6e-4 |
| LR schedule | Polynomial decay (power=0.9) |
| Warmup | 500 iterations |
| Batch size | 2 per GPU × 2 GPUs = **4 total** |
| Max iterations | 40,000 |
| Validation interval | every 4,000 iterations |
| Early stopping patience | 3 |
| Gradient clipping | 5.0 |
| Mixed precision | AMP (FP16) |
| Distributed training | DDP via `torchrun` |

---

## Output Artifacts

After training, results are saved to `work_dirs/<experiment>/`:

```
work_dirs/luot1_500/
├── best_model.pth              # Best checkpoint (highest val mIoU)
├── checkpoint_iter012000.pth   # Periodic checkpoint (every 12k iters)
├── benchmark_results.csv       # Validation metrics per checkpoint
├── learning_curves.png         # mIoU + val loss over iterations
├── per_class_iou_best.png      # Per-class IoU bar chart at best epoch
└── vis/
    └── iter004000_s0.png       # Prediction visualisations (RGB | GT | Pred)
```

---

## Estimated Training Time (2× T4 on Kaggle)

| Experiment | ~Iterations | ~Duration |
|---|---|---|
| Luot 1 (500 imgs) | 20,000–24,000 | ~1.8 h |
| Luot 2 (1,000 imgs) | 28,000–36,000 | ~2.5 h |
| Luot 3 (1,500 imgs) | 36,000–40,000 | ~3.2 h |
| **Total (sequential)** | | **~7.5–8 h** |

Early stopping (patience=3) may reduce total time significantly for smaller datasets.

---

## License

This project is for academic and research purposes.
