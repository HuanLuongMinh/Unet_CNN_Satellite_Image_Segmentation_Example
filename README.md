# UNet CNN — Satellite Image Semantic Segmentation

An ablation study comparing encoder/decoder/loss choices on the [OpenEarthMap](https://open-earth-map.org/) dataset.

Năm track thí nghiệm hoàn toàn độc lập về code (không file nào bị chia sẻ/sửa đổi chéo) cùng tồn tại trong repo:

| Track | Encoder | Decoder | Loss | Base LR | Train script |
|---|---|---|---|---|---|
| **A — baseline (gốc)** | ResNet-101 (ImageNet) | CNN (Channel Projection + DecoderBlock) | CE + Dice | 6e-4 | `src/train_cnn.py` |
| **B — CNN/ResNeXt101_32x16d** | ResNeXt-101_32x16d (SWSL, ~1B ảnh pretrain) | CNN (giống Track A) | CE thuần | 1e-5 | `src/train_cnn_resnext101_32.py` |
| **C — UNetFormer/ResNeXt101_32x16d** | ResNeXt-101_32x16d (SWSL) | UNetFormer (Global-Local Attention transformer) | CE thuần | 1e-5 | `src/train_unet_former_reshnet101_32.py` |
| **D — UNetFormer + CombinedLoss** | ResNeXt-101_32x16d (SWSL) | UNetFormer (giống Track C) | CE + Dice (CombinedLoss) | 1e-5 | `src/train_unet_former_reshnet101_32_combineLoss.py` |
| **E — UNetFormer + DAPCN (mới)** | ResNeXt-101_32x16d (SWSL) | UNetFormer (giống Track C, decode_channels=256) | CE + Boundary + DAPG (DAPCN) | 1e-5 | `src/train_unetformer_reshnet101_32_dapcn.py` |

Track B/C/D/E chỉ thay đổi đúng các biến nêu trên so với baseline tương ứng, **giữ nguyên dữ liệu, batch size, số iteration, seed, early-stopping patience** — để so sánh công bằng. Track D so với Track C chỉ khác duy nhất loss function (CombinedLoss thay vì CE thuần); kiến trúc model là bản copy 1:1 (`src/models/unet_former_reshnet101_32_combineLoss.py` ≡ `src/models/unet_former_reshnet101_32.py`). Track E thêm cơ chế DAPCN (Dynamic Anchor Prototype Grouping + Boundary loss) vào decoder UNetFormer — KHÔNG có contrastive loss (đã loại bỏ có chủ ý khỏi bản port), và dùng `decode_channels=256` (rộng hơn Track C/D) theo đúng thiết kế DAPCN gốc.

---

## Architecture

### Track A & B — CNN Decoder

```
Input (1024×1024, train crop 512×512)
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│  Encoder (theo track)                                          │
│  Track A — ResNet-101        → channels [256, 512, 1024, 2048] │
│  Track B — ResNeXt-101_32x16d → channels [256, 512, 1024, 2048]│
│  (ResNet-18 / MiT-B0 cũng được hỗ trợ trong unetcnn.py gốc)    │
└────────────────────────────────────────────────────────────────┘
     │  4 multi-scale feature maps (stride 4→32)
     ▼
┌─────────────────────────────────────┐
│  Channel Projection Block           │
│  1×1 Conv → unified [64,128,256,512]│
└─────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  CNN Decoder (fixed)         │  3× DecoderBlock (upsample + skip + ConvBNReLU) │
└──────────────────────────────┘
     │
     ▼
Segmentation Map (9 classes)
```

**Key design**: Channel Projection Block chuẩn hóa output encoder về cùng kích thước trước khi decode, nên CNN decoder giống nhau tuyệt đối giữa Track A và Track B — đảm bảo so sánh công bằng khi đổi encoder.

### Track C & D — UNetFormer Decoder

```
Input (1024×1024, train crop 512×512)
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│  Encoder: ResNeXt-101_32x16d (SWSL) → channels [256,512,1024,2048] │
└────────────────────────────────────────────────────────────────┘
     │  4 multi-scale feature maps (stride 4→32)
     ▼
┌──────────────────────────────────────────────────────────────┐
│  UNetFormer Decoder (Global-Local Attention, paper-faithful) │
│  Stage 4 (deepest): Conv1x1 → Block (GLA self-attention)     │
│  Stage 3: Weighted Fusion (skip) → Block (GLA)               │
│  Stage 2: Weighted Fusion (skip) → Block (GLA)                │
│  Stage 1: FeatureRefinementHead (spatial + channel attention) │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
Segmentation Map (9 classes)
```

**Key design**: Decoder port từ kiến trúc UNetFormer gốc — không có DAPCN (không boundary loss, không prototype/contrastive learning, không dynamic anchor point), chỉ gồm `GlobalLocalAttention` + `Block` (MLP) + `WF` (Weighted Fusion) + `FeatureRefinementHead`. Kiến trúc giống tuyệt đối giữa Track C và Track D — chỉ khác loss function dùng để train.

### Track E — UNetFormer + DAPCN Decoder (mới)

```
Input (1024×1024, train crop 512×512)
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│  Encoder: ResNeXt-101_32x16d (SWSL) → channels [256,512,1024,2048] │
└────────────────────────────────────────────────────────────────┘
     │  4 multi-scale feature maps (stride 4→32)
     ▼
┌────────────────────────────────────────────────────────────────┐
│  UNetFormer Decoder (GLA, decode_channels=256 — rộng hơn Track C/D) │
│  Stage 4→1: giống Track C/D (Conv1x1/WF → Block GLA → FRH)     │
└────────────────────────────────────────────────────────────────┘
     │  fused feature (256-dim, sau FeatureRefinementHead)
     ├──────────────► Classifier (Conv1x1) ──► logits ──► Segmentation Map
     │
     ├──► DynamicAnchorModule (học prototype + EM refinement)
     │        └─► DAPGLoss (Dynamic Anchor Prototype Grouping) = Loss DAPCN
     │
     └──► extract_boundary_map(logits) vs compute_boundary_gt(mask)
              └─► Binary Cross-Entropy = Loss Boundary

Train: tổng loss = CE(logits, mask) + Loss Boundary + Loss DAPCN
Eval : chỉ forward tới logits (không tính 2 loss phụ trợ)
```

**Key design**: Decoder GLA giống Track C/D nhưng `decode_channels=256` (4× rộng hơn) theo đúng thiết kế DAPCN gốc — fused feature sau `FeatureRefinementHead` được dùng đồng thời cho classifier **và** 2 nhánh auxiliary loss. `DAPGLoss`/`DynamicAnchorModule` (điều khiển bởi `proto_lambda`) là **Loss DAPCN thật** — **không phải** contrastive loss (contrastive đã bị loại bỏ hoàn toàn, không tồn tại trong code). Model `forward(x, gt=None)` trả `logits` khi eval (`gt=None`), trả `(logits, aux_loss_dict)` khi train.

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

Mỗi track có 3 lượt train (500 / 1,000 / 1,500 ảnh), dùng **chung 1 seed (42)** trong mỗi track (model init + sampler shuffle) — chỉ khác lượng dữ liệu train, để cô lập đúng biến đang khảo sát.

### Track A — ResNet-101 baseline (gốc)

| Experiment | Train images | Config |
|---|---|---|
| Luot 1 | 500 | `configs/cnn/luot1_500.yaml` |
| Luot 2 | 1,000 | `configs/cnn/luot2_1000.yaml` |
| Luot 3 | 1,500 | `configs/cnn/luot3_1500.yaml` |

### Track B — CNN decoder, ResNeXt-101_32x16d, CE thuần, lr=1e-5

| Experiment | Train images | Config |
|---|---|---|
| Luot 1 | 500 | `configs/cnn_reshnet101_32/luot1_500.yaml` |
| Luot 2 | 1,000 | `configs/cnn_reshnet101_32/luot2_1000.yaml` |
| Luot 3 | 1,500 | `configs/cnn_reshnet101_32/luot3_1500.yaml` |

### Track C — UNetFormer decoder, ResNeXt-101_32x16d, CE thuần, lr=1e-5

| Experiment | Train images | Config |
|---|---|---|
| Luot 1 | 500 | `configs/unet_former_reshnet101_32/luot1_500.yaml` |
| Luot 2 | 1,000 | `configs/unet_former_reshnet101_32/luot2_1000.yaml` |
| Luot 3 | 1,500 | `configs/unet_former_reshnet101_32/luot3_1500.yaml` |

### Track D — UNetFormer decoder, ResNeXt-101_32x16d, CombinedLoss (CE+Dice), lr=1e-5

| Experiment | Train images | Config |
|---|---|---|
| Luot 1 | 500 | `configs/unet_former_reshnet101_32_combineLoss/luot1_500.yaml` |
| Luot 2 | 1,000 | `configs/unet_former_reshnet101_32_combineLoss/luot2_1000.yaml` |
| Luot 3 | 1,500 | `configs/unet_former_reshnet101_32_combineLoss/luot3_1500.yaml` |

CombinedLoss có thể chỉnh trọng số qua section `LOSS` trong config:

```yaml
LOSS:
  TYPE: "CombinedLoss"
  CE_WEIGHT: 1.0
  DICE_WEIGHT: 1.0
```

### Track E — UNetFormer decoder + DAPCN, ResNeXt-101_32x16d, CE+Boundary+DAPG, lr=1e-5 (mới)

| Experiment | Train images | Config |
|---|---|---|
| Luot 1 | 500 | `configs/unetformer_reshnet101_32_dapcn/luot1_500.yaml` |
| Luot 2 | 1,000 | `configs/unetformer_reshnet101_32_dapcn/luot2_1000.yaml` |
| Luot 3 | 1,500 | `configs/unetformer_reshnet101_32_dapcn/luot3_1500.yaml` |

DAPCN có section riêng `DAPCN` trong config để chỉnh siêu tham số boundary/DAPG (giữ nguyên giá trị gốc, không phải biến ablation):

```yaml
DAPCN:
  BOUNDARY_LAMBDA: 0.15         # trọng số Loss Boundary
  PROTO_LAMBDA: 0.1             # trọng số Loss DAPCN (DAPG)
  BOUNDARY_MODE: "sobel"
  DA_MAX_GROUPS: 64
  DA_TEMPERATURE: 0.5
  DA_NUM_ITERS: 3
  DAPG_MARGIN: 0.3
  DAPG_LAMBDA_INTER: 0.5
  DAPG_LAMBDA_QUALITY: 0.1
  IGNORE_INDEX: 255
```

---

## Project Structure

```
UnetFormer Satellite Image/
├── configs/
│   ├── cnn/                                      # Track A configs
│   ├── cnn_reshnet101_32/                        # Track B configs
│   ├── unet_former_reshnet101_32/                # Track C configs
│   ├── unet_former_reshnet101_32_combineLoss/    # Track D configs
│   │   ├── luot1_500.yaml
│   │   ├── luot2_1000.yaml
│   │   └── luot3_1500.yaml
│   └── unetformer_reshnet101_32_dapcn/           # Track E configs (mới)
│       ├── luot1_500.yaml
│       ├── luot2_1000.yaml
│       └── luot3_1500.yaml
├── src/
│   ├── train_cnn.py                                      # Track A training script (DDP + AMP)
│   ├── train_cnn_resnext101_32.py                        # Track B training script (DDP + AMP + checkpoint/resume)
│   ├── train_unet_former_reshnet101_32.py                # Track C training script (DDP + AMP + checkpoint/resume)
│   ├── train_unet_former_reshnet101_32_combineLoss.py    # Track D training script (giống Track C, đổi loss)
│   ├── train_unetformer_reshnet101_32_dapcn.py           # Track E training script (mới — giống Track C, + aux loss)
│   ├── data/
│   │   ├── dataset.py                    # OpenEarthMap PyTorch Dataset
│   │   └── transforms.py                 # Albumentations pipelines
│   ├── models/
│   │   ├── unetcnn.py                                  # Track A: multi-encoder CNN U-Net (resnet18/resnet101/mit_b0)
│   │   ├── unetcnn_resnext101_32.py                    # Track B: CNN U-Net với encoder resnext101_32x16d
│   │   ├── unet_former_reshnet101_32.py                # Track C: UNetFormer (GLA transformer decoder)
│   │   ├── unet_former_reshnet101_32_combineLoss.py    # Track D: kiến trúc giống 100% Track C
│   │   └── unetformer_reshnet101_32_dapcn.py           # Track E: UNetFormer + DAPCN (boundary + DAPG)
│   └── utils/
│       ├── losses.py                     # CrossEntropyLoss thuần + DiceLoss + CombinedLoss (CE+Dice)
│       ├── metrics.py                    # Confusion-matrix mIoU
│       ├── callbacks.py                  # Early stopping
│       └── visualizer.py                 # RGB prediction visualiser
├── Tools/
│   ├── create_splits.py                  # Generate train split .txt files
│   ├── prepare_splits.py                 # Validate & prepare val split
│   └── get_resume_checkpoint.py          # Xác định/upload checkpoint để resume (Track B)
├── dataset/
│   └── val_2000_fixed.txt                # Pre-generated validation split
├── requirements.txt
├── run_cnn_experiment.sh                                   # Launcher Track A
├── run_cnn_reshnet101_32_experiment.sh                     # Launcher Track B
├── run_unet_former_reshnet101_32_experiment.sh             # Launcher Track C
├── run_unet_former_reshnet101_32_combineLoss_experiment.sh # Launcher Track D
├── run_unetformer_reshnet101_32_dapcn_experiment.sh        # Launcher Track E (mới)
├── run_all.sh                             # Chạy tuần tự cả 3 lượt Track A
├── resume_checkpoint.sh                   # Resume Track B sau khi session bị ngắt
└── read_resume.md                         # Hướng dẫn resume Track B chi tiết
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
torch==2.10.0+cu128   torchvision==0.25.0+cu128   (pre-installed trên Kaggle, không cài lại)
timm==1.0.26           albumentations>=1.3.1
rasterio>=1.4          einops>=0.7.0
```

---

## Usage

### Track A — ResNet-101 baseline

```bash
# Train with 500 / 1000 / 1500 images
bash run_cnn_experiment.sh 1
bash run_cnn_experiment.sh 2
bash run_cnn_experiment.sh 3

# Quick smoke-test (5 iterations only)
bash run_cnn_experiment.sh 1 --dry-run
```

Manual launch:

```bash
python Tools/create_splits.py \
    --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
    --output-dir /kaggle/working/unetcnn-openearthmap

torchrun --nproc_per_node=2 src/train_cnn.py --config configs/cnn/luot1_500.yaml
```

### Track B — CNN decoder, ResNeXt-101_32x16d

```bash
bash run_cnn_reshnet101_32_experiment.sh 1
bash run_cnn_reshnet101_32_experiment.sh 2
bash run_cnn_reshnet101_32_experiment.sh 3

# Quick smoke-test (5 iterations only)
bash run_cnn_reshnet101_32_experiment.sh 1 --dry-run
```

Manual launch:

```bash
python Tools/create_splits.py \
    --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
    --output-dir /kaggle/working/unetcnn-resnext101-openearthmap

torchrun --nproc_per_node=2 src/train_cnn_resnext101_32.py --config configs/cnn_reshnet101_32/luot1_500.yaml
```

**Nếu Kaggle session bị ngắt giữa lúc train Track B**, dùng `resume_checkpoint.sh` để tiếp tục từ `latest_checkpoint.pth` (lưu sau mỗi lần validation) — xem chi tiết tại [read_resume.md](read_resume.md):

```bash
bash resume_checkpoint.sh 1
# hoặc bỏ qua câu hỏi nếu đã biết đường dẫn checkpoint:
bash resume_checkpoint.sh 1 --path /kaggle/working/.../latest_checkpoint.pth
```

### Track C — UNetFormer decoder, ResNeXt-101_32x16d, CE thuần

```bash
bash run_unet_former_reshnet101_32_experiment.sh 1
bash run_unet_former_reshnet101_32_experiment.sh 2
bash run_unet_former_reshnet101_32_experiment.sh 3

# Quick smoke-test (5 iterations only)
bash run_unet_former_reshnet101_32_experiment.sh 1 --dry-run
```

Manual launch (hỗ trợ `--resume <work_dir>/latest_checkpoint.pth` trực tiếp trên train script, chưa có wrapper `resume_checkpoint.sh` riêng cho track này):

```bash
python Tools/create_splits.py \
    --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
    --output-dir /kaggle/working/unetformer-resnext101-openearthmap

torchrun --nproc_per_node=2 src/train_unet_former_reshnet101_32.py --config configs/unet_former_reshnet101_32/luot1_500.yaml
```

### Track D — UNetFormer decoder + CombinedLoss (CE+Dice)

```bash
bash run_unet_former_reshnet101_32_combineLoss_experiment.sh 1
bash run_unet_former_reshnet101_32_combineLoss_experiment.sh 2
bash run_unet_former_reshnet101_32_combineLoss_experiment.sh 3

# Quick smoke-test (5 iterations only)
bash run_unet_former_reshnet101_32_combineLoss_experiment.sh 1 --dry-run
```

Manual launch (hỗ trợ `--resume <work_dir>/latest_checkpoint.pth` trực tiếp trên train script):

```bash
python Tools/create_splits.py \
    --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
    --output-dir /kaggle/working/unetformer-resnext101-combinedloss-openearthmap

torchrun --nproc_per_node=2 src/train_unet_former_reshnet101_32_combineLoss.py --config configs/unet_former_reshnet101_32_combineLoss/luot1_500.yaml
```

### Track E — UNetFormer decoder + DAPCN (CE+Boundary+DAPG) (mới)

```bash
bash run_unetformer_reshnet101_32_dapcn_experiment.sh 1
bash run_unetformer_reshnet101_32_dapcn_experiment.sh 2
bash run_unetformer_reshnet101_32_dapcn_experiment.sh 3

# Quick smoke-test (5 iterations only)
bash run_unetformer_reshnet101_32_dapcn_experiment.sh 1 --dry-run
```

Manual launch (hỗ trợ `--resume <work_dir>/latest_checkpoint.pth` trực tiếp trên train script):

```bash
python Tools/create_splits.py \
    --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
    --output-dir /kaggle/working/unetformer-resnext101-dapcn-openearthmap

torchrun --nproc_per_node=2 src/train_unetformer_reshnet101_32_dapcn.py --config configs/unetformer_reshnet101_32_dapcn/luot1_500.yaml
```

---

## Training Configuration

| Parameter | Track A | Track B | Track C | Track D | Track E (mới) |
|---|---|---|---|---|---|
| Encoder | ResNet-101 (ImageNet) | ResNeXt-101_32x16d (SWSL) | ResNeXt-101_32x16d (SWSL) | ResNeXt-101_32x16d (SWSL) | ResNeXt-101_32x16d (SWSL) |
| Decoder | CNN | CNN (giống Track A) | UNetFormer (GLA transformer) | UNetFormer (giống Track C) | UNetFormer + DAPCN (decode_channels=256) |
| Loss | CrossEntropy + Dice | CrossEntropy thuần | CrossEntropy thuần | CrossEntropy + Dice (CombinedLoss) | CrossEntropy + Boundary + DAPG (DAPCN) |
| Optimizer | AdamW | AdamW | AdamW | AdamW | AdamW |
| Base LR | 6e-4 | 1e-5 | 1e-5 | 1e-5 | 1e-5 |
| LR schedule | Polynomial decay (power=0.9) | Polynomial decay (power=0.9) | Polynomial decay (power=0.9) | Polynomial decay (power=0.9) | Polynomial decay (power=0.9) |
| Warmup | 500 iterations | 500 iterations | 500 iterations | 500 iterations | 500 iterations |
| Seed | 42 (chung cho cả 3 lượt) | 42 (chung cho cả 3 lượt) | 42 (chung cho cả 3 lượt) | 42 (chung cho cả 3 lượt) | 42 (chung cho cả 3 lượt) |
| Batch size | 2/GPU × 2 GPU = 4 total | 2/GPU × 2 GPU = 4 total | 2/GPU × 2 GPU = 4 total | 2/GPU × 2 GPU = 4 total | 2/GPU × 2 GPU = 4 total |
| Max iterations | 40,000 | 40,000 | 40,000 | 40,000 | 40,000 |
| Validation interval | every 4,000 iterations | every 4,000 iterations | every 4,000 iterations | every 4,000 iterations | every 4,000 iterations |
| Early stopping patience | 4 | 4 | 4 | 4 | 4 |
| Gradient clipping | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| Mixed precision | AMP (FP16) | AMP (FP16) | AMP (FP16) | AMP (FP16) | AMP (FP16) |
| Distributed training | DDP via `torchrun` | DDP via `torchrun` | DDP via `torchrun` | DDP via `torchrun` | DDP via `torchrun` |
| Checkpoint/resume | mỗi 12,000 iter (chỉ trọng số) | mỗi lần validation (đầy đủ model+optimizer+scaler+early-stopping, hỗ trợ `--resume`) | mỗi lần validation (giống Track B) | mỗi lần validation (giống Track B/C) | mỗi lần validation (giống Track B/C/D) |

---

## Output Artifacts

### Track A — `work_dirs/<experiment>/`

```
work_dirs/luot1_500/
├── best_model.pth              # Best checkpoint (highest val mIoU)
├── checkpoint_iter012000.pth   # Periodic checkpoint (every 12k iters, weights only)
├── benchmark_results.csv       # Validation metrics per checkpoint
├── learning_curves.png         # mIoU + val loss over iterations
├── per_class_iou_best.png      # Per-class IoU bar chart at best epoch
└── vis/
    └── iter004000_s0.png       # Prediction visualisations (RGB | GT | Pred)
```

### Track B / C / D / E — `work_dirs/<experiment>/`

```
work_dirs/luot1_500/
├── best_model.pth              # Best checkpoint (ghi đè ngay khi có mIoU mới tốt nhất)
├── latest_checkpoint.pth       # Checkpoint đầy đủ (model+optimizer+scaler+early-stopping),
│                                 ghi đè sau MỖI lần validation — dùng để --resume
├── benchmark_results.csv       # Ghi tăng dần sau mỗi validation, có cột is_best (True/False)
├── learning_curves.png         # mIoU + val loss over iterations
├── per_class_iou_best.png      # Per-class IoU bar chart at best epoch
└── vis/
    └── iter004000_s0.png       # Prediction visualisations (RGB | GT | Pred)
```

Track B/C/D/E không có file `checkpoint_iter*.pth` định kỳ như Track A — đã thay bằng `latest_checkpoint.pth` ghi đè mỗi validation (đỡ tốn dung lượng, đủ để resume).

---

## Estimated Training Time (2× T4 on Kaggle)

### Track A — ResNet-101

| Experiment | ~Iterations | ~Duration |
|---|---|---|
| Luot 1 (500 imgs) | 20,000–24,000 | ~1.8 h |
| Luot 2 (1,000 imgs) | 28,000–36,000 | ~2.5 h |
| Luot 3 (1,500 imgs) | 36,000–40,000 | ~3.2 h |
| **Total (sequential)** | | **~7.5–8 h** |

Early stopping (patience=4) có thể giảm đáng kể thời gian với dataset nhỏ.

### Track B — CNN decoder, ResNeXt-101_32x16d (ước tính, chưa đo thực tế)

Encoder nặng hơn ~3.5–4.5× (FLOPs/throughput) so với ResNet-101, và lr=1e-5 thấp hơn nhiều nên khó early-stop sớm như Track A — nhiều khả năng cả 3 lượt chạy gần hết 40,000 iter.

| Experiment | ~Duration |
|---|---|
| Mỗi lượt (500/1000/1500) | ~11–15 h |
| **Total (3 lượt, tuần tự)** | **~34–44 h** |

### Track C & D — UNetFormer decoder, ResNeXt-101_32x16d (ước tính, chưa đo thực tế)

Decoder transformer (Global-Local Attention) nặng hơn CNN decoder của Track B do thêm self-attention theo window; Track D có thêm bước tính `DiceLoss` mỗi iteration (rẻ hơn nhiều so với forward/backward, không ảnh hưởng đáng kể tốc độ so với Track C). Cùng lr=1e-5 nên cũng khó early-stop sớm.

| Experiment | ~Duration |
|---|---|
| Mỗi lượt (500/1000/1500) | ~12–17 h |
| **Total (3 lượt, tuần tự, mỗi track)** | **~36–48 h** |

### Track E — UNetFormer + DAPCN, ResNeXt-101_32x16d (ước tính, chưa đo thực tế)

`decode_channels=256` (4× rộng hơn Track C/D) khiến decoder nặng hơn đáng kể; cộng thêm `DynamicAnchorModule` (EM refinement 3 iterations) + `DAPGLoss` + tính boundary map mỗi iteration — tổng overhead cao hơn Track C/D, dù vẫn rẻ hơn nhiều so với forward/backward của encoder. Cùng lr=1e-5 nên cũng khó early-stop sớm.

| Experiment | ~Duration |
|---|---|
| Mỗi lượt (500/1000/1500) | ~14–20 h |
| **Total (3 lượt, tuần tự)** | **~42–60 h** |

⚠️ Mỗi lượt (Track B/C/D/E) có thể **vượt giới hạn session GPU ~9h của Kaggle** — dùng `--resume <work_dir>/latest_checkpoint.pth` trên train script tương ứng để tiếp tục sau khi session bị ngắt (Track B có thêm wrapper `resume_checkpoint.sh`, xem [read_resume.md](read_resume.md); Track C/D/E resume thủ công qua `torchrun ... --resume ...`). Khuyến nghị chạy `--dry-run` trước để đo tốc độ thực tế và hiệu chỉnh lại ước tính này.

---

## License

This project is for academic and research purposes.
